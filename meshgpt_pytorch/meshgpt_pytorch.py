from math import ceil

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import Decoder

from x_transformers.attend import Attend

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from ema_pytorch import EMA

from torch_geometric.nn.conv import SAGEConv

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

# tensor helper functions

@beartype
def discretize_coors(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@beartype
def undiscretize_coors(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@beartype
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, channels, _, device = *t.shape, t.device

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = torch.float, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)
    return F.conv1d(t, kernel, padding = half_width, groups = channels)

@beartype
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)

# resnet block

class Block(Module):
    def __init__(self, dim, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        *,
        groups = 8
    ):
        super().__init__()
        self.block1 = Block(dim, groups = groups)
        self.block2 = Block(dim, groups = groups)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + x

# main classes

class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        dim,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        encoder_depth = 2,
        decoder_depth = 2,
        dim_codebook = 192,
        num_quantizers = 2,           # or 'D' in the paper
        codebook_size = 16384,        # they use 16k, shared codebook between layers
        rq_kwargs: dict = dict(),
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions
    ):
        super().__init__()

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)
        self.project_in = nn.Linear(dim_coor_embed * 9, dim)

        self.encoders = ModuleList([])

        for _ in range(encoder_depth):
            sage_conv = SAGEConv(dim, dim)

            self.encoders.append(sage_conv)

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.project_dim_codebook = nn.Linear(dim, dim_codebook * 3)

        self.quantizer = ResidualVQ(
            dim = dim_codebook,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            shared_codebook = True,
            commitment_weight = 1.,
            **rq_kwargs
        )

        self.project_codebook_out = nn.Linear(dim_codebook * 3, dim)

        self.decoders = ModuleList([])

        for _ in range(decoder_depth):
            resnet_block = ResnetBlock(dim)
            self.decoders.append(resnet_block)

        self.to_coor_logits = nn.Sequential(
            nn.Linear(dim, num_discrete_coors * 9),
            Rearrange('... (v c) -> ... v c', v = 9)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @beartype
    def encode(
        self,
        *,
        vertices:   TensorType['b', 'nv', 3, int],
        faces:      TensorType['b', 'nf', 3, int],
        face_edges: TensorType['b', 2, 'e', int],
        return_face_coordinates = False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        c - coordinates (3)
        d - embed dim
        """

        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        _, num_faces, _ = faces.shape

        faces_vertices = repeat(faces, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)

        face_coords = vertices.gather(-2, faces_vertices)
        face_coords = rearrange(face_coords, 'b nf nv c -> b nf (nv c)') # 9 coordinates per face

        face_embed = self.coor_embed(face_coords)
        face_embed = rearrange(face_embed, 'b nf c d -> b nf (c d)')

        face_embed = self.project_in(face_embed)

        batch_arange = torch.arange(batch, device = device)
        batch_offset = batch_arange * num_faces
        batch_offset = rearrange(batch_offset, 'b -> b 1 1')

        face_edges = face_edges + batch_offset
        face_edges = rearrange(face_edges, 'b ij e -> ij (b e)')

        x = rearrange(face_embed, 'b nf d -> (b nf) d')

        for conv in self.encoders:
            x = conv(x, face_edges)

        x = rearrange(x, '(b nf) d -> b nf d', b = batch)

        if not return_face_coordinates:
            return x

        return x, face_coords

    @beartype
    def quantize(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_embed: TensorType['b', 'nf', 'd', float],
    ):
        batch, device = faces.shape[0], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nv d) -> b nf nv d', nv = 3)

        vertex_dim = face_embed.shape[-1]
        faces_with_dim = repeat(faces, 'b nf nv -> b nf nv d', d = vertex_dim)

        faces_with_dim = rearrange(faces_with_dim, 'b ... d -> b (...) d')
        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # residual VQ

        quantized, codes, commit_loss = self.quantizer(averaged_vertices)

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = quantized.gather(-2, faces_with_dim)
        face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        face_embed_output = self.project_codebook_out(face_embed_output)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, faces_with_quantized_dim)

        return face_embed_output, codes_output, commit_loss

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float]
    ):
        quantized = rearrange(quantized, 'b n d -> b d n')

        x = quantized

        for resnet_block in self.decoders:
            x = resnet_block(x)

        return rearrange(x, 'b d n -> b n d')

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        return_discrete_codes = False
    ):

        # handle different code shapes

        codes = rearrange(codes, 'b ... -> b (...)')
        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        face_embed_output = self.project_codebook_out(quantized)
        decoded = self.decode(face_embed_output)

        pred_face_coords = self.to_coor_logits(decoded)
        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = 3)

        # back to continuous space

        continuous_coors = undiscretize_coors(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        if not return_discrete_codes:
            return continuous_coors

        return continuous_coors, pred_face_coords

    def tokenize(self, *args, **kwargs):
        assert 'return_codes' not in kwargs
        return self.forward(*args, return_codes = True, **kwargs)

    @beartype
    def forward(
        self,
        *,
        vertices: TensorType['b', 'nv', 3, float],
        faces: TensorType['b', 'nf', 3, int],
        face_edges: TensorType['b', 2, 'ij', int],
        return_codes = False,
        return_loss_breakdown = False
    ):
        discretized_vertices = discretize_coors(
            vertices,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range,
        )

        encoded, face_coordinates = self.encode(
            vertices = discretized_vertices,
            faces = faces,
            face_edges = face_edges,
            return_face_coordinates = True
        )

        quantized, codes, commit_loss = self.quantize(
            face_embed = encoded,
            faces = faces
        )

        if return_codes:
            return codes

        decode = self.decode(quantized)

        pred_face_coords = self.to_coor_logits(decode)

        # prepare for recon loss

        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma)

            # cross entropy with localized smoothing

            recon_loss = (-target_one_hot * pred_log_prob).sum(dim = 1).mean()

        # calculate total loss

        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (recon_loss, commit_loss)

        return recon_loss, loss_breakdown

class MeshTransformer(Module):
    @beartype
    def __init__(
        self,
        autoencoder: MeshAutoencoder,
        *,
        dim = 512,
        max_seq_len = 8192,
        attn_num_tokens = 128 ** 2,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(),
        ignore_index = -100
    ):
        super().__init__()

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token_id = self.codebook_size

        # they use axial positional embeddings

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)
        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        # main autoregressive attention network

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            **attn_kwargs
        )

        self.to_logits = nn.Linear(dim, self.codebook_size + 1)

    def generate(self):
        return self

    def forward(
        self,
        codes,
        return_loss = True
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        batch, seq_len, device = *codes.shape, codes.device
        assert divisible_by(seq_len, self.num_quantizers)

        if return_loss:
            codes, labels = codes[:, :-1], codes

        codes = self.token_embed(codes)

        # auto append sos token

        sos = repeat(self.sos_token, 'd -> b d', b = batch)
        codes, _ = pack([sos, codes], 'b * d')

        # codebook embed + absolute positions

        seq_arange = torch.arange(seq_len, device = device)

        codes = codes + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        level_embed = repeat(self.quantize_level_embed, 'n d -> (r n) d', r = seq_len // self.num_quantizers)
        codes = codes + level_embed

        # attention

        attended = self.decoder(codes)

        # logits

        logits = self.to_logits(attended)

        if self.num_quantizers > 1:
            eos_mask = ((torch.arange(seq_len, device = device) - 1)  % self.num_quantizers == 0)
            eos_mask = rearrange(eos_mask, 'n -> 1 n 1')
            eos_mask = F.pad(eos_mask, (logits.shape[-1] - 1, 0), value = False)
            logits = logits.masked_fill(eos_mask, -torch.finfo(logits.dtype).max)

        if not return_loss:
            return logits

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels
        )

        return ce_loss
