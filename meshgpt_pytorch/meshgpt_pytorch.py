import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple

from einops import rearrange, repeat, reduce
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
        num_quantizers = 2,         # or 'D' in the paper
        codebook_size = 16384,      # they use 16k, shared codebook between layers
        rq_kwargs: dict = dict()
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
            **rq_kwargs
        )

        self.project_codebook_out = nn.Linear(dim_codebook * 3, dim)

        self.decoders = ModuleList([])

        for _ in range(decoder_depth):
            sage_conv = SAGEConv(dim, dim)

            self.decoders.append(sage_conv)

        self.to_coor_logits = nn.Sequential(
            nn.Linear(dim, num_discrete_coors * 9),
            Rearrange('... (v c) -> ... v c', v = 9)
        )

    @beartype
    def encode(
        self,
        *,
        vertices:   TensorType['b', 'nv', 3, int],
        faces:      TensorType['b', 'nf', 3, int],
        face_edges: TensorType['b', 2, 'e', int]
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

        face_edges += batch_offset
        face_edges = rearrange(face_edges, 'b ij e -> ij (b e)')

        x = rearrange(face_embed, 'b nf d -> (b nf) d')

        for conv in self.encoders:
            x = conv(x, face_edges)

        x = rearrange(x, '(b nf) d -> b nf d', b = batch)

        return x

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

        num = vertices.scatter_add(-2, faces_with_dim, face_embed)
        den = torch.zeros_like(vertices).scatter_add(-2, faces, torch.ones_like(face_embed))

        averaged_vertices = num / den.clamp(min = 1e-5)

        # residual VQ

        quantized, codes, commit_loss = self.quantizer(averaged_vertices)

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = quantized.gather(-2, faces_with_dim)
        face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, faces_with_quantized_dim)

        return face_embed_output, codes_output, commit_loss

    @beartype
    def decode(
        self,
        codes
    ):
        raise NotImplementedError

    @beartype
    def decode_from_codes_to_vertices(
        self,
        codes: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @beartype
    def forward(
        self,
        *,
        vertices: Tensor,
        faces: Tensor,
        face_edges: Tensor,
        return_quantized = False
    ):
        discretized_vertices = discretize_coors(
            vertices,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range,
        )

        encoded = self.encode(
            vertices = discretized_vertices,
            faces = faces,
            face_edges = face_edges
        )

        quantized, codes, commit_loss = self.quantize(
            face_embed = encoded,
            faces = faces
        )

        if return_quantized:
            return quantized

        decode = self.decode(quantized)

        pred_coor_bins = self.to_coor_logits(decode)

        return loss

class MeshGPT(Module):
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
        self.num_quantizers = autoncoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token_id = self.codebook_size + 1

        # they use axial positional embeddings

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)
        self.quantize_level_embed = nn.Embedding(self.num_quantizers, dim)
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        # main autoregressive attention network

        self.decoder = Decoder(
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
        c
    ):
        seq_len, device = x.shape[-2], device
        assert divisible_by(seq_len, self.num_quantizers) == 0

        seq_arange = torch.arange(seq_len, device = device)

        # codebook embed + absolute positions

        x = self.token_embed(codes)
        x = x + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        level_embed = repeat('n d -> (r n) d', r = seq_len // self.num_quantizers)
        x = x + level_embed

        # attention

        x = self.decoder(x)

        # logits

        logits = self.to_logits
        return logits
