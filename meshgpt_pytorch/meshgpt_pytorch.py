from pathlib import Path
from functools import partial
from math import ceil, pi

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple, Callable, Optional, List, Dict

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import Decoder
from x_transformers.attend import Attend
from x_transformers.x_transformers import RMSNorm, LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.version import __version__

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from torch_geometric.nn.conv import SAGEConv

from gateloop_transformer import SimpleGateLoopLayer

from tqdm import tqdm
from packaging import version

import pickle

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_empty(l):
    return len(l) == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

# continuous embed

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )

# additional encoder features
# 1. angle (3), 2. area (1), 3. normals (3)

def derive_angle(x, y, eps = 1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()

@torch.no_grad()
def get_derived_face_features(
    face_coords: TensorType['b', 'nf', 3, 3, float]  # 3 vertices with 3 coordinates
):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2)

    angles  = derive_angle(face_coords, shifted_face_coords)

    edge1, edge2, _ = (face_coords - shifted_face_coords).unbind(dim = 2)

    normals = l2norm(torch.cross(edge1, edge2, dim = -1))
    area = normals.norm(dim = -1, keepdim = True) * 0.5

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )   

# tensor helper functions

@beartype
def discretize(
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
def undiscretize(
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
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
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

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

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

    def forward(
        self,
        x,
        mask = None
    ):
        h = self.block1(x, mask = mask)
        h = self.block2(h, mask = mask)
        return h + x

# linear attention

class LinearAttention(Module):
    """
    using the specific linear attention proposed by El-Nouby et al. (https://arxiv.org/abs/2106.09681)
    """

    @beartype
    def __init__(
        self,
        dim,
        *,
        dim_head = 32,
        heads = 8,
        scale = 8,
        flash = False,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv = 3, h = heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(
            scale = scale,
            causal = False,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    def forward(
        self,
        x,
        mask = None,
        channel_first = True
    ):
        if channel_first:
            x = rearrange(x, 'b d n -> b n d')

        x = self.norm(x)
        q, k, v = self.to_qkv(x)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 1 n')
            q, k, v = map(lambda t: t.masked_fill(~mask, 0.), (q, k, v))

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        if exists(mask):
            q, k, v = map(lambda t: t.masked_fill(~mask, 0.), (q, k, v))

        out, *_ = self.attend(q, k, v)

        out = self.to_out(out)

        if channel_first:
            out = rearrange(out, 'b n d -> b d n')

        return out

# main classes

class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        dim,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_area = 128,
        dim_area_embed = 16,
        num_discrete_normals = 128,
        dim_normal_embed = 64,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        encoder_depth = 2,
        decoder_depth = 2,
        dim_codebook = 192,
        num_quantizers = 2,           # or 'D' in the paper
        codebook_size = 16384,        # they use 16k, shared codebook between layers
        use_residual_lfq = True,      # whether to use the latest lookup-free quantization
        rq_kwargs: dict = dict(),
        rvq_stochastic_sample_codes = True,
        sageconv_kwargs: dict = dict(
            normalize = True,
            project = True
        ),
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions
        linear_attention = False,
        linear_attn_kwargs: dict = dict(
            dim_head = 32,
            heads = 4
        ),
        pad_id = -1,
        flash_attn = True
    ):
        super().__init__()

        # autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._config = pickle.dumps(_locals)

        # main face coordinate embedding

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # initial dimension

        init_dim = dim_coor_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed

        # project into model dimension

        self.project_in = nn.Linear(init_dim, dim)

        self.encoders = ModuleList([])

        for _ in range(encoder_depth):
            attn = LinearAttention(dim, flash = flash_attn, **linear_attn_kwargs) if linear_attention else None
            sage_conv = SAGEConv(dim, dim, **sageconv_kwargs)

            self.encoders.append(ModuleList([
                attn,
                sage_conv
            ]))

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.project_dim_codebook = nn.Linear(dim, dim_codebook * 3)

        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                commitment_loss_weight = 1.,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                shared_codebook = True,
                commitment_weight = 1.,
                stochastic_sample_codes = rvq_stochastic_sample_codes,
                **rq_kwargs
            )

        self.pad_id = pad_id # for variable lengthed faces, padding quantized ids will be set to this value

        self.project_codebook_out = nn.Linear(dim_codebook * 3, dim)

        self.decoders = ModuleList([])

        for _ in range(decoder_depth):
            attn = LinearAttention(dim, flash = flash_attn, **linear_attn_kwargs) if linear_attention else None
            resnet_block = ResnetBlock(dim)

            self.decoders.append(ModuleList([
                attn,
                resnet_block
            ]))

        self.to_coor_logits = nn.Sequential(
            nn.Linear(dim, num_discrete_coors * 9),
            Rearrange('... (v c) -> ... v c', v = 9)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved mesh autoencoder at version {pkg["version"]}, but current package version is {__version__}')

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)

        tokenizer.load_state_dict(pkg['model'], strict = strict)
        return tokenizer

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.state_dict(),
            config = self._config,
            version = __version__,
        )

        torch.save(pkg, str(path))

    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 3, int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask:        TensorType['b', 'nf', bool],
        face_edges_mask:  TensorType['b', 'e', bool],
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

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)

        # continuous face coords

        face_coords = vertices.gather(-2, faces_vertices)

        # compute derived features and embed

        derived_features = get_derived_face_features(face_coords)

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        # discretize vertices for face coordinate embedding

        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 coordinates per face

        face_coor_embed = self.coor_embed(discrete_face_coords)
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        # combine all features and project into model dimension

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        face_embed = self.project_in(face_embed)

        # handle variable lengths by using masked_select and masked_scatter

        # first handle edges
        # needs to be offset by number of faces for each batch

        face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
        face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')

        face_edges = face_edges + face_index_offsets
        face_edges = face_edges[face_edges_mask]
        face_edges = rearrange(face_edges, 'be ij -> ij be')

        # next prepare the face_mask for using masked_select and masked_scatter

        dtype = face_embed.dtype
        orig_face_embed_shape = face_embed.shape

        def to_orig_face_embed_shape(flattened_face_embed):
            zeros = torch.zeros(orig_face_embed_shape, device = device, dtype = dtype)
            face_mask_with_append_dim = rearrange(face_mask, '... -> ... 1')
            return zeros.masked_scatter(face_mask_with_append_dim, flattened_face_embed)

        face_embed = face_embed[face_mask]

        for maybe_attn, conv in self.encoders:

            if exists(maybe_attn):
                face_embed = to_orig_face_embed_shape(face_embed)
                face_embed = maybe_attn(face_embed, mask = face_mask, channel_first = False) + face_embed
                face_embed = face_embed[face_mask]

            face_embed = conv(face_embed, face_edges)

        face_embed = to_orig_face_embed_shape(face_embed)

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @beartype
    def quantize(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, num_faces, device = *faces.shape[:2], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nv d) -> b nf nv d', nv = 3)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = F.pad(vertices, (0, 0, 0, 1), value = 0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nv -> b (nf nv) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict()

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # residual VQ

        quantized, codes, commit_loss = self.quantizer(averaged_vertices, mask = mask, **quantize_kwargs)

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = quantized.gather(-2, faces_with_dim)
        face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        face_embed_output = self.project_codebook_out(face_embed_output)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, faces_with_quantized_dim)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nv) 1', nv = 3)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized
        x = rearrange(x, 'b n d -> b d n')

        for maybe_attn, resnet_block in self.decoders:
            if exists(maybe_attn):
                x = maybe_attn(x, mask = face_mask, channel_first = True) + x

            x = resnet_block(x, mask = conv_face_mask)

        return rearrange(x, 'b d n -> b n d')

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: Optional[TensorType['b', 'n', bool]] = None,
        return_discrete_codes = False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nv q) -> b nf', 'all', nv = 3, q = self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        quantized = quantized.masked_fill(~face_mask[..., None], 0.)
        face_embed_output = self.project_codebook_out(quantized)

        decoded = self.decode(
            face_embed_output,
            face_mask = face_mask
        )

        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = 3)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        if not return_discrete_codes:
            return continuous_coors

        return continuous_coors, pred_face_coords

    @torch.no_grad()
    def tokenize(self, *args, **kwargs):
        assert 'return_codes' not in kwargs
        self.eval()
        return self.forward(*args, return_codes = True, **kwargs)

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
        rvq_sample_codebook_temp = 1.
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )

        quantized, codes, commit_loss = self.quantize(
            face_embed = encoded,
            faces = faces,
            face_mask = face_mask,
            rvq_sample_codebook_temp = rvq_sample_codebook_temp
        )

        if return_codes:
            assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'
            return codes

        decode = self.decode(
            quantized,
            face_mask = face_mask
        )

        pred_face_coords = self.to_coor_logits(decode)

        # compute reconstructed faces if needed

        if return_recon_faces or only_return_recon_faces:

            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))

        if only_return_recon_faces:
            return recon_faces

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

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = 9)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss

        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # calculate loss breakdown if needed

        loss_breakdown = (recon_loss, commit_loss)

        # some return logic

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown

class MeshTransformer(Module):
    @beartype
    def __init__(
        self,
        autoencoder: MeshAutoencoder,
        *,
        dim = 512,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            num_mem_kv = 4
        ),
        fine_pre_gateloop_depth = 2,
        fine_attn_depth = 2,
        fine_attn_dim_head = 32,
        fine_attn_heads = 8,
        pad_id = -1,
        condition_on_text = False,
        text_condition_model_types = ('t5',),
        text_condition_cond_drop_prob = 0.25
    ):
        super().__init__()

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token_id = self.codebook_size

        # they use axial positional embeddings

        assert divisible_by(max_seq_len, 3 * self.num_quantizers) # 3 vertices per face, with D codes per vertex

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(3, dim))

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        # text condition

        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                cond_drop_prob = text_condition_cond_drop_prob
            )
            cross_attn_dim_context = self.conditioner.dim_latent

        # for summarizing the vertices of each face

        self.to_face_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * 3 * dim, dim),
            nn.LayerNorm(dim)
        )

        # main autoregressive attention network
        # attending to a face token

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            flash_attn = flash_attn,
            cross_attend = condition_on_text,
            cross_attn_dim_context = cross_attn_dim_context,
            **attn_kwargs
        )

        # address a weakness in attention

        self.gateloop_layers = ModuleList([])

        for _ in range(fine_pre_gateloop_depth):
            self.gateloop_layers.append(SimpleGateLoopLayer(dim))

        # decoding the vertices, 2-stage hierarchy

        self.fine_decoder = Decoder(
            dim = dim,
            depth = fine_attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            flash_attn = flash_attn,
            **attn_kwargs
        )

        # to logits

        self.to_logits = nn.Linear(dim, self.codebook_size + 1)

        # padding id
        # force the autoencoder to use the same pad_id given in transformer

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_scale = 1.,
        cache_kv = False
    ):
        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'
            if exists(texts):
                text_embeds = self.embed_texts(texts)

            batch_size = default(batch_size, text_embeds.shape[0])

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        # for now, kv cache disabled when conditioning on text

        assert not cache_kv, 'caching not available yet'
        can_cache = cache_kv and (not self.condition_on_text or cond_scale == 1.)

        cache = None

        for i in tqdm(range(curr_length, self.max_seq_len)):
            # [sos] v1([q1] [q2] [q1] [q2] [q1] [q2]) v2([q1] [q2] [q1] [q2] [q1] [q2]) -> 0 1 2 3 4 5 6 7 8 9 10 11 12 -> F v1(F F F F F T) v2(F F F F F T)

            can_eos = i != 0 and divisible_by(i, self.num_quantizers * 3)  # only allow for eos to be decoded at the end of each face, defined as 3 vertices with D residual VQ codes

            logits, new_cache = self.forward_on_codes(
                codes,
                cache = cache,
                text_embeds = text_embeds,
                return_loss = False,
                return_cache = True,
                append_eos = False,
                cond_scale = cond_scale
            )

            if can_cache:
                cache = new_cache

            logits = logits[:, -1]

            if not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                sample = filtered_logits.argmax(dim = -1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate

            is_eos_codes = (codes == self.eos_token_id)

            if not is_eos_codes.any(dim = -1).all():
                continue

            shifted_is_eos_tokens = F.pad(is_eos_codes, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            codes = codes.masked_fill(mask, self.pad_id)
            break

        if return_codes:
            codes = codes[:, 1:] # remove sos
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        self.autoencoder.eval()
        return self.autoencoder.decode_from_codes_to_faces(codes)

    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, int],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        cache:          Optional[LayerIntermediates] = None,
        **kwargs
    ):
        codes = self.autoencoder.tokenize(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges
        )

        return self.forward_on_codes(codes, cache = cache, **kwargs)

    @classifier_free_guidance
    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = 0.
    ):
        # handle text conditions

        attn_context_kwargs = dict()

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'

            if exists(texts):
                text_embeds = self.conditioner.embed_texts(texts)

            if exists(codes):
                assert text_embeds.shape[0] == codes.shape[0], 'batch size of texts or text embeddings is not equal to the batch size of the mesh codes'

            _, maybe_dropped_text_embeds = self.conditioner(
                text_embeds = text_embeds,
                cond_drop_prob = cond_drop_prob
            )

            attn_context_kwargs = dict(
                context = maybe_dropped_text_embeds.embed,
                context_mask = maybe_dropped_text_embeds.mask
            )

        # take care of codes that may be flattened

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # get some variable

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # auto append eos token

        if append_eos:
            assert exists(codes)

            code_lens = ((codes != self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)

            codes = F.pad(codes, (0, 1), value = 0)

            batch_arange = torch.arange(batch, device = device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')

            codes[batch_arange, code_lens] = self.eos_token_id

        # if returning loss, save the labels for cross entropy

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed (each residual VQ id)

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions

        seq_arange = torch.arange(codes.shape[-2], device = device)

        codes = codes + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        code_len = codes.shape[1]

        level_embed = repeat(self.quantize_level_embed, 'q d -> (r q) d', r = ceil(code_len / self.num_quantizers))
        codes = codes + level_embed[:code_len]

        # embedding for each vertex

        vertex_embed = repeat(self.vertex_embed, 'nv d -> (r nv q) d', r = ceil(code_len / (3 * self.num_quantizers)), q = self.num_quantizers)
        codes = codes + vertex_embed[:code_len]

        # create a token per face, by summarizing the 3 vertices
        # this is similar in design to the RQ transformer from Lee et al. https://arxiv.org/abs/2203.01941

        num_tokens_per_face = self.num_quantizers * 3
        code_len_is_multiple_of_face = divisible_by(code_len, num_tokens_per_face)
        next_multiple_code_len = ceil(code_len / num_tokens_per_face) * num_tokens_per_face

        codes = pad_to_length(codes, next_multiple_code_len, dim = -2)

        # grouped codes will be used for the second stage

        grouped_codes = rearrange(codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # create the coarse tokens for the first attention network

        face_codes = grouped_codes if code_len_is_multiple_of_face else grouped_codes[:, :-1]
        face_codes = rearrange(face_codes, 'b nf n d -> b nf (n d)')
        face_codes = self.to_face_tokens(face_codes)

        # caches

        coarse_cache, fine_cache = cache if exists(cache) else (None, None)

        # attention on face codes (coarse)

        attended_face_codes, coarse_cache = self.decoder(
            face_codes,
            cache = coarse_cache,
            return_hiddens = True,
            **attn_context_kwargs
        )

        # auto prepend sos token

        sos = repeat(self.sos_token, 'd -> b d', b = batch)

        attended_face_codes, _ = pack([sos, attended_face_codes], 'b * d')

        grouped_codes = pad_to_length(grouped_codes, attended_face_codes.shape[-2], dim = 1)
        fine_vertex_codes, _ = pack([attended_face_codes, grouped_codes], 'b n * d')

        fine_vertex_codes = fine_vertex_codes[..., :-1, :]

        # gateloop layers

        if not is_empty(self.gateloop_layers):
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> b (nf n) d')

            for gateloop in self.gateloop_layers:
                fine_vertex_codes = gateloop(fine_vertex_codes) + fine_vertex_codes

            fine_vertex_codes = rearrange(fine_vertex_codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # fine attention - 2nd stage

        fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> (b nf) n d')

        attended_vertex_codes, fine_cache = self.fine_decoder(
            fine_vertex_codes,
            cache = fine_cache,
            return_hiddens = True
        )

        # reconstitute original sequence

        attended_vertex_codes = rearrange(attended_vertex_codes, '(b nf) n d -> b (nf n) d', b = batch)
        attended_vertex_codes = attended_vertex_codes[:, :(code_len + 1)]

        # logits

        logits = self.to_logits(attended_vertex_codes)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, (coarse_cache, fine_cache)

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return ce_loss
