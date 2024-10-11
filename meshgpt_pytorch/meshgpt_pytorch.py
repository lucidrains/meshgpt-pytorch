from __future__ import annotations

from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

from pytorch_custom_utils import save_load

from beartype.typing import Tuple, Callable, List, Dict, Any
from meshgpt_pytorch.typing import Float, Int, Bool, typecheck

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from einx import get_at

from x_transformers import Decoder
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

from local_attention import LocalMHA

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.version import __version__

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from torch_geometric.nn.conv import SAGEConv

from gateloop_transformer import SimpleGateLoopLayer

from tqdm import tqdm

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(x):
    return len(x) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

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

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

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

def masked_mean(tensor, mask, dim = -1, eps = 1e-5):
    if not exists(mask):
        return tensor.mean(dim = dim)

    mask = rearrange(mask, '... -> ... 1')
    tensor = tensor.masked_fill(~mask, 0.)

    total_el = mask.sum(dim = dim)
    num = tensor.sum(dim = dim)
    den = total_el.float().clamp(min = eps)
    mean = num / den
    mean = mean.masked_fill(total_el == 0, 0.)
    return mean

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
@typecheck
def get_derived_face_features(
    face_coords: Float['b nf nvf 3']  # 3 or 4 vertices with 3 coordinates
):
    is_quad = face_coords.shape[-2] == 4

    # shift face coordinates depending on triangles or quads

    shifted_face_coords = torch.roll(face_coords, 1, dims = (2,))

    angles  = derive_angle(face_coords, shifted_face_coords)

    if is_quad:
        # @sbriseid says quads need to be shifted by 2
        shifted_face_coords = torch.roll(shifted_face_coords, 1, dims = (2,))

    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2)

    cross_product = torch.cross(edge1, edge2, dim = -1)

    normals = l2norm(cross_product)
    area = cross_product.norm(dim = -1, keepdim = True) * 0.5

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )   

# tensor helper functions

@typecheck
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

@typecheck
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

@typecheck
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)
    return rearrange(out, 'b c n -> b n c')

@typecheck
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

class FiLM(Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_gamma = nn.Linear(dim, dim_out, bias = False)
        self.to_beta = nn.Linear(dim, dim_out)

        self.gamma_mult = nn.Parameter(torch.zeros(1,))
        self.beta_mult = nn.Parameter(torch.zeros(1,))

    def forward(self, x, cond):
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = tuple(rearrange(t, 'b d -> b 1 d') for t in (gamma, beta))

        # for initializing to identity

        gamma = (1 + self.gamma_mult * gamma.tanh())
        beta = beta.tanh() * self.beta_mult

        # classic film

        return x * gamma + beta

class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        reduction_factor = 4,
        min_dim = 16
    ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min = 1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = PixelNorm(dim = 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out, dropout = dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        res = self.residual_conv(x)
        h = self.block1(x, mask = mask)
        h = self.block2(h, mask = mask)
        h = self.excite(h, mask = mask)
        return h + res

# gateloop layers

class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    ):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim, use_heinsen = use_heinsen)
            self.gateloops.append(gateloop)

    def forward(
        self,
        x,
        cache = None
    ):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        return x, new_caches

# main classes

@save_load(version = __version__)
class MeshAutoencoder(Module):
    @typecheck
    def __init__(
        self,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_area = 128,
        dim_area_embed = 16,
        num_discrete_normals = 128,
        dim_normal_embed = 64,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        encoder_dims_through_depth: Tuple[int, ...] = (
            64, 128, 256, 256, 576
        ),
        init_decoder_conv_kernel = 7,
        decoder_dims_through_depth: Tuple[int, ...] = (
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256, 256, 256,
            384, 384, 384
        ),
        dim_codebook = 192,
        num_quantizers = 2,           # or 'D' in the paper
        codebook_size = 16384,        # they use 16k, shared codebook between layers
        use_residual_lfq = True,      # whether to use the latest lookup-free quantization
        rq_kwargs: dict = dict(
            quantize_dropout = True,
            quantize_dropout_cutoff_index = 1,
            quantize_dropout_multiple_of = 1,
        ),
        rvq_kwargs: dict = dict(
            kmeans_init = True,
            threshold_ema_dead_code = 2,
        ),
        rlfq_kwargs: dict = dict(
            frac_per_sample_entropy = 1.,
            soft_clamp_input_value = 10.
        ),
        rvq_stochastic_sample_codes = True,
        sageconv_kwargs: dict = dict(
            normalize = True,
            project = True
        ),
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions
        attn_encoder_depth = 0,
        attn_decoder_depth = 0,
        local_attn_kwargs: dict = dict(
            dim_head = 32,
            heads = 8
        ),
        local_attn_window_size = 64,
        linear_attn_kwargs: dict = dict(
            dim_head = 8,
            heads = 16
        ),
        use_linear_attn = True,
        pad_id = -1,
        flash_attn = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        resnet_dropout = 0,
        checkpoint_quantizer = False,
        quads = False
    ):
        super().__init__()

        self.num_vertices_per_face = 3 if not quads else 4
        total_coordinates_per_face = self.num_vertices_per_face * 3

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

        # attention related

        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        # initial dimension

        init_dim = dim_coor_embed * (3 * self.num_vertices_per_face) + dim_angle_embed * self.num_vertices_per_face + dim_normal_embed * 3 + dim_area_embed

        # project into model dimension

        self.project_in = nn.Linear(init_dim, dim_codebook)

        # initial sage conv 
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook, init_encoder_dim, **sageconv_kwargs)

        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        )

        self.encoders = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )

            self.encoders.append(sage_conv)
            curr_dim = dim_layer

        self.encoder_attn_blocks = ModuleList([])

        for _ in range(attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(curr_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = ff_dropout))
            ]))

        # residual quantization

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * self.num_vertices_per_face)

        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                commitment_loss_weight = 1.,
                **rlfq_kwargs,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                shared_codebook = True,
                commitment_weight = 1.,
                rotation_trick = True,
                stochastic_sample_codes = rvq_stochastic_sample_codes,
                **rvq_kwargs,
                **rq_kwargs
            )

        self.checkpoint_quantizer = checkpoint_quantizer # whether to memory checkpoint the quantizer

        self.pad_id = pad_id # for variable lengthed faces, padding quantized ids will be set to this value

        # decoder

        decoder_input_dim = dim_codebook * 3

        self.decoder_attn_blocks = ModuleList([])

        for _ in range(attn_decoder_depth):
            self.decoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(decoder_input_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = decoder_input_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(decoder_input_dim), FeedForward(decoder_input_dim, glu = True, dropout = ff_dropout))
            ]))

        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * self.num_vertices_per_face, init_decoder_dim, kernel_size = init_decoder_conv_kernel, padding = init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.decoders = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout = resnet_dropout)

            self.decoders.append(resnet_block)
            curr_dim = dim_layer

        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * total_coordinates_per_face),
            Rearrange('... (v c) -> ... v c', v = total_coordinates_per_face)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @property
    def device(self):
        return next(self.parameters()).device
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: Dict | None,
        resume_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ): 
        model_filename = "mesh-autoencoder.bin" 
        model_file = Path(model_id) / model_filename 
        if not model_file.exists(): 
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )    
        model = cls.init_and_load(model_file,strict=strict) 
        model.to(map_location)
        return model
    
    @typecheck
    def encode(
        self,
        *,
        vertices:         Float['b nv 3'],
        faces:            Int['b nf nvf'],
        face_edges:       Int['b e 2'],
        face_mask:        Bool['b nf'],
        face_edges_mask:  Bool['b e'],
        return_face_coordinates = False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        nvf - number of vertices per face (3 or 4) - triangles vs quads
        c - coordinates (3)
        d - embed dim
        """

        _, num_faces, num_vertices_per_face = faces.shape

        assert self.num_vertices_per_face == num_vertices_per_face

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        # continuous face coords

        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad)

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
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face

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

        orig_face_embed_shape = face_embed.shape[:2]

        face_embed = face_embed[face_mask]

        # initial sage conv followed by activation and norm

        face_embed = self.init_sage_conv(face_embed, face_edges)

        face_embed = self.init_encoder_act_and_norm(face_embed)

        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges)

        shape = (*orig_face_embed_shape, face_embed.shape[-1])

        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)

        for linear_attn, attn, ff in self.encoder_attn_blocks:
            if exists(linear_attn):
                face_embed = linear_attn(face_embed, mask = face_mask) + face_embed

            face_embed = attn(face_embed, mask = face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @typecheck
    def quantize(
        self,
        *,
        faces: Int['b nf nvf'],
        face_mask: Bool['b n'],
        face_embed: Float['b nf d'],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, device = faces.shape[0], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nvf d) -> b nf nvf d', nvf = self.num_vertices_per_face)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nvf -> b (nf nvf) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict(mask = mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = get_at('b [n] d, b nf nvf -> b nf (nvf d)', quantized, faces)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        codes_output = get_at('b [n] q, b nf nvf -> b (nf nvf) q', codes, faces)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss

    @typecheck
    def decode(
        self,
        quantized: Float['b n d'],
        face_mask:  Bool['b n']
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized

        for linear_attn, attn, ff in self.decoder_attn_blocks:
            if exists(linear_attn):
                x = linear_attn(x, mask = face_mask) + x

            x = attn(x, mask = face_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~conv_face_mask, 0.)
        x = self.init_decoder_conv(x)

        for resnet_block in self.decoders:
            x = resnet_block(x, mask = conv_face_mask)

        return rearrange(x, 'b d n -> b n d')

    @typecheck
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: Bool['b n'] | None = None,
        return_discrete_codes = False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nvf q) -> b nf', 'all', nvf = self.num_vertices_per_face, q = self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nvf) d -> b nf (nvf d)', nvf = self.num_vertices_per_face)

        decoded = self.decode(
            quantized,
            face_mask = face_mask
        )

        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = self.num_vertices_per_face)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        # mask out with nan

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, face_mask

        return continuous_coors, pred_face_coords, face_mask

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges = None, **kwargs):
        assert 'return_codes' not in kwargs

        inputs = [vertices, faces, face_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'faces', 'face_edges'], inputs))

        self.eval()

        codes = self.forward(
            **input_kwargs,
            return_codes = True,
            **kwargs
        )

        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @typecheck
    def forward(
        self,
        *,
        vertices:       Float['b nv 3'],
        faces:          Int['b nf nvf'],
        face_edges:     Int['b e 2'] | None = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
        rvq_sample_codebook_temp = 1.
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        device = faces.device

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

            codes = codes.masked_fill(~repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face), self.pad_id)
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

            recon_faces = rearrange(recon_faces, 'b nf (nvf c) -> b nf nvf c', nvf = self.num_vertices_per_face)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')

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

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = self.num_vertices_per_face * 3)
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

@save_load(version = __version__)
class MeshTransformer(Module, PyTorchModelHubMixin):
    @typecheck
    def __init__(
        self,
        autoencoder: MeshAutoencoder,
        *,
        dim: int | Tuple[int, int] = 512,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            attn_num_mem_kv = 4
        ),
        cross_attn_num_mem_kv = 4, # needed for preventing nan when dropping out text condition
        dropout = 0.,
        coarse_pre_gateloop_depth = 2,
        coarse_post_gateloop_depth = 0,
        coarse_adaptive_rmsnorm = False,
        fine_pre_gateloop_depth = 2,
        gateloop_use_heinsen = False,
        fine_attn_depth = 2,
        fine_attn_dim_head = 32,
        fine_attn_heads = 8,
        fine_cross_attend_text = False, # additional conditioning - fine transformer cross attention to text tokens
        pad_id = -1,
        num_sos_tokens = None,
        condition_on_text = False,
        text_cond_with_film = False,
        text_condition_model_types = ('t5',),
        text_condition_model_kwargs = (dict(),),
        text_condition_cond_drop_prob = 0.25,
        quads = False,
    ):
        super().__init__()
        self.num_vertices_per_face = 3 if not quads else 4

        assert autoencoder.num_vertices_per_face == self.num_vertices_per_face, 'autoencoder and transformer must both support the same type of mesh (either all triangles, or all quads)'

        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.eos_token_id = self.codebook_size

        # the fine transformer sos token
        # as well as a projection of pooled text embeddings to condition it

        num_sos_tokens = default(num_sos_tokens, 1 if not condition_on_text else 4)
        assert num_sos_tokens > 0

        self.num_sos_tokens = num_sos_tokens
        self.sos_token = nn.Parameter(torch.randn(num_sos_tokens, dim))

        # they use axial positional embeddings

        assert divisible_by(max_seq_len, self.num_vertices_per_face * self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by (3 x {self.num_quantizers}) = {3 * self.num_quantizers}' # 3 or 4 vertices per face, with D codes per vertex

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(self.num_vertices_per_face, dim))

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        # text condition

        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None
        dim_text = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                model_kwargs = text_condition_model_kwargs,
                cond_drop_prob = text_condition_cond_drop_prob,
                text_embed_pad_value = -1.
            )

            dim_text = self.conditioner.dim_latent
            cross_attn_dim_context = dim_text

            self.text_coarse_film_cond = FiLM(dim_text, dim) if text_cond_with_film else identity
            self.text_fine_film_cond = FiLM(dim_text, dim_fine) if text_cond_with_film else identity

        # for summarizing the vertices of each face

        self.to_face_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * self.num_vertices_per_face * dim, dim),
            nn.LayerNorm(dim)
        )

        self.coarse_gateloop_block = GateLoopBlock(dim, depth = coarse_pre_gateloop_depth, use_heinsen = gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        self.coarse_post_gateloop_block = GateLoopBlock(dim, depth = coarse_post_gateloop_depth, use_heinsen = gateloop_use_heinsen) if coarse_post_gateloop_depth > 0 else None

        # main autoregressive attention network
        # attending to a face token

        self.coarse_adaptive_rmsnorm = coarse_adaptive_rmsnorm

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            heads = attn_heads,
            attn_dim_head = attn_dim_head,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            use_adaptive_rmsnorm = coarse_adaptive_rmsnorm,
            dim_condition = dim_text,
            cross_attend = condition_on_text,
            cross_attn_dim_context = cross_attn_dim_context,
            cross_attn_num_mem_kv = cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # projection from coarse to fine, if needed

        self.maybe_project_coarse_to_fine = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()

        # address a weakness in attention

        self.fine_gateloop_block = GateLoopBlock(dim, depth = fine_pre_gateloop_depth, use_heinsen = gateloop_use_heinsen) if fine_pre_gateloop_depth > 0 else None

        # decoding the vertices, 2-stage hierarchy

        self.fine_cross_attend_text = condition_on_text and fine_cross_attend_text

        self.fine_decoder = Decoder(
            dim = dim_fine,
            depth = fine_attn_depth,
            heads = attn_heads,
            attn_dim_head = attn_dim_head,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = self.fine_cross_attend_text,
            cross_attn_dim_context = cross_attn_dim_context,
            cross_attn_num_mem_kv = cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # to logits

        self.to_logits = nn.Linear(dim_fine, self.codebook_size + 1)

        # padding id
        # force the autoencoder to use the same pad_id given in transformer

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: Dict | None,
        resume_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ): 
        model_filename = "mesh-transformer.bin" 
        model_file = Path(model_id) / model_filename 

        if not model_file.exists(): 
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        model = cls.init_and_load(model_file,strict=strict) 
        model.to(map_location)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @typecheck
    @torch.no_grad()
    def embed_texts(self, texts: str | List[str]):
        single_text = not isinstance(texts, list)
        if single_text:
            texts = [texts]

        assert exists(self.conditioner)
        text_embeds = self.conditioner.embed_texts(texts).detach()

        if single_text:
            text_embeds = text_embeds[0]

        return text_embeds

    @eval_decorator
    @torch.no_grad()
    @typecheck
    def generate(
        self,
        prompt: Tensor | None = None,
        batch_size: int | None = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_scale = 1.,
        cache_kv = True,
        max_seq_len = None,
        face_coords_to_file: Callable[[Tensor], Any] | None = None
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

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

        cache = (None, None)

        for i in tqdm(range(curr_length, max_seq_len)):

            # example below for triangles, extrapolate for quads
            # v1([q1] [q2] [q1] [q2] [q1] [q2]) v2([eos| q1] [q2] [q1] [q2] [q1] [q2]) -> 0 1 2 3 4 5 6 7 8 9 10 11 12 -> v1(F F F F F F) v2(T F F F F F) v3(T F F F F F)

            can_eos = i != 0 and divisible_by(i, self.num_quantizers * self.num_vertices_per_face)  # only allow for eos to be decoded at the end of each face, defined as 3 or 4 vertices with D residual VQ codes

            output = self.forward_on_codes(
                codes,
                text_embeds = text_embeds,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
                cond_scale = cond_scale,
                cfg_routed_kwargs = dict(
                    cache = cache
                )
            )

            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

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

            if is_eos_codes.any(dim = -1).all():
                break

        # mask out to padding anything after the first eos

        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        # remove a potential extra token from eos, if breaked early

        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_vertices_per_face * self.num_vertices_per_face
        codes = codes[:, :round_down_code_len]

        # early return of raw residual quantizer codes

        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        self.autoencoder.eval()
        face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

        if not exists(face_coords_to_file):
            return face_coords, face_mask

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    def forward(
        self,
        *,
        vertices:       Int['b nv 3'],
        faces:          Int['b nf nvf'],
        face_edges:     Int['b e 2'] | None = None,
        codes:          Tensor | None = None,
        cache:          LayerIntermediates | None = None,
        **kwargs
    ):
        if not exists(codes):
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
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None
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

            text_embed, text_mask = maybe_dropped_text_embeds

            pooled_text_embed = masked_mean(text_embed, text_mask, dim = 1)

            attn_context_kwargs = dict(
                context = text_embed,
                context_mask = text_mask
            )

            if self.coarse_adaptive_rmsnorm:
                attn_context_kwargs.update(
                    condition = pooled_text_embed
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

            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)

            codes = F.pad(codes, (0, 1), value = 0)

            batch_arange = torch.arange(batch, device = device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')

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

        vertex_embed = repeat(self.vertex_embed, 'nv d -> (r nv q) d', r = ceil(code_len / (self.num_vertices_per_face * self.num_quantizers)), q = self.num_quantizers)
        codes = codes + vertex_embed[:code_len]

        # create a token per face, by summarizing the 3 or 4 vertices
        # this is similar in design to the RQ transformer from Lee et al. https://arxiv.org/abs/2203.01941

        num_tokens_per_face = self.num_quantizers * self.num_vertices_per_face

        curr_vertex_pos = code_len % num_tokens_per_face # the current intra-face vertex-code position id, needed for caching at the fine decoder stage

        code_len_is_multiple_of_face = divisible_by(code_len, num_tokens_per_face)

        next_multiple_code_len = ceil(code_len / num_tokens_per_face) * num_tokens_per_face

        codes = pad_to_length(codes, next_multiple_code_len, dim = -2)

        # grouped codes will be used for the second stage

        grouped_codes = rearrange(codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # create the coarse tokens for the first attention network

        face_codes = grouped_codes if code_len_is_multiple_of_face else grouped_codes[:, :-1]
        face_codes = rearrange(face_codes, 'b nf n d -> b nf (n d)')
        face_codes = self.to_face_tokens(face_codes)

        face_codes_len = face_codes.shape[-2]

        # cache logic

        (
            cached_attended_face_codes,
            coarse_cache,
            fine_cache,
            coarse_gateloop_cache,
            coarse_post_gateloop_cache,
            fine_gateloop_cache
        ) = cache if exists(cache) else ((None,) * 6)

        if exists(cache):
            cached_face_codes_len = cached_attended_face_codes.shape[-2]
            cached_face_codes_len_without_sos = cached_face_codes_len - 1

            need_call_first_transformer = face_codes_len > cached_face_codes_len_without_sos
        else:
            # auto prepend sos token

            sos = repeat(self.sos_token, 'n d -> b n d', b = batch)
            face_codes, packed_sos_shape = pack([sos, face_codes], 'b * d')

            # if no kv cache, always call first transformer

            need_call_first_transformer = True

        should_cache_fine = not divisible_by(curr_vertex_pos + 1, num_tokens_per_face)

        # condition face codes with text if needed

        if self.condition_on_text:
            face_codes = self.text_coarse_film_cond(face_codes, pooled_text_embed)

        # attention on face codes (coarse)

        if need_call_first_transformer:
            if exists(self.coarse_gateloop_block):
                face_codes, coarse_gateloop_cache = self.coarse_gateloop_block(face_codes, cache = coarse_gateloop_cache)

            attended_face_codes, coarse_cache = self.decoder(
                face_codes,
                cache = coarse_cache,
                return_hiddens = True,
                **attn_context_kwargs
            )

            if exists(self.coarse_post_gateloop_block):
                face_codes, coarse_post_gateloop_cache = self.coarse_post_gateloop_block(face_codes, cache = coarse_post_gateloop_cache)

        else:
            attended_face_codes = None

        attended_face_codes = safe_cat((cached_attended_face_codes, attended_face_codes), dim = -2)

        # if calling without kv cache, pool the sos tokens, if greater than 1 sos token

        if not exists(cache):
            sos_tokens, attended_face_codes = unpack(attended_face_codes, packed_sos_shape, 'b * d')
            last_sos_token = sos_tokens[:, -1:]
            attended_face_codes = torch.cat((last_sos_token, attended_face_codes), dim = 1)

        # maybe project from coarse to fine dimension for hierarchical transformers

        attended_face_codes = self.maybe_project_coarse_to_fine(attended_face_codes)

        grouped_codes = pad_to_length(grouped_codes, attended_face_codes.shape[-2], dim = 1)
        fine_vertex_codes, _ = pack([attended_face_codes, grouped_codes], 'b n * d')

        fine_vertex_codes = fine_vertex_codes[..., :-1, :]

        # gateloop layers

        if exists(self.fine_gateloop_block):
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> b (nf n) d')
            orig_length = fine_vertex_codes.shape[-2]
            fine_vertex_codes = fine_vertex_codes[:, :(code_len + 1)]

            fine_vertex_codes, fine_gateloop_cache = self.fine_gateloop_block(fine_vertex_codes, cache = fine_gateloop_cache)

            fine_vertex_codes = pad_to_length(fine_vertex_codes, orig_length, dim = -2)
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # fine attention - 2nd stage

        if exists(cache):
            fine_vertex_codes = fine_vertex_codes[:, -1:]

            if exists(fine_cache):
                for attn_intermediate in fine_cache.attn_intermediates:
                    ck, cv = attn_intermediate.cached_kv
                    ck, cv = [rearrange(t, '(b nf) ... -> b nf ...', b = batch) for t in (ck, cv)]

                    # when operating on the cached key / values, treat self attention and cross attention differently

                    layer_type = attn_intermediate.layer_type

                    if layer_type == 'a':
                        ck, cv = [t[:, -1, :, :curr_vertex_pos] for t in (ck, cv)]
                    elif layer_type == 'c':
                        ck, cv = [t[:, -1, ...] for t in (ck, cv)]

                    attn_intermediate.cached_kv = (ck, cv)

        num_faces = fine_vertex_codes.shape[1]
        one_face = num_faces == 1

        fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> (b nf) n d')

        if one_face:
            fine_vertex_codes = fine_vertex_codes[:, :(curr_vertex_pos + 1)]

        # handle maybe cross attention conditioning of fine transformer with text

        fine_attn_context_kwargs = dict()

        # optional text cross attention conditioning for fine transformer

        if self.fine_cross_attend_text:
            repeat_batch = fine_vertex_codes.shape[0] // text_embed.shape[0]

            text_embed = repeat(text_embed, 'b ... -> (b r) ...' , r = repeat_batch)
            text_mask = repeat(text_mask, 'b ... -> (b r) ...', r = repeat_batch)

            fine_attn_context_kwargs = dict(
                context = text_embed,
                context_mask = text_mask
            )

        # also film condition the fine vertex codes

        if self.condition_on_text:
            repeat_batch = fine_vertex_codes.shape[0] // pooled_text_embed.shape[0]

            pooled_text_embed = repeat(pooled_text_embed, 'b ... -> (b r) ...', r = repeat_batch)
            fine_vertex_codes = self.text_fine_film_cond(fine_vertex_codes, pooled_text_embed)

        # fine transformer

        attended_vertex_codes, fine_cache = self.fine_decoder(
            fine_vertex_codes,
            cache = fine_cache,
            **fine_attn_context_kwargs,
            return_hiddens = True
        )

        if not should_cache_fine:
            fine_cache = None

        if not one_face:
            # reconstitute original sequence

            embed = rearrange(attended_vertex_codes, '(b nf) n d -> b (nf n) d', b = batch)
            embed = embed[:, :(code_len + 1)]
        else:
            embed = attended_vertex_codes

        # logits

        logits = self.to_logits(embed)

        if not return_loss:
            if not return_cache:
                return logits

            next_cache = (
                attended_face_codes,
                coarse_cache,
                fine_cache,
                coarse_gateloop_cache,
                coarse_post_gateloop_cache,
                fine_gateloop_cache
            )

            return logits, next_cache

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return ce_loss
