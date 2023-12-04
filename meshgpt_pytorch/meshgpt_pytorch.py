import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformers import (
    TransformerWrapper,
    Decoder,
    AutoregressiveWrapper
)

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

        self.project_dim_codebook = nn.Linear(dim, dim_codebook * 9)

        self.quantizer = ResidualVQ(
            dim = dim_codebook,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            shared_codebook = True,
            **rq_kwargs
        )

        self.project_codebook_out = nn.Linear(dim_codebook * 9, dim)

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
        vertices: Tensor,
        faces: Tensor,
        face_edges: Tensor
    ):
        x = faces

        for conv in self.encoders:
            x = conv(x, face_edges)

        return x

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
            vertices = vertices,
            faces = faces,
            face_edges = face_edges
        )

        quantized, aux_loss = self.quantizer(encoded)

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
        attn_num_tokens = 128 ** 2,
        attn_depth = 6,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_kwargs: dict = dict(),
        ignore_index = -100
    ):
        super().__init__()

        self.decoder = TransformerWrapper(
            attn_num_tokens = num_tokens,
            attn_layers = Decoder(
                depth = attn_depth,
                dim_head = attn_dim_head,
                heads = attn_heads,
                **attn_kwargs
            )
        )

        self.ignore_index = ignore_index

        self.autoregressive_wrapper = AutoregressiveWrapper(
            self.decoder,
            ignore_index = ignore_index
        )

    def generate(self):
        return self

    def forward(
        self,
        x
    ):
        return x
