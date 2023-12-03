import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype

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

# main classes

class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        dim,
        rq_kwargs: dict = dict()
    ):
        super().__init__()
        self.quantizer = self.ResidualVQ(**rq_kwargs)

    @beartype
    def encode(
        self,
        faces: Tensor,
        face_edges: Tensor
    ):
        return faces

    @beartype
    def decode(
        self,
        codes
    ):
        return faces

    @beartype
    def forward(
        self,
        faces: Tensor,
        face_edges: Tensor,
        return_quantized = False
    ):
        encoded = self.encode(faces, face_edges)
        quantized, aux_loss = self.quantizer(encoded)

        if return_quantized:
            return quantized

        decode = self.decode(quantized)

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
