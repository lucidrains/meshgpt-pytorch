import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformer import (
    Decoder,
    AutoregressiveWrapper
)

from vector_quantize_pytorch import ResidualVQ
from torch_geometric.nn.conv import SAGEConv

# helper functions

def exists(v):
    return v is not None

# main classes

class MeshAutoencoder(Module):
    raise NotImplementedError

class MeshGPT(Module):
    raise NotImplementedError
