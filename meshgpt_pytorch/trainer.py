from functools import partial
from contextlib import nullcontext

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from beartype import beartype
from beartype.typing import Optional

from ema_pytorch import EMA

from meshgpt_pytorch.meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# optimizer

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    opt_kwargs = dict(lr = lr, betas = betas, eps = eps)

    if wd == 0:
        return Adam(params, **opt_kwargs)

    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    if not group_wd_params:
        return AdamW(params, **opt_kwargs)

    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)

# custom collater

def custom_collate(data, pad_id = -1):
    raise NotImplementedError

# autoencoder trainer

class MeshAutoencoderTrainer(Module):
    @beartype
    def __init__(
        self,
        model: MeshAutoencoder,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        max_grad_norm: Optional[float] = None,
        ema_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model
        self.ema_model = EMA(model, **ema_kwargs)

        self.optimizer = get_optimizer(lr = learning_rate, wd = weight_decay, **optimizer_kwargs)

    def forward(self):
        raise NotImplementedError

# mesh transformer trainer

class MeshTransformerTrainer(Module):
    @beartype
    def __init__(
        self,
        model: MeshTransformer,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: Optional[float] = 0.5,
        ema_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model

        self.optimizer = get_optimizer(lr = learning_rate, wd = weight_decay, **optimizer_kwargs)

    def forward(self):
        raise NotImplementedError
