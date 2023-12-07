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

def cycle(dl):
    while True:
        for data in dl:
            yield data

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

        if self.is_main:
            self.ema_model = EMA(model, **ema_kwargs)

        self.optimizer = get_optimizer(model.parameters(), lr = learning_rate, wd = weight_decay, **optimizer_kwargs)

        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        (
            self.model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.dataloader
        )

        self.max_grad_norm = max_grad_norm
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        for _ in range(self.num_train_steps):

            with self.accelerator.autocast():
                vertices, faces = next(dl_iter)

                loss = self.model(
                    vertices = vertices,
                    faces = faces
                )

                self.accelerator.backward(loss)

            self.print(f'loss: {loss.item():.3f}')

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.wait()

            if self.is_main:
                self.ema_model.update()

            self.wait()

            step += 1
            self.step.add_(1)

        self.print('training complete')

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

        self.optimizer = get_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            filter_by_requires_grad = True,
            **optimizer_kwargs
        )

        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        (
            self.model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.dataloader
        )

        self.max_grad_norm = max_grad_norm
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        for _ in range(self.num_train_steps):

            with self.accelerator.autocast():
                vertices, faces = next(dl_iter)

                loss = self.model(
                    vertices = vertices,
                    faces = faces
                )

                self.accelerator.backward(loss)

            self.print(f'loss: {loss.item():.3f}')

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

        self.print('training complete')