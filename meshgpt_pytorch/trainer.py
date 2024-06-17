from __future__ import annotations

from pathlib import Path
from functools import partial
from packaging import version
from contextlib import nullcontext

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule,
    add_wandb_tracker_contextmanager
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from beartype.typing import Tuple, Type, List
from meshgpt_pytorch.typing import typecheck, beartype_isinstance

from ema_pytorch import EMA

from meshgpt_pytorch.data import custom_collate

from meshgpt_pytorch.version import __version__

from meshgpt_pytorch.meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]

# autoencoder trainer

@add_wandb_tracker_contextmanager()
class MeshAutoencoderTrainer(Module):
    @typecheck
    def __init__(
        self,
        model: MeshAutoencoder,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        val_dataset: Dataset | None = None,
        val_every: int = 100,
        val_num_batches: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        max_grad_norm: float | None = None,
        ema_kwargs: dict = dict(
            use_foreach = True
        ),
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every = 1000,
        checkpoint_folder = './checkpoints',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges'],
        warmup_steps = 1000,
        use_wandb_tracking = False
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        # accelerator

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, **ema_kwargs)

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = get_adam_optimizer(model.parameters(), lr = learning_rate, wd = weight_decay, **optimizer_kwargs),
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id)
        )

        self.should_validate = exists(val_dataset)

        if self.should_validate:
            assert len(val_dataset) > 0, 'your validation dataset is empty'

            self.val_every = val_every
            self.val_num_batches = val_num_batches

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = True,
                drop_last = True,
                collate_fn = partial(custom_collate, pad_id = model.pad_id)
            )

        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert beartype_isinstance(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        (
            self.model,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader
        )

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

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

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            version = __version__,
            step = self.step.item(),
            config = self.unwrapped_model._config
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved mesh autoencoder at version {pkg["version"]}, but current package version is {__version__}')

        self.model.load_state_dict(pkg['model'])
        self.ema_model.load_state_dict(pkg['ema_model'])
        self.optimizer.load_state_dict(pkg['optimizer'])

        self.step.copy_(pkg['step'])

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        maybe_del(forward_kwargs, 'texts', 'text_embeds')
        return forward_kwargs

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.is_main and self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():

                    total_loss, (recon_loss, commit_loss) = self.model(
                        **forward_kwargs,
                        return_loss_breakdown = True
                    )

                    self.accelerator.backward(total_loss / self.grad_accum_every)

            self.print(f'recon loss: {recon_loss.item():.3f} | commit loss: {commit_loss.sum().item():.3f}')

            self.log(
                total_loss = total_loss.item(),
                commit_loss = commit_loss.sum().item(),
                recon_loss = recon_loss.item()
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.is_main:
                self.ema_model.update()

            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every):

                total_val_recon_loss = 0.
                self.ema_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)

                        val_loss, (val_recon_loss, val_commit_loss) = self.ema_model(
                            **forward_kwargs,
                            return_loss_breakdown = True
                        )

                        total_val_recon_loss += (val_recon_loss / num_val_batches)

                self.print(f'valid recon loss: {total_val_recon_loss:.3f}')

                self.log(val_loss = total_val_recon_loss)

            self.wait()

            if self.is_main and divisible_by(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'mesh-autoencoder.ckpt.{checkpoint_num}.pt')

            self.wait()

        self.print('training complete')

# mesh transformer trainer

@add_wandb_tracker_contextmanager()
class MeshTransformerTrainer(Module):
    @typecheck
    def __init__(
        self,
        model: MeshTransformer,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: float | None = 0.5,
        val_dataset: Dataset | None = None,
        val_every = 1,
        val_num_batches = 5,
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every = 1000,
        checkpoint_folder = './checkpoints',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges', 'texts'],
        warmup_steps = 1000,
        use_wandb_tracking = False
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            filter_by_requires_grad = True,
            **optimizer_kwargs
        )

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = optimizer,
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id)
        )

        self.should_validate = exists(val_dataset)

        if self.should_validate:
            assert len(val_dataset) > 0, 'your validation dataset is empty'

            self.val_every = val_every
            self.val_num_batches = val_num_batches

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = True,
                drop_last = True,
                collate_fn = partial(custom_collate, pad_id = model.pad_id)
            )

        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert beartype_isinstance(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        (
            self.model,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader
        )

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

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

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item(),
            version = __version__
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved mesh transformer at version {pkg["version"]}, but current package version is {__version__}')

        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    loss = self.model(**forward_kwargs)

                    self.accelerator.backward(loss / self.grad_accum_every)

            self.print(f'loss: {loss.item():.3f}')

            self.log(loss = loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every):

                total_val_loss = 0.
                self.unwrapped_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)

                        val_loss = self.unwrapped_model(**forward_kwargs)

                        total_val_loss += (val_loss / num_val_batches)

                self.print(f'valid recon loss: {total_val_loss:.3f}')

                self.log(val_loss = total_val_loss)

            self.wait()

            if self.is_main and divisible_by(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'mesh-transformer.ckpt.{checkpoint_num}.pt')

            self.wait()

        self.print('training complete')
