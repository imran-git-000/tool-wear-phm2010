import copy
import math
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch import Tensor, nn
from tqdm.auto import tqdm

from tool_wear_phm2010.data_types.models import TrainResult
from tool_wear_phm2010.utils.utils import resolve_device, set_seed

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.tensorboard import SummaryWriter

    from tool_wear_phm2010.config.configs import TrainConfig


class Trainer:
    """Train a model on (n_channels, window_size) windows, self-supervised."""

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        horizon: int = 1,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.config = train_config
        # None when nobody is watching; the loop is otherwise unchanged
        self.writer = writer
        # only read when the caller trains a forecaster
        self.horizon = horizon
        self.train_result: TrainResult = TrainResult(
            train_losses=[],
            val_losses=[],
        )
        if self.config.seed is not None:
            # only covers training-time randomness such as dropout
            set_seed(self.config.seed)
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(
        self,
        dataloader: TorchDataLoader,
        is_forecasting: bool,
    ) -> float:
        """One epoch of training. Returns the mean loss."""
        self.model.train()
        # the running sum stays on the device: reading it every batch would
        # sync the host and stall the next batch's loading
        total = torch.zeros((), device=self.device)
        n_windows = 0
        for windows, _, _ in dataloader:
            batch: Tensor = windows.to(self.device, non_blocking=True)
            inputs, targets = (
                self.split(batch, self.horizon)
                if is_forecasting
                else self.reconstruct(batch)
            )
            loss: Tensor = self.loss_fn(self.model(inputs), targets)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # weight by batch size: a partial last batch must count for less
            total += loss.detach() * batch.shape[0]
            n_windows += batch.shape[0]
        return float(total) / n_windows

    @torch.inference_mode()
    def val_step(
        self,
        dataloader: TorchDataLoader,
        is_forecasting: bool,
    ) -> float:
        """One pass over held-out windows. Returns the mean loss."""
        self.model.eval()
        total = torch.zeros((), device=self.device)
        n_windows = 0
        for windows, _, _ in dataloader:
            batch: Tensor = windows.to(self.device, non_blocking=True)
            inputs, targets = (
                self.split(batch, self.horizon)
                if is_forecasting
                else self.reconstruct(batch)
            )
            loss: Tensor = self.loss_fn(self.model(inputs), targets)
            # weight by batch size: eval loaders keep their partial batch
            total += loss * batch.shape[0]
            n_windows += batch.shape[0]
        return float(total) / n_windows

    def train(
        self,
        train_dataloader: TorchDataLoader,
        val_dataloader: TorchDataLoader,
        is_forecasting: bool,
    ) -> TrainResult:
        """Fit the model, keeping the weights with the lowest val loss."""
        best_loss, best_state, waited = float('inf'), None, 0
        self.logger.info(
            'Start training for {epochs} epochs with {loss} and {optimizer}',
            epochs=self.config.epochs,
            loss=self.loss_fn.__class__.__name__,
            optimizer=self.optimizer.__class__.__name__,
        )
        for epoch in tqdm(range(1, self.config.epochs + 1), desc='training'):
            train_loss = self.train_step(
                dataloader=train_dataloader,
                is_forecasting=is_forecasting,
            )
            val_loss = self.val_step(
                dataloader=val_dataloader,
                is_forecasting=is_forecasting,
            )
            self.train_result.train_losses.append(train_loss)
            self.train_result.val_losses.append(val_loss)
            self.logger.info(
                'Epoch {epoch}: train_loss={train:.6f} val_loss={val:.6f}',
                epoch=epoch,
                train=train_loss,
                val=val_loss,
            )
            if self.writer is not None:
                self.writer.add_scalars(
                    'loss',
                    {'train': train_loss, 'val': val_loss},
                    epoch,
                )
                self.writer.flush()
            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                # a non-finite loss never beats best_loss
                raise RuntimeError(
                    f'Training diverged at epoch {epoch}: '
                    f'train_loss={train_loss}, val_loss={val_loss}'
                )

            if val_loss < best_loss:
                best_loss, waited = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                waited += 1
                if self.config.patience and waited >= self.config.patience:
                    self.logger.info(
                        'Early stopping at epoch {epoch}',
                        epoch=epoch,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.logger.info('Best validation loss: {loss:.6f}', loss=best_loss)
        return self.train_result

    @staticmethod
    def reconstruct(windows: Tensor) -> tuple[Tensor, Tensor]:
        """Pair a window with itself: the model rebuilds its own input."""
        return windows, windows

    @staticmethod
    def split(windows: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
        """Shift windows along time into (input, target)."""
        return windows[..., :-horizon], windows[..., horizon:]
