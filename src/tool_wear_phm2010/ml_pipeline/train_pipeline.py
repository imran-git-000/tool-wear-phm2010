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
    """Train a model to predict tool wear from windows."""

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.config = train_config
        # None when nobody is watching; the loop is otherwise unchanged
        self.writer = writer
        self.train_result: TrainResult = TrainResult(
            train_losses=[],
            val_losses=[],
        )
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

    def train_step(self, dataloader: TorchDataLoader) -> float:
        """One epoch of training. Returns the mean loss."""
        self.model.train()
        # the running sum stays on the device: reading it every batch would
        # sync the host and stall the next batch's loading
        total = torch.zeros((), device=self.device)
        n_windows = 0
        for windows, wear, _ in dataloader:
            inputs, targets = self.to_device(windows, wear)
            loss: Tensor = self.loss_fn(self.model(inputs), targets)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # weight by batch size: a partial last batch must count for less
            total += loss.detach() * windows.shape[0]
            n_windows += windows.shape[0]
        return float(total) / n_windows

    @torch.inference_mode()
    def val_step(self, dataloader: TorchDataLoader) -> float:
        """One pass over held-out windows. Returns the mean loss."""
        self.model.eval()
        total = torch.zeros((), device=self.device)
        n_windows = 0
        for windows, wear, _ in dataloader:
            inputs, targets = self.to_device(windows, wear)
            loss: Tensor = self.loss_fn(self.model(inputs), targets)
            # weight by batch size: eval loaders keep their partial batch
            total += loss * windows.shape[0]
            n_windows += windows.shape[0]
        return float(total) / n_windows

    def train(
        self,
        train_dataloader: TorchDataLoader,
        val_dataloader: TorchDataLoader,
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
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss = self.val_step(dataloader=val_dataloader)
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

    def to_device(
        self, windows: Tensor, wear: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Move a batch to the device as (windows, target wear)."""
        return (
            windows.to(self.device, non_blocking=True),
            self.tool_wear(wear.to(self.device, non_blocking=True)),
        )

    @staticmethod
    def tool_wear(wear: Tensor) -> Tensor:
        """Worst flute, in mm, shaped (batch, 1).

        mm rather than 10^-3 mm keeps the target near the scale of a freshly
        initialised output layer. Unlabelled (NaN) windows must not reach
        the trainer: NaN makes the loss non-finite.
        """
        return wear.amax(dim=1, keepdim=True) / 1000
