from typing import TYPE_CHECKING

import torch
from loguru import logger

from tool_wear_phm2010.ml_pipeline.train_pipeline import Trainer
from tool_wear_phm2010.utils.utils import resolve_device

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor, nn
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.tensorboard import SummaryWriter

    from tool_wear_phm2010.config.configs import TrainConfig
    from tool_wear_phm2010.data_types.models import TrainResult


class Regressor:
    """Predict tool wear (worst flute, mm) from a window.

    The model maps (batch, n_channels, window_size) to (batch, 1).
    """

    def __init__(self, model: nn.Module, train_config: TrainConfig) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.train_config = train_config
        self.device = resolve_device(train_config.device)
        self.model = model.to(self.device)

    def fit(
        self,
        train_loader: TorchDataLoader,
        val_loader: TorchDataLoader,
        writer: SummaryWriter | None = None,
    ) -> TrainResult:
        """Minimise wear MSE on labelled windows."""
        trainer = Trainer(
            model=self.model,
            train_config=self.train_config,
            writer=writer,
        )
        result = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
        # the trainer restored the weights of the best epoch
        self.model = trainer.model
        return result

    @torch.inference_mode()
    def predict_batch(self, windows: Tensor) -> np.ndarray:
        """Predicted wear in mm, shaped (batch,)."""
        self.model.eval()
        batch = windows.to(self.device, non_blocking=True)
        return self.model(batch).squeeze(1).cpu().numpy()

    def state(self) -> dict[str, Tensor]:
        return self.model.state_dict()

    def load_state(self, state: dict[str, Tensor]) -> None:
        self.model.load_state_dict(state)
