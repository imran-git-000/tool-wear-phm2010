from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from tool_wear_phm2010.data_types.models import EvalResult, Metrics
from tool_wear_phm2010.ml_pipeline.train_pipeline import Trainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader

    from tool_wear_phm2010.config.configs import EvalConfig
    from tool_wear_phm2010.regressor.regressor import Regressor


class Evaluator:
    """Pool a regressor's window predictions per cut and measure the error."""

    def __init__(
        self,
        eval_config: EvalConfig,
        regressor: Regressor,
        test_dataloader: TorchDataLoader,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.regressor = regressor
        self.test_dataloader = test_dataloader
        self.aggregation = eval_config.aggregation

    def evaluate(self) -> EvalResult:
        """Predict wear for every test cut and compare with the measurement."""
        predictions, targets = self.predict(
            regressor=self.regressor,
            dataloader=self.test_dataloader,
        )
        metrics = self.metrics(targets=targets, predictions=predictions)

        self.logger.info('Evaluation: {metrics}', metrics=metrics)
        return EvalResult(
            predictions=predictions,
            targets=targets,
            metrics=metrics,
        )

    def predict(
        self,
        regressor: Regressor,
        dataloader: TorchDataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-cut predicted and measured wear, ordered by cut position."""
        predictions, targets, cuts = [], [], []
        for windows, wear, batch_cuts in dataloader:
            predictions.append(regressor.predict_batch(windows=windows))
            targets.append(Trainer.tool_wear(wear).squeeze(1).numpy())
            cuts.append(batch_cuts.numpy())

        windows = pd.DataFrame(
            {
                'cut': np.concatenate(cuts),
                'prediction': np.concatenate(predictions),
                'target': np.concatenate(targets),
            },
        )
        # all windows of a cut share its target, so any of them will do
        per_cut = windows.groupby('cut').agg(
            prediction=('prediction', self.aggregation.value),
            target=('target', 'first'),
        )
        return per_cut['prediction'].to_numpy(), per_cut['target'].to_numpy()

    @staticmethod
    def metrics(targets: np.ndarray, predictions: np.ndarray) -> Metrics:
        """Root mean squared error, mean absolute error and R²."""
        errors = predictions - targets
        return Metrics(
            rmse=float(np.sqrt(np.mean(errors**2))),
            mae=float(np.mean(np.abs(errors))),
            r2=float(
                1 - np.sum(errors**2) / np.sum((targets - targets.mean()) ** 2)
            ),
        )
