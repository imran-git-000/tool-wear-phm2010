from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from tool_wear_phm2010.data_types.enums import Channel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tool_wear_phm2010.data_types.models import Cut


class DataTransformer:
    """Z-score each channel with statistics from training cuts."""

    def __init__(self) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, cuts: Iterable[Cut]) -> None:
        """Compute per-channel mean and std, streaming over cuts."""
        n = 0
        total = np.zeros(len(Channel))
        total_sq = np.zeros(len(Channel))
        for cut in cuts:
            x = cut.signals.astype(np.float64)
            n += len(x)
            total += x.sum(axis=0)
            total_sq += (x**2).sum(axis=0)

        mean = total / n
        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(total_sq / n - mean**2).astype(np.float32)
        self.logger.info(
            'Fitted on {n} samples: mean={mean} std={std}',
            n=n,
            mean=self.mean.round(4),
            std=self.std.round(4),
        )

    def transform(self, cut: Cut) -> Cut:
        """Apply the fitted statistics."""
        mean, std = self._stats()
        return replace(cut, signals=(cut.signals - mean) / std)

    def inverse_transform(self, cut: Cut) -> Cut:
        """Undo the fitted statistics."""
        mean, std = self._stats()
        return replace(cut, signals=cut.signals * std + mean)

    def _stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.std is None:
            raise ValueError('call fit() first')
        return self.mean, self.std
