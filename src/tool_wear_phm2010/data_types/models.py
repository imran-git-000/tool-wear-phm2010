from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Cut:
    cutter: str
    index: int
    signals: np.ndarray  # (n_samples, len(Channel)), float32
    wear: np.ndarray | None  # (3,) per-flute wear in 10^-3 mm


@dataclass
class TrainResult:
    """Training and validation loss curves."""

    train_losses: list[float]
    val_losses: list[float]
