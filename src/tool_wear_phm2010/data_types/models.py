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


@dataclass
class Metrics:
    """Per-cut wear error, in mm."""

    rmse: float
    mae: float
    r2: float


@dataclass
class EvalResult:
    """Cut-level predictions and the metrics derived from them."""

    predictions: np.ndarray  # (n_cuts,) predicted wear, mm
    targets: np.ndarray  # (n_cuts,) measured worst-flute wear, mm
    metrics: Metrics


@dataclass
class RunResult:
    """One leave-one-cutter-out fold."""

    test_cutter: str
    eval_result: EvalResult
    train_result: TrainResult
