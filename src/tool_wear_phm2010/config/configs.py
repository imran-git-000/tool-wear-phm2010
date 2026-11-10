from dataclasses import dataclass

from tool_wear_phm2010.data_types.enums import Aggregation


@dataclass
class TrainConfig:
    """Training loop configuration."""

    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int  # early stopping; 0 disables it
    device: str
    seed: int
    checkpoint_name: str


@dataclass
class EvalConfig:
    """Every tunable knob of the evaluation lives here."""

    aggregation: Aggregation  # how window predictions pool into one per cut
    output_artifacts: str

    def __post_init__(self) -> None:
        self.aggregation = Aggregation(self.aggregation)
