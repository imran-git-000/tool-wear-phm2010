from dataclasses import dataclass

from tool_wear_phm2010.data_types.enums import Aggregation


@dataclass
class ModelConfig:
    """Transformer encoder hyperparameters."""

    num_channels: int
    window_size: int
    patch_size: int  # samples per token; window_size must be a multiple
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    dropout: float


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
