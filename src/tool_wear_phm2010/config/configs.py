from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from tool_wear_phm2010.data_types.enums import Aggregation

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DataConfig:
    """Which cutters to use and how to batch their windows."""

    cutters: list[str]  # labelled cutters; each one is the test set once
    val_fraction: float  # share of training cuts held out for validation
    hop: int
    batch_size: int
    num_workers: int


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


@dataclass
class Config:
    """The whole config file: one field per top-level YAML section."""

    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig
    eval_config: EvalConfig


class ConfigLoader:
    """Load a config file into the dataclasses."""

    @staticmethod
    def load_config(path: Path) -> Config:
        """Read the YAML file into the config dataclasses."""
        raw_dict = yaml.safe_load(path.read_text(encoding='utf-8'))
        return Config(
            data_config=DataConfig(**raw_dict['data']),
            model_config=ModelConfig(**raw_dict['model']),
            train_config=TrainConfig(**raw_dict['training']),
            eval_config=EvalConfig(**raw_dict['evaluation']),
        )
