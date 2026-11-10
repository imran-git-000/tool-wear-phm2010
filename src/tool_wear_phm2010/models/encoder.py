from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from tool_wear_phm2010.config.configs import ModelConfig


class BatchNorm(nn.Module):
    """BatchNorm over the hidden dimension of (batch, seq, hidden) tensors.

    The paper uses it instead of LayerNorm: it copes better with outlier
    values in time series.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Embeddings(nn.Module):
    """Project patches of the signal to tokens and add learnable positions.

    A strided Conv1d (eq. 2 of the paper) turns each patch of patch_size
    samples across all channels into one token, so attention runs over
    window_size / patch_size tokens instead of every 50 kHz sample.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.window_size % config.patch_size:
            raise ValueError(
                f'window_size {config.window_size} is not a multiple of '
                f'patch_size {config.patch_size}'
            )
        self.projection = nn.Conv1d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        num_tokens = config.window_size // config.patch_size
        self.position_embeddings = nn.Embedding(
            num_tokens,
            config.hidden_size,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, windows: Tensor) -> Tensor:
        # (batch, channels, window) -> (batch, tokens, hidden)
        tokens = self.projection(windows).transpose(1, 2)
        positions = torch.arange(tokens.shape[1], device=windows.device)
        return self.dropout(tokens + self.position_embeddings(positions))


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                f'hidden_size {config.hidden_size} is not a multiple of '
                f'num_attention_heads {config.num_attention_heads}'
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = (
            config.hidden_size // config.num_attention_heads
        )
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        # (batch, seq, hidden) -> (batch, heads, seq, head_size)
        batch, seq_len, _ = x.shape
        x = x.view(
            batch,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: Tensor) -> Tensor:
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        # softmax(QK^T / sqrt(d)) V, with dropout on the attention weights
        context = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return context.permute(0, 2, 1, 3).flatten(2)


class EncoderLayer(nn.Module):
    """Self-attention then feed-forward, each with residual and BatchNorm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = SelfAttention(config)
        self.attention_output = nn.Linear(
            config.hidden_size,
            config.hidden_size,
        )
        self.attention_norm = BatchNorm(config.hidden_size)
        self.intermediate = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
        )
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = BatchNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention = self.dropout(
            self.attention_output(self.attention(hidden_states)),
        )
        hidden_states = self.attention_norm(hidden_states + attention)
        feed_forward = self.dropout(
            self.output(nn.functional.gelu(self.intermediate(hidden_states))),
        )
        return self.output_norm(hidden_states + feed_forward)


class Encoder(nn.Module):
    """Windows (batch, channels, window) -> tokens (batch, tokens, hidden)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, windows: Tensor) -> Tensor:
        hidden_states = self.embeddings(windows)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncoderRegressor(nn.Module):
    """Encoder with a regression head: (batch, channels, window) -> (batch, 1).

    As in the paper (eq. 3), the output vectors of all tokens are
    concatenated and fed to one linear layer.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        num_tokens = config.window_size // config.patch_size
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(num_tokens * config.hidden_size, 1)

    def forward(self, windows: Tensor) -> Tensor:
        tokens = self.encoder(windows)
        return self.head(self.dropout(tokens.flatten(1)))
