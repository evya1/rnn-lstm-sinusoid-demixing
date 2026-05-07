"""Vanilla RNN model for sinusoid demixing."""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """Vanilla RNN model for sinusoid demixing.

    Input sequence shape: (batch, seq_len, features_per_timestep)
    where features_per_timestep = 1 (sigma_t) + num_components (one-hot selector).

    Output: predicted clean window of shape (batch, seq_len).

    Args:
        input_size: Features per timestep (1 + num_components).
        hidden_size: Number of RNN hidden units.
        num_layers: Number of stacked RNN layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def __repr__(self) -> str:
        return (
            f"RNNModel(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predicted clean window, shape (batch, seq_len).
        """
        out, _ = self.rnn(x)
        return self.output_layer(out).squeeze(-1)
