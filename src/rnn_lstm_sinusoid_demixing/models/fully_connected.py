"""Fully connected baseline model for sinusoid demixing."""

import torch
import torch.nn as nn


class FullyConnectedModel(nn.Module):
    """Feed-forward model for sinusoid demixing.

    Input: flattened [composite_window (window_size,), one_hot (num_components,)]
    Output: predicted clean window of shape (window_size,)

    Args:
        window_size: Number of samples in the context window.
        num_components: Number of sinusoidal components (one-hot length).
        hidden_size: Number of units in each hidden layer.
    """

    def __init__(
        self,
        window_size: int,
        num_components: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_components = num_components
        self.hidden_size = hidden_size
        input_size = window_size + num_components
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, window_size),
        )

    def __repr__(self) -> str:
        return (
            f"FullyConnectedModel(window_size={self.window_size}, "
            f"num_components={self.num_components}, "
            f"hidden_size={self.hidden_size})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size + num_components).

        Returns:
            Predicted clean window, shape (batch, window_size).
        """
        return self.net(x)
