"""Loss function wrappers for training."""

import torch.nn as nn


def mse_loss() -> nn.MSELoss:
    """Return the standard mean-squared-error loss used across all models."""
    return nn.MSELoss()
