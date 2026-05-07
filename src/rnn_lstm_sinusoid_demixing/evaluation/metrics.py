"""Evaluation metrics for sinusoid demixing models."""

import numpy as np
from numpy.typing import NDArray


def compute_mse(predictions: NDArray, targets: NDArray) -> float:
    """Compute mean squared error between predictions and targets.

    Args:
        predictions: Predicted values, any shape.
        targets: Ground-truth values, same shape as predictions.

    Returns:
        Scalar MSE value.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    return float(np.mean((pred - tgt) ** 2))
