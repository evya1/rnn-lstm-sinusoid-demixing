"""Evaluation metrics for sinusoid demixing models."""

from numpy.typing import NDArray


def compute_mse(predictions: NDArray, targets: NDArray) -> float:
    """Compute mean squared error between predictions and targets.

    Args:
        predictions: Predicted values, any shape.
        targets: Ground-truth values, same shape as predictions.

    Returns:
        Scalar MSE value.
    """
    raise NotImplementedError("Phase 7: metrics.compute_mse")
