"""Plotting utilities for signals, training curves, and model comparison."""

from pathlib import Path

from numpy.typing import NDArray


def plot_signals(
    time: NDArray,
    signals: dict[str, NDArray],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot one or more signals on a shared time axis.

    Args:
        time: 1-D time array.
        signals: Mapping label -> 1-D signal array.
        title: Plot title.
        save_path: If provided, save figure to this path instead of showing.
    """
    raise NotImplementedError("Phase 8: plots.plot_signals")


def plot_loss_curves(
    history: dict[str, list[float]],
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dict with keys 'train' and 'val'.
        model_name: Used in the plot title.
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 8: plots.plot_loss_curves")


def plot_prediction_vs_target(
    target: NDArray,
    prediction: NDArray,
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Overlay a model's prediction against the ground-truth target window.

    Args:
        target: Ground-truth signal window.
        prediction: Model prediction.
        model_name: Used in the plot title.
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 8: plots.plot_prediction_vs_target")


def plot_mse_vs_noise(
    noise_levels: list[float],
    mse_by_model: dict[str, list[float]],
    save_path: Path | None = None,
) -> None:
    """Plot MSE as a function of noise level for each model.

    Args:
        noise_levels: List of noise standard deviations.
        mse_by_model: Mapping model_name -> list of MSE values per noise level.
        save_path: If provided, save figure to this path.
    """
    raise NotImplementedError("Phase 8: plots.plot_mse_vs_noise")
