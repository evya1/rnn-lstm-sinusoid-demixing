"""Plotting utilities for signals, training curves, and model comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_signals(
    time: NDArray,
    signals: dict[str, NDArray],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot one or more signals on a shared time axis.

    Args:
        time: 1-D time array (seconds).
        signals: Mapping label -> 1-D signal array.
        title: Plot title.
        save_path: If provided, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, signal in signals.items():
        ax.plot(time, signal, label=label, linewidth=0.9, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_loss_curves(
    history: dict[str, list[float]],
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dict with keys 'train' and 'val', each a list of per-epoch loss.
        model_name: Used in the plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="Train", linewidth=1.5)
    ax.plot(epochs, history["val"], label="Validation", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name.upper()} — Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_prediction_vs_target(
    target: NDArray,
    prediction: NDArray,
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Overlay a model's prediction against the ground-truth target window.

    Args:
        target: Ground-truth signal window (1-D).
        prediction: Model prediction (1-D, same length as target).
        model_name: Used in the plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(target, label="Target", linewidth=2.0, color="steelblue")
    ax.plot(prediction, label="Prediction", linestyle="--", linewidth=2.0, color="tomato")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{model_name.upper()} — Prediction vs Target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_mse_vs_noise(
    noise_levels: list[float],
    mse_by_model: dict[str, list[float]],
    save_path: Path | None = None,
) -> None:
    """Plot MSE as a function of noise level for each model.

    Args:
        noise_levels: List of noise standard deviations used in the sweep.
        mse_by_model: Mapping model_name -> list of MSE values (one per noise level).
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"fc": "o", "rnn": "s", "lstm": "^"}
    for model_name, mse_values in mse_by_model.items():
        ax.plot(
            noise_levels,
            mse_values,
            marker=markers.get(model_name, "o"),
            label=model_name.upper(),
            linewidth=1.5,
            markersize=7,
        )
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
