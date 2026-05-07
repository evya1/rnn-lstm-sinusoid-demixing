"""Dataset construction from sinusoidal signals.

Builds (input_window, selector, target_window) triples for model training.

Input representation:
    FC:       flat vector [sigma_window (W,), one_hot (C,)] → shape (W+C,)
    RNN/LSTM: sequence tensor, shape (W, 1+C)
              where each timestep is [sigma_t, c_1, c_2, c_3, c_4]

This module returns raw (inputs, selectors, targets) arrays.
Model-specific concatenation / interleaving happens in the training layer.
"""

import numpy as np
from numpy.typing import NDArray


def make_one_hot(index: int, num_classes: int) -> NDArray:
    """Return a one-hot vector of length num_classes with index set to 1.

    Args:
        index: Position to set to 1 (0-based).
        num_classes: Total number of classes.

    Returns:
        1-D float32 array of shape (num_classes,).

    Raises:
        ValueError: If index is out of range.
    """
    if not (0 <= index < num_classes):
        raise ValueError(f"index {index} out of range for num_classes={num_classes}.")
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[index] = 1.0
    return vec


def extract_windows(signal: NDArray, window_size: int) -> NDArray:
    """Slide a fixed-size window over a 1-D signal.

    Uses numpy's sliding_window_view for a zero-copy strided view before
    casting to float32.

    Args:
        signal: 1-D array of shape (num_samples,).
        window_size: Number of samples per window.

    Returns:
        2-D float32 array of shape (num_samples - window_size + 1, window_size).

    Raises:
        ValueError: If window_size exceeds signal length.
    """
    if window_size > len(signal):
        raise ValueError(
            f"window_size {window_size} exceeds signal length {len(signal)}."
        )
    return np.lib.stride_tricks.sliding_window_view(signal, window_size).astype(np.float32)


def build_dataset(
    composite_signal: NDArray,
    clean_components: list[NDArray],
    window_size: int,
    num_components: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build the full (inputs, selectors, targets) dataset.

    num_examples = num_windows * num_components, where
    num_windows = len(composite_signal) - window_size + 1.

    Examples are grouped by component: all windows for component 0 come
    first, then all windows for component 1, and so on.

    Args:
        composite_signal: 1-D noisy composite signal.
        clean_components: List of 1-D clean sinusoid arrays.
        window_size: Context window length in samples.
        num_components: Number of sinusoidal components.

    Returns:
        inputs:    float32, shape (num_examples, window_size)
        selectors: float32, shape (num_examples, num_components)
        targets:   float32, shape (num_examples, window_size)

    Raises:
        ValueError: If len(clean_components) != num_components.
    """
    if len(clean_components) != num_components:
        raise ValueError(
            f"Expected {num_components} components, got {len(clean_components)}."
        )

    composite_windows = extract_windows(composite_signal, window_size)
    num_windows = len(composite_windows)

    inputs = np.tile(composite_windows, (num_components, 1)).astype(np.float32)

    selectors = np.repeat(
        np.eye(num_components, dtype=np.float32),
        num_windows,
        axis=0,
    )

    targets = np.vstack(
        [extract_windows(comp, window_size) for comp in clean_components]
    ).astype(np.float32)

    return inputs, selectors, targets
