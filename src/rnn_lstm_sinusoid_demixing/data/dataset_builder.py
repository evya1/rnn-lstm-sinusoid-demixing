"""Dataset construction from sinusoidal signals.

Builds (input_window, selector, target_window) triples for model training.

Input representation:
    FC:       flat vector [sigma_window (10,), one_hot (4,)] → shape (14,)
    RNN/LSTM: sequence tensor, shape (seq_len=10, features=5)
              where each timestep is [sigma_t, c_1, c_2, c_3, c_4]
"""

from numpy.typing import NDArray


def make_one_hot(index: int, num_classes: int) -> NDArray:
    """Return a one-hot vector of length num_classes with index set to 1.

    Args:
        index: Position to set to 1 (0-based).
        num_classes: Total number of classes.

    Returns:
        1-D float32 array of shape (num_classes,).
    """
    raise NotImplementedError("Phase 5: dataset_builder.make_one_hot")


def extract_windows(
    signal: NDArray,
    window_size: int,
) -> NDArray:
    """Slide a fixed-size window over a 1-D signal.

    Args:
        signal: 1-D array of shape (num_samples,).
        window_size: Number of samples per window.

    Returns:
        2-D array of shape (num_windows, window_size).
    """
    raise NotImplementedError("Phase 5: dataset_builder.extract_windows")


def build_dataset(
    composite_signal: NDArray,
    clean_components: list[NDArray],
    window_size: int,
    num_components: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build the full (inputs, selectors, targets) dataset.

    Args:
        composite_signal: 1-D noisy composite signal.
        clean_components: List of 1-D clean sinusoid arrays (length = num_components).
        window_size: Context window length in samples.
        num_components: Number of sinusoidal components.

    Returns:
        inputs:    shape (num_examples, window_size)
        selectors: shape (num_examples, num_components)
        targets:   shape (num_examples, window_size)
    """
    raise NotImplementedError("Phase 5: dataset_builder.build_dataset")
