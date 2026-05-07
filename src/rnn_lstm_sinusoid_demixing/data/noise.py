"""Noise generation utilities for signal components."""

import numpy as np
from numpy.typing import NDArray


def gaussian_noise(
    shape: tuple[int, ...],
    std: float,
    random_seed: int = 42,
) -> NDArray:
    """Return a Gaussian noise array.

    Args:
        shape: Output array shape.
        std: Standard deviation.
        random_seed: Seed for reproducibility.

    Returns:
        Array of the given shape filled with N(0, std) samples.
    """
    rng = np.random.default_rng(random_seed)
    return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)
