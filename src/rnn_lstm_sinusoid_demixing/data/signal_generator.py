"""Clean and noisy sinusoidal signal generation.

Implements the noise-before-summation pipeline required by the assignment:

    S_i_noisy = A_i * sin(2*pi*f_i*t + phi_i) + noise_i
    Sigma_noisy = sum(S_i_noisy for i in components)
"""

from numpy.typing import NDArray


def generate_time_axis(sampling_rate: int, duration_seconds: float) -> NDArray:
    """Return a uniformly spaced time axis.

    Args:
        sampling_rate: Samples per second.
        duration_seconds: Total duration in seconds.

    Returns:
        1-D array of shape (num_samples,).
    """
    raise NotImplementedError("Phase 4: signal_generator.generate_time_axis")


def generate_clean_sinusoid(
    time: NDArray,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> NDArray:
    """Return a single clean sinusoidal component.

    Args:
        time: 1-D time axis array.
        frequency: Frequency in Hz.
        amplitude: Signal amplitude.
        phase: Initial phase in radians.

    Returns:
        1-D array of shape (num_samples,).
    """
    raise NotImplementedError("Phase 4: signal_generator.generate_clean_sinusoid")


def generate_noisy_composite(
    clean_components: list[NDArray],
    noise_level: float,
    random_seed: int = 42,
) -> tuple[list[NDArray], NDArray]:
    """Add noise to each component and return noisy components + their sum.

    Noise is added to each component *before* summation (assignment requirement).

    Args:
        clean_components: List of 1-D clean sinusoid arrays.
        noise_level: Standard deviation of Gaussian noise.
        random_seed: Seed for reproducibility.

    Returns:
        (noisy_components, composite_signal) where composite_signal is the
        element-wise sum of all noisy components.
    """
    raise NotImplementedError("Phase 4: signal_generator.generate_noisy_composite")
