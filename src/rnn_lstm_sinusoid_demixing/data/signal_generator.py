"""Clean and noisy sinusoidal signal generation.

Implements the noise-before-summation pipeline required by the assignment:

    S_i_noisy = A_i * sin(2*pi*f_i*t + phi_i) + noise_i
    Sigma_noisy = sum(S_i_noisy for i in components)
"""

import numpy as np
from numpy.typing import NDArray

from rnn_lstm_sinusoid_demixing.data.noise import gaussian_noise
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig


def generate_time_axis(sampling_rate: int, duration_seconds: float) -> NDArray:
    """Return a uniformly spaced time axis of shape (num_samples,).

    Args:
        sampling_rate: Samples per second.
        duration_seconds: Total duration in seconds.

    Returns:
        1-D float32 array where t[k] = k / sampling_rate.
    """
    num_samples = int(sampling_rate * duration_seconds)
    return np.linspace(0.0, duration_seconds, num_samples, endpoint=False, dtype=np.float32)


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
        1-D float32 array of shape (num_samples,).
    """
    return (amplitude * np.sin(2.0 * np.pi * frequency * time + phase)).astype(np.float32)


def generate_noisy_composite(
    clean_components: list[NDArray],
    noise_level: float,
    random_seed: int = 42,
) -> tuple[list[NDArray], NDArray]:
    """Add noise to each component and return noisy components + their sum.

    Each component receives independent noise with a unique seed offset
    (random_seed + component_index) to ensure independence.

    Args:
        clean_components: List of 1-D clean sinusoid arrays.
        noise_level: Standard deviation of Gaussian noise.
        random_seed: Base seed; component i uses random_seed + i.

    Returns:
        (noisy_components, composite_signal) where composite_signal is the
        element-wise sum of all noisy components.
    """
    noisy = [
        component + gaussian_noise(component.shape, std=noise_level, random_seed=random_seed + i)
        for i, component in enumerate(clean_components)
    ]
    composite = np.add.reduce(noisy).astype(np.float32)
    return noisy, composite


def build_signals(
    config: SignalConfig,
    amplitudes: list[float] | None = None,
    phases: list[float] | None = None,
) -> tuple[NDArray, list[NDArray], list[NDArray], NDArray]:
    """Build all signals from a SignalConfig.

    Args:
        config: Signal generation configuration.
        amplitudes: Per-component amplitudes. Defaults to all 1.0.
        phases: Per-component initial phases in radians. Defaults to all 0.0.

    Returns:
        (time, clean_components, noisy_components, composite_signal)
    """
    if amplitudes is None:
        amplitudes = [1.0] * config.num_components
    if phases is None:
        phases = [0.0] * config.num_components

    time = generate_time_axis(config.sampling_rate, config.duration_seconds)
    clean = [
        generate_clean_sinusoid(time, f, a, p)
        for f, a, p in zip(config.frequencies, amplitudes, phases)
    ]
    noisy, composite = generate_noisy_composite(clean, config.noise_level, config.random_seed)
    return time, clean, noisy, composite
