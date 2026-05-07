# PRD — Signal Generation

## Purpose

Define the requirements for generating clean sinusoidal signals, adding
per-component noise, and producing the noisy composite signal used as the
model input.

## Scope

Covers modules:
- `src/rnn_lstm_sinusoid_demixing/data/signal_generator.py`
- `src/rnn_lstm_sinusoid_demixing/data/noise.py`

## Core Semantic Rule (non-negotiable)

Noise is added to **each individual component before summation**:

```
S_i_noisy = A_i * sin(2*pi*f_i*t + phi_i) + noise_i
Sigma_noisy = sum(S_i_noisy for i in 0..N-1)
```

Implementing "sum first, then add noise once" as the main pipeline is
forbidden by the assignment specification.

## Function Interface

### `generate_time_axis(sampling_rate, duration_seconds) -> NDArray`

| Parameter | Type | Description |
|-----------|------|-------------|
| `sampling_rate` | `int` | Samples per second (must be > 0) |
| `duration_seconds` | `float` | Total duration in seconds (must be > 0) |

Returns a 1-D float32 array of shape `(num_samples,)` where
`num_samples = int(sampling_rate * duration_seconds)` and
`t[k] = k / sampling_rate`.

### `generate_clean_sinusoid(time, frequency, amplitude, phase) -> NDArray`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time` | `NDArray` | — | 1-D time axis from `generate_time_axis` |
| `frequency` | `float` | — | Frequency in Hz |
| `amplitude` | `float` | `1.0` | Signal amplitude |
| `phase` | `float` | `0.0` | Initial phase in radians |

Returns `amplitude * sin(2π * frequency * time + phase)` as float32.

### `generate_noisy_composite(clean_components, noise_level, random_seed) -> tuple[list[NDArray], NDArray]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clean_components` | `list[NDArray]` | — | One 1-D array per component |
| `noise_level` | `float` | — | Std of Gaussian noise (`0.0` = no noise) |
| `random_seed` | `int` | `42` | Base seed; component `i` uses `seed + i` |

Returns `(noisy_components, composite_signal)`.

### `build_signals(config, amplitudes, phases) -> tuple[NDArray, list, list, NDArray]`

Convenience function that orchestrates the full pipeline for a given
`SignalConfig`. Intended for use by the dataset builder.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `SignalConfig` | — | Experiment configuration |
| `amplitudes` | `list[float] \| None` | `None` | Per-component amplitudes (default: all 1.0) |
| `phases` | `list[float] \| None` | `None` | Per-component phases in radians (default: all 0.0) |

Returns `(time, clean_components, noisy_components, composite_signal)`.

## Baseline Configuration

```python
frequencies = [1.0, 3.0, 5.0, 7.0]
sampling_rate = 1000          # samples/sec
duration_seconds = 10.0       # seconds → 10 000 samples
noise_level = 0.1             # default; swept in experiments
num_components = 4
random_seed = 42
```

## Output Contracts

- All returned arrays are `float32`.
- No `NaN` or `Inf` values appear in any output.
- `composite_signal` is exactly the element-wise sum of `noisy_components`.
- Results are deterministic given the same `random_seed`.
- With `noise_level = 0.0`, `noisy_components[i] == clean_components[i]`.

## Acceptance Tests

- Clean sinusoid shape is `(num_samples,)`.
- `composite == np.stack(noisy).sum(axis=0)`.
- No `NaN` or `Inf` in any output.
- All outputs are `float32`.
- Two calls with the same seed return identical arrays.
- Two calls with different seeds return different arrays (for `noise_level > 0`).
- Zero noise level leaves components unchanged.
