# PRD — Signal Generation

> **TODO:** This stub must be completed before Phase 4 (data generation) begins.

## Purpose

Define the requirements for generating clean sinusoidal signals, adding per-component noise, and producing the noisy composite signal used as the model input.

## Scope

Covers modules: `src/.../data/signal_generator.py`, `src/.../data/noise.py`.

## Core Semantic Rule (non-negotiable)

Noise is added to **each individual component** before summation:

```
S_i_noisy = A_i * sin(2*pi*f_i*t + phi_i) + noise_i
Sigma_noisy = sum(S_i_noisy)
```

Implementing "sum first, then add noise once" as the main pipeline is forbidden.

## Requirements

- [ ] Generate a time axis from sampling rate and duration.
- [ ] Generate one clean sinusoid per frequency with configurable amplitude and phase.
- [ ] Add independent Gaussian noise to each component.
- [ ] Return both the list of noisy components and their element-wise sum.
- [ ] Support configurable noise level, frequencies, sampling rate, and duration via `SignalConfig`.
- [ ] All outputs must be float32 NumPy arrays with no NaN or Inf values.
- [ ] Signal generation must be deterministic given a random seed.

## Acceptance Tests

- Generated clean sinusoids have shape `(num_samples,)` with `num_samples = sampling_rate * duration_seconds`.
- Noisy composite signal equals element-wise sum of noisy components.
- No NaN or Inf values appear in any output.
- Output dtype is float32.
