# Product Requirements Document — RNN-LSTM Sinusoid Demixing

## 1. Project Goal

Build a Python project that generates synthetic sinusoidal signals, combines them into a noisy composite signal, and trains neural network models to reconstruct a selected clean sinusoidal component from the noisy mixture.

The project compares three model families:

- Fully Connected network
- RNN
- LSTM

## 2. Problem Description

The system generates four clean sinusoidal signals. Each signal may receive noise, and the noisy signals are summed into one composite noisy signal.

For each training example, the model receives:

- A short context window from the noisy composite signal.
- A one-hot selector vector that indicates which clean sinusoidal component should be reconstructed.

The target output is the clean window of the selected sinusoidal component.

## 3. Functional Requirements

- Generate four clean sinusoidal signals.
- Add configurable noise to the individual sinusoidal signals.
- Build a noisy composite signal by summing the noisy components.
- Construct a dataset using fixed-length context windows.
- Represent the selected target component using a one-hot selector.
- Train and evaluate Fully Connected, RNN, and LSTM models.
- Report quantitative metrics, mainly MSE.
- Generate plots comparing prediction and target signals.
- Compare model behavior under different noise levels and frequency configurations.

## 4. Non-Functional Requirements

- Use Python with uv for dependency and environment management.
- Keep the code modular and maintainable.
- Store implementation under the `src/` directory.
- Store tests under the `tests/` directory.
- Store documentation under the `docs/` directory.
- Avoid committing generated heavy artifacts such as model checkpoints.
- Include clear instructions in `README.md`.

## 5. Success Criteria

The project is considered successful if:

- The data generation pipeline works correctly.
- The dataset builder produces valid inputs and targets.
- All three model types can run a forward pass.
- Training can be executed without crashes.
- Evaluation metrics and plots are produced.
- The README explains the experiments and conclusions clearly.
