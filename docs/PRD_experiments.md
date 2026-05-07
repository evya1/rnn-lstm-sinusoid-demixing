# PRD — Experiments

> **TODO:** This stub must be completed before Phase 8 (visualization and results) begins.

## Purpose

Define the experiment protocol: which configurations to run, what to measure, and what to report.

## Baseline Experiment

```python
frequencies     = [1, 3, 5, 7]
noise_levels    = [0.00, 0.01, 0.05, 0.10, 0.20]
context_window  = 10
models          = ["fully_connected", "rnn", "lstm"]
random_seed     = 42
```

## Additional Frequency Scenarios

```python
frequency_scenarios = {
    "baseline":  [1, 3, 5, 7],
    "low_mixed": [0.5, 1, 3, 7],
    "wide_gap":  [1, 5, 20, 40],
    "close_low": [1, 2, 3, 4],
}
```

## Required Outputs

- [ ] Training and validation loss curves per model.
- [ ] Test MSE table (model × noise level).
- [ ] MSE vs noise level plot.
- [ ] Prediction vs target example plots for each model.
- [ ] Saved experiment config alongside each result.

## Acceptance Criteria

- Every reported result is reproducible from a documented command.
- Plots are saved to `results/` and referenced in `README.md`.
- Results are interpreted in writing (not just listed).
