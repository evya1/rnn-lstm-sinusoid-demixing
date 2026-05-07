# PRD — Experiments

## Purpose

Define the experiment protocol: which configurations to run, what to measure,
and what artefacts to produce and save.

## Scope

Covers modules: `src/.../training/`, `src/.../evaluation/`, `src/.../data/dataloader.py`,
`src/.../visualization/plots.py`, `src/.../experiments/runner.py`, `src/.../sdk/sdk.py`.

## Data Pipeline for Training

1. `build_signals(config)` → `(time, clean, noisy, composite)`
2. `build_dataset(composite, clean, window_size, num_components)` → `(inputs, selectors, targets)`
3. `split_dataset(inputs, selectors, targets, val_split, test_split, seed)` → train / val / test dicts
4. `make_loader(..., model_type, batch_size)` → `DataLoader` yielding `(model_input, target)` batches
5. `Trainer(model, optimizer, loss_fn).fit(train_loader, val_loader, num_epochs)` → loss history
6. `compute_mse(predictions, targets)` → scalar
7. `compare_models(results)` → sorted dict of `{model_name: test_mse}`

## Trainer Interface

| Method | Signature | Returns |
|--------|-----------|---------|
| `train_epoch` | `(loader) -> float` | mean train loss for the epoch |
| `evaluate` | `(loader) -> float` | mean loss over the loader |
| `fit` | `(train_loader, val_loader, num_epochs) -> dict` | `{"train": [...], "val": [...]}` |

The DataLoader passed to the Trainer must already yield model-ready tensors
(`make_loader` handles this via `prepare_fc_input` / `prepare_seq_input`).

## Baseline Experiment

```python
frequencies     = [1, 3, 5, 7]
noise_levels    = [0.00, 0.01, 0.05, 0.10, 0.20]
context_window  = 10
models          = ["fc", "rnn", "lstm"]
num_epochs      = 50
batch_size      = 64
random_seed     = 42
```

All three models use the **same** dataset split and seed for fair comparison.

## Additional Frequency Scenarios

```python
frequency_scenarios = {
    "baseline":  [1, 3, 5, 7],
    "low_mixed": [0.5, 1, 3, 7],
    "wide_gap":  [1, 5, 20, 40],
    "close_low": [1, 2, 3, 4],
}
```

## Visualization Pipeline

After `Trainer.fit` and test-set evaluation, the following plots are produced by
`visualization/plots.py` and saved to `results/`:

| Function | Output file | Input |
|----------|-------------|-------|
| `plot_signals` | `signals_clean.png`, `signals_noisy.png`, `signals_composite.png` | `build_signals` output |
| `plot_loss_curves` | `loss_curves_{model}.png` | `Trainer.fit` history |
| `plot_prediction_vs_target` | `prediction_vs_target_{model}.png` | one test example |
| `plot_mse_vs_noise` | `mse_vs_noise.png` | noise sweep results |

The experiment runner `experiments/runner.py` exports:
- `run_single(signal_config, training_config)` → per-model results dict
- `run_noise_sweep(training_config, noise_levels, base_config)` → `{model: [mse_per_level]}`

The SDK facade `sdk/sdk.py` orchestrates the full pipeline end-to-end and saves all artefacts.

## Required Outputs

- [x] Training and validation loss curves per model (saved to `results/`).
- [x] Test MSE table: model × noise level.
- [x] MSE vs noise level plot.
- [x] Prediction vs target example plots for each model.
- [x] Saved experiment config (JSON) alongside each result.

## Fairness Rules

- Identical dataset and splits for all three models.
- Same loss function: `nn.MSELoss`.
- Same hidden size (64) unless a specific ablation changes it.
- Seed reset before each model's training run.

## Acceptance Criteria

- `Trainer.fit` completes without error for all three model types.
- All reported MSE values are finite positive floats.
- `compare_models` returns a dict sorted by ascending test MSE.
- Every reported result is reproducible from a documented command.
- Plots are saved to `results/` and referenced in `README.md`.
