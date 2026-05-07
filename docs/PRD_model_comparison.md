# PRD — Model Comparison

> **TODO:** This stub must be completed before Phase 6 (models) begins.

## Purpose

Define the requirements for implementing and fairly comparing the three model families.

## Scope

Covers modules: `src/.../models/`, `src/.../training/`, `src/.../evaluation/`.

## Models

| Model | Input | Output |
|---|---|---|
| FullyConnectedModel | `(batch, window_size + num_components)` | `(batch, window_size)` |
| RNNModel | `(batch, seq_len, 1 + num_components)` | `(batch, seq_len)` |
| LSTMModel | `(batch, seq_len, 1 + num_components)` | `(batch, seq_len)` |

## Fairness Rules

- All models must be trained on identical datasets and identical splits.
- All models must use the same loss function (MSE).
- Hidden sizes are configurable and must default to the same value.
- Random seeds must be identical across model runs.

## Requirements

- [ ] All three models support a forward pass with the documented input shapes.
- [ ] `Trainer` supports a `fit(train_loader, val_loader, num_epochs)` interface.
- [ ] `compute_mse(predictions, targets)` returns a scalar float.
- [ ] `compare_models(results)` produces a summary table.

## Acceptance Tests

- Forward passes return the expected output shapes without crashing.
- Smoke training run (3 epochs, small dataset) completes without error.
- MSE values are finite positive floats.
