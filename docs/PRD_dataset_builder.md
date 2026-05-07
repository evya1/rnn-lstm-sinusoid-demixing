# PRD — Dataset Builder

> **TODO:** This stub must be completed before Phase 5 (dataset builder) begins.

## Purpose

Define requirements for constructing the sliding-window dataset used to train and evaluate all models.

## Scope

Covers module: `src/.../data/dataset_builder.py`.

## Input Representation

Each training example is a triple `(input, selector, target)`:

| Field | FC shape | RNN/LSTM shape |
|---|---|---|
| input | `(window_size + num_components,)` | `(seq_len, 1 + num_components)` |
| selector | `(num_components,)` one-hot | embedded in sequence |
| target | `(window_size,)` | `(seq_len,)` |

## Requirements

- [ ] `extract_windows(signal, window_size)` → shape `(num_windows, window_size)`.
- [ ] `make_one_hot(index, num_classes)` → valid one-hot float32 vector.
- [ ] `build_dataset(...)` returns aligned `(inputs, selectors, targets)` arrays.
- [ ] Target window corresponds to the clean component selected by the one-hot vector.
- [ ] All split boundaries (train/val/test) are reproducible given the random seed.
- [ ] No data leakage between splits.

## Acceptance Tests

- Dataset shapes are correct for both FC and RNN/LSTM representations.
- One-hot selectors are valid (exactly one 1, rest 0).
- Target windows correspond to the correct clean component.
- No NaN or Inf values.
