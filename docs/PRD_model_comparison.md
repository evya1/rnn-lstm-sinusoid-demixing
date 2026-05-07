# PRD — Model Comparison

## Purpose

Define the architecture, input/output contracts, and fairness rules for the
three model families that are compared on the sinusoid demixing task.

## Scope

Covers modules:
- `src/rnn_lstm_sinusoid_demixing/models/fully_connected.py`
- `src/rnn_lstm_sinusoid_demixing/models/rnn_model.py`
- `src/rnn_lstm_sinusoid_demixing/models/lstm_model.py`

Training, evaluation, and visualisation are out of scope for this PRD.

## Model Architectures

### FullyConnectedModel (`fully_connected.py`)

A three-layer feed-forward network. The composite window and one-hot selector
are concatenated into a single flat vector before the first linear layer.

| Layer | In | Out | Activation |
|-------|-----|-----|------------|
| Linear 1 | `window_size + num_components` | `hidden_size` | ReLU |
| Linear 2 | `hidden_size` | `hidden_size` | ReLU |
| Linear 3 | `hidden_size` | `window_size` | — |

**Input:** `(batch, window_size + num_components)`  
**Output:** `(batch, window_size)`

### RNNModel (`rnn_model.py`)

A single-layer vanilla RNN followed by a time-distributed linear projection.
The one-hot selector is injected at every timestep alongside the composite
signal value.

**Input:** `(batch, seq_len, 1 + num_components)` — each timestep is `[σ_t, c_1…c_4]`  
**Output:** `(batch, seq_len)` — one predicted value per timestep

### LSTMModel (`lstm_model.py`)

Identical input/output contract to `RNNModel`. Replaces the vanilla RNN cell
with an LSTM cell, enabling the model to learn longer-range dependencies via
gated memory. Used to study whether gating improves demixing quality.

**Input:** `(batch, seq_len, 1 + num_components)`  
**Output:** `(batch, seq_len)`

## Input Preparation (caller responsibility)

The dataset builder (`dataset_builder.py`) returns raw
`(inputs, selectors, targets)` arrays. The training layer must prepare
model-specific tensors before each forward pass:

```python
# FC
x_fc = torch.cat([window, selector], dim=-1)          # (batch, W + C)

# RNN / LSTM
x_seq = torch.cat([window.unsqueeze(-1), selector     # (batch, W, 1 + C)
                   .unsqueeze(1).expand(-1, W, -1)], dim=-1)
```

## Fairness Rules

- All three models are trained on identical dataset splits with the same seed.
- All models use MSE loss.
- `hidden_size` defaults to `64` for all three (from `constants.py`).
- Random seeds are fixed before each model's training run.

## Constructor Interface

| Model | Key parameters |
|-------|---------------|
| `FullyConnectedModel` | `window_size`, `num_components`, `hidden_size=64` |
| `RNNModel` | `input_size` (= 1 + num_components), `hidden_size=64`, `num_layers=1` |
| `LSTMModel` | `input_size` (= 1 + num_components), `hidden_size=64`, `num_layers=1` |

All models implement `__repr__` returning their key hyperparameters for
logging and debugging.

## Acceptance Tests

- All three models instantiate without error from default parameters.
- `FullyConnectedModel.forward(x)` with `x` of shape `(B, W+C)` returns `(B, W)`.
- `RNNModel.forward(x)` with `x` of shape `(B, W, 1+C)` returns `(B, W)`.
- `LSTMModel.forward(x)` with `x` of shape `(B, W, 1+C)` returns `(B, W)`.
- All forward pass outputs are finite (no NaN or Inf).
- `__repr__` returns a non-empty string containing the class name.
