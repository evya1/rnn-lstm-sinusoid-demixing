# RNN-LSTM Sinusoid Demixing

Exercise 01 — AI Agent Orchestration / Agentic AI Systems

A Python project that generates noisy composite sinusoidal signals and trains
**Fully Connected**, **RNN**, and **LSTM** models to reconstruct a selected
clean sinusoidal component from the mixture.

---

## Assignment Context

This project is submitted for Exercise 01 of the course
*AI Agent Orchestration / Agentic AI Systems*. It demonstrates professional
software development through the Vibe Coding lifecycle:

```
Idea → PRD → PLAN → TODO → Verify → Execute → Test → Document → Push
```

---

## Problem Statement

Four clean sinusoids are generated at frequencies **[1, 3, 5, 7] Hz**.
Independent Gaussian noise is added to **each component before summation**:

```
S_i_noisy = A_i · sin(2π f_i t + φ_i) + noise_i
Σ_noisy   = S_1_noisy + S_2_noisy + S_3_noisy + S_4_noisy
```

A model receives a short context window from `Σ_noisy` and a one-hot selector
vector `C`, and must predict the clean window of the selected component:

```
Input:  (Σ_noisy window, C)   →   Target: S_j_clean window
```

---

## Installation

Requires Python ≥ 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/evya1/rnn-lstm-sinusoid-demixing.git
cd rnn-lstm-sinusoid-demixing
uv sync --group dev
```

---

## Usage

```bash
# Run CLI entry point
uv run rnn-lstm-sinusoid-demixing

# Run all tests
uv run pytest

# Run only unit tests
uv run pytest tests/unit -v

# Run integration tests
uv run pytest tests/integration -v

# Lint
uv run ruff check .

# Coverage report
uv run pytest --cov=src --cov-report=term-missing
```

---

## Configuration

All experiment parameters live in `config/default.json` and are mirrored
as typed dataclasses in `src/.../shared/config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequencies` | `[1, 3, 5, 7]` | Hz, one per component |
| `sampling_rate` | `1000` | samples/second |
| `duration_seconds` | `10.0` | total signal length |
| `context_window` | `10` | samples per input window |
| `noise_level` | `0.1` | Gaussian noise std |
| `num_components` | `4` | number of sinusoids |
| `random_seed` | `42` | reproducibility seed |
| `batch_size` | `64` | training mini-batch size |
| `learning_rate` | `0.001` | Adam learning rate |
| `num_epochs` | `50` | training epochs |

---

## Data Generation

Implemented in `src/.../data/signal_generator.py` and `noise.py`.

```python
from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig

config = SignalConfig()  # loads defaults
time, clean, noisy, composite = build_signals(config)
# time:      (10000,)  float32 time axis
# clean:     list of 4 × (10000,) clean sinusoids
# noisy:     list of 4 × (10000,) noisy sinusoids
# composite: (10000,)  sum of noisy components
```

---

## Dataset Construction

Implemented in `src/.../data/dataset_builder.py`.

```python
from rnn_lstm_sinusoid_demixing.data.dataset_builder import build_dataset

inputs, selectors, targets = build_dataset(composite, clean, window_size=10, num_components=4)
# inputs:    (num_examples, 10)  — composite windows
# selectors: (num_examples,  4)  — one-hot component selectors
# targets:   (num_examples, 10)  — corresponding clean windows
```

`num_examples = (num_samples − window_size + 1) × num_components = 39 964`

---

## Model Architectures

All models created via `create_model()` in `src/.../models/factory.py`.

| Model | Input shape | Output shape | Key layers |
|-------|-------------|--------------|------------|
| `FullyConnectedModel` | `(B, 14)` | `(B, 10)` | Linear(14→64)→ReLU→Linear(64→64)→ReLU→Linear(64→10) |
| `RNNModel` | `(B, 10, 5)` | `(B, 10)` | RNN(5→64) + Linear(64→1) per timestep |
| `LSTMModel` | `(B, 10, 5)` | `(B, 10)` | LSTM(5→64) + Linear(64→1) per timestep |

FC input: `window (10) ‖ selector (4)` = 14 features.
RNN/LSTM input: each timestep `[σ_t, c_1, c_2, c_3, c_4]` = 5 features.

```python
from rnn_lstm_sinusoid_demixing.models.factory import create_model
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig

model = create_model("lstm", SignalConfig(), TrainingConfig())
```

---

## Training & Evaluation

Implemented in `src/.../training/trainer.py`, `data/dataloader.py`,
`evaluation/metrics.py`, and `evaluation/compare.py`.

```python
import torch
from rnn_lstm_sinusoid_demixing.data.dataloader import split_dataset, make_loader
from rnn_lstm_sinusoid_demixing.data.dataset_builder import build_dataset
from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
from rnn_lstm_sinusoid_demixing.evaluation.compare import compare_models
from rnn_lstm_sinusoid_demixing.evaluation.metrics import compute_mse
from rnn_lstm_sinusoid_demixing.models.factory import create_model
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig
from rnn_lstm_sinusoid_demixing.training.losses import mse_loss
from rnn_lstm_sinusoid_demixing.training.trainer import Trainer

sc, tc = SignalConfig(), TrainingConfig()
_, clean, _, composite = build_signals(sc)
inputs, selectors, targets = build_dataset(composite, clean, sc.context_window, sc.num_components)
splits = split_dataset(inputs, selectors, targets, random_seed=tc.random_seed)

results = {}
for model_type in ("fc", "rnn", "lstm"):
    model = create_model(model_type, sc, tc)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.learning_rate)
    trainer = Trainer(model, optimizer, mse_loss())

    train_loader = make_loader(*splits["train"], model_type, tc.batch_size)
    val_loader   = make_loader(*splits["val"],   model_type, tc.batch_size, shuffle=False)
    history = trainer.fit(train_loader, val_loader, tc.num_epochs)
    # history == {"train": [loss_epoch1, ...], "val": [loss_epoch1, ...]}

    test_loader = make_loader(*splits["test"], model_type, tc.batch_size, shuffle=False)
    results[model_type] = trainer.evaluate(test_loader)

ranking = compare_models(results)
# ranking == {"fc": 0.021, "lstm": 0.018, "rnn": 0.024}  (sorted ascending)
```

**Trainer interface:**

| Method | Signature | Returns |
|--------|-----------|---------|
| `train_epoch` | `(loader) -> float` | Mean train loss for the epoch |
| `evaluate` | `(loader) -> float` | Mean loss over the loader |
| `fit` | `(train_loader, val_loader, num_epochs) -> dict` | `{"train": [...], "val": [...]}` |

The `DataLoader` from `make_loader` already yields model-ready tensors
(`prepare_fc_input` / `prepare_seq_input` are applied once at construction).

---

## Project Structure

```
src/rnn_lstm_sinusoid_demixing/
├── constants.py          # project-wide defaults
├── main.py               # CLI entry point (typer)
├── data/
│   ├── signal_generator.py   # Phase 04 ✓
│   ├── noise.py              # Phase 04 ✓
│   ├── dataset_builder.py    # Phase 05 ✓
│   └── dataloader.py         # Phase 07 ✓
├── models/
│   ├── fully_connected.py    # Phase 06 ✓
│   ├── rnn_model.py          # Phase 06 ✓
│   ├── lstm_model.py         # Phase 06 ✓
│   ├── input_prep.py         # Phase 06 ✓
│   └── factory.py            # Phase 06 ✓
├── training/
│   ├── trainer.py            # Phase 07 ✓
│   └── losses.py             # Phase 07 ✓
├── evaluation/
│   ├── metrics.py            # Phase 07 ✓
│   └── compare.py            # Phase 07 ✓
├── experiments/
│   └── runner.py             # Phase 08 ✓
├── visualization/
│   └── plots.py              # Phase 08 ✓
└── shared/
    ├── config.py             # SignalConfig, TrainingConfig
    └── paths.py              # project path helpers
```

---

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 01 | Repository setup | ✅ merged |
| 02 | Documentation (PRD, PLAN, TODO) | ✅ merged |
| 03 | Project skeleton | ✅ merged |
| 04 | Data generation | ✅ merged |
| 05 | Dataset builder | ✅ merged |
| 06 | Models (FC, RNN, LSTM) | ✅ merged |
| 07 | Training & evaluation loop | ✅ merged |
| 08 | Visualization & results | ✅ merged |
| 09 | Final submission polish | 🔜 planned |

---

## Testing

```
uv run pytest            →  190 passed  (unit + integration)
uv run ruff check .      →  All checks passed
uv run pytest --cov=src  →  91% coverage
```

Unit tests cover: config validation, path helpers, signal generation,
dataset construction, all three models, input preparation, model factory,
dataloader splitting, MSE metric, model comparison, and all four plot functions.

Integration tests cover: full pipeline from `build_signals` →
`build_dataset` → `create_model` → forward pass; full training loop
(Trainer.fit → compute_mse → compare_models); and `run_single` /
`run_noise_sweep` with tiny configs.

---

## Experiment Protocol

To reproduce all results and regenerate all plots:

```bash
uv run rnn-lstm-sinusoid-demixing
```

Experiment configuration (saved in `results/experiment_config.json`):

```python
frequencies    = [1, 3, 5, 7]          # Hz per component
noise_levels   = [0.00, 0.01, 0.05, 0.10, 0.20]
context_window = 10                     # samples per window
models         = ["fc", "rnn", "lstm"]
num_epochs     = 50
batch_size     = 64
random_seed    = 42
```

All three models use **identical dataset splits and the same seed** for fair comparison
(see Fairness Rules in `docs/PRD_experiments.md`).

---

## Results

### Signals

**Clean sinusoidal components (1, 3, 5, 7 Hz):**

![Clean Components](results/signals_clean.png)

**Noisy components (noise σ = 0.1) and composite sum:**

![Noisy Components](results/signals_noisy.png)

![Noisy Composite Signal](results/signals_composite.png)

---

### Training & Validation Loss Curves

All models trained for 50 epochs, batch size 64, learning rate 0.001.

![FC Loss Curves](results/loss_curves_fc.png)

![RNN Loss Curves](results/loss_curves_rnn.png)

![LSTM Loss Curves](results/loss_curves_lstm.png)

---

### Prediction vs Target Examples

Each plot shows the model's reconstruction of a randomly selected clean
sinusoidal window from the held-out test set (noise σ = 0.1).

![FC Prediction](results/prediction_vs_target_fc.png)

![RNN Prediction](results/prediction_vs_target_rnn.png)

![LSTM Prediction](results/prediction_vs_target_lstm.png)

---

### Test MSE at noise σ = 0.1

| Model | Test MSE |
|-------|----------|
| **FC** | **0.2344** |
| LSTM | 0.2634 |
| RNN | 0.2696 |

The Fully Connected model achieves the lowest test MSE at this noise level.
Exact values are in `results/mse_summary.json`.

---

### MSE vs Noise Level

![MSE vs Noise Level](results/mse_vs_noise.png)

Full sweep data (noise levels 0.00, 0.01, 0.05, 0.10, 0.20) in
`results/mse_noise_sweep.json`.

---

## Conclusions

**FC outperforms RNN and LSTM** on this demixing task at moderate noise (σ=0.1).
This is expected: the one-hot selector gives the model a direct frequency hint,
so the demixing problem reduces to a linear projection from the composite window.
A simple MLP is sufficient to learn this mapping.

**RNN and LSTM** carry the overhead of sequential computation (recurrent hidden
state across 10 timesteps) without benefiting from long-range temporal structure —
the 10-sample context window is too short for recurrence to add value over a
direct feed-forward path.

**The task is partially solved**: all three models predict the correct shape and
phase of the target sinusoid, but residual noise and the short context window
limit reconstruction fidelity. Higher sampling rates, longer context windows, or
a denoising pre-processing step would likely reduce MSE further.

---

## Limitations

- Context window of 10 samples is short: only one full cycle is visible for
  the 1 Hz component at 10 Hz sampling, making reconstruction harder.
- Amplitude is fixed at 1.0; varying amplitudes are not tested.
- Only baseline frequency scenario (`[1, 3, 5, 7]`) was run; additional
  frequency scenarios (`low_mixed`, `wide_gap`, `close_low`) are defined in
  `constants.py` but not yet benchmarked.
- No GPU training; runs on CPU only.

---

## AI Usage Disclosure

This project was developed using **Claude Code** (claude-sonnet-4-6) as an
AI coding assistant throughout all phases. All AI prompts and decisions are
logged in `docs/PROMPTS.md`. All generated code was reviewed and accepted
by the student before committing.
