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

## Project Structure

```
src/rnn_lstm_sinusoid_demixing/
├── constants.py          # project-wide defaults
├── main.py               # CLI entry point (typer)
├── data/
│   ├── signal_generator.py   # Phase 04 ✓
│   ├── noise.py              # Phase 04 ✓
│   └── dataset_builder.py    # Phase 05 ✓
├── models/
│   ├── fully_connected.py    # Phase 06 ✓
│   ├── rnn_model.py          # Phase 06 ✓
│   ├── lstm_model.py         # Phase 06 ✓
│   ├── input_prep.py         # Phase 06 ✓
│   └── factory.py            # Phase 06 ✓
├── training/
│   ├── trainer.py            # Phase 07 (stub)
│   └── losses.py             # Phase 07 (stub)
├── evaluation/
│   ├── metrics.py            # Phase 07 (stub)
│   └── compare.py            # Phase 07 (stub)
├── visualization/
│   └── plots.py              # Phase 08 (stub)
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
| 07 | Training & evaluation loop | 🔜 next |
| 08 | Visualization & results | 🔜 planned |
| 09 | Final submission polish | 🔜 planned |

---

## Testing

```
uv run pytest            →  131 passed  (unit + integration)
uv run ruff check .      →  All checks passed
```

Unit tests cover: config validation, path helpers, signal generation,
dataset construction, all three models, input preparation, and model factory.

Integration tests cover: full pipeline from `build_signals` →
`build_dataset` → `create_model` → forward pass for all three model families.

---

## Experiment Protocol *(Phase 07–08)*

Planned experiments:

- Baseline noise sweep: `noise_levels = [0.00, 0.01, 0.05, 0.10, 0.20]`
- Frequency scenarios: `baseline`, `low_mixed`, `wide_gap`, `close_low`
- All models trained on identical splits with the same random seed

Results, plots, and conclusions will be added after Phase 07–08.

---

## AI Usage Disclosure

This project was developed using **Claude Code** (claude-sonnet-4-6) as an
AI coding assistant throughout all phases. All AI prompts and decisions are
logged in `docs/PROMPTS.md`. All generated code was reviewed and accepted
by the student before committing.
