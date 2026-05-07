# Prompt Log — RNN-LSTM Sinusoid Demixing

This file records important AI prompts and AI-assisted decisions used during the project.

## Prompt 1 — Assignment Analysis

Purpose: Understand the requirements of Exercise 01 from the assignment files and lecture transcript.

Summary:
Asked the AI assistant to analyze what needs to be done in Exercise 01, including repository setup, documentation requirements, signal generation, model comparison, and submission instructions.

Outcome:
The project was interpreted as a software project for comparing Fully Connected, RNN, and LSTM models on reconstructing clean sinusoidal components from noisy composite signals.

## Prompt 2 — Repository Setup

Purpose: Decide how to initialize the GitHub repository using Python and uv.

Summary:
Asked whether to open a new PyCharm project or clone the GitHub repository first, and whether to use `uv init --package`.

Outcome:
The repository was cloned locally and initialized as a uv-based Python package with a src-based layout.

## Prompt 3 — Git and Commit Hygiene

Purpose: Recover from mistakenly staging `.venv` and decide what should go into the first commit.

Summary:
Asked how to unstage `.venv` without deleting local files, how to configure `.gitignore`, and what should be included in the first signed commit.

Outcome:
`.venv` was removed from the Git index, `.gitignore` was added, and the first commit was limited to the project foundation.

## Prompt 4 — Documentation Planning

Purpose: Create the first version of the required documentation files.

Summary:
Asked what to do after creating `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, and `docs/PROMPTS.md`.

Outcome:
The documentation files were populated with initial meaningful content before committing them.

## Prompt 6 — Phase 3: Project Skeleton

**Date:** 2026-05-07
**Tool/Agent:** Claude Code
**Purpose:** Build the full module skeleton for the project so that all subsequent phases can add logic without structural changes.
**Prompt Summary:** Asked Claude to read CLAUDE.md, README.md, docs/PRD.md, docs/PLAN.md, docs/TODO.md, inspect the repository state, identify gaps, plan Phase 3, and implement the skeleton after approval.
**Files Affected:**
- `pyproject.toml` — added `torch>=2.0`, `pytest-cov>=6.0`
- `src/rnn_lstm_sinusoid_demixing/__init__.py` — thinned; re-exports `main`
- `src/rnn_lstm_sinusoid_demixing/main.py` — CLI entry point (typer)
- `src/rnn_lstm_sinusoid_demixing/constants.py` — project-wide defaults
- `src/rnn_lstm_sinusoid_demixing/shared/config.py` — `SignalConfig`, `TrainingConfig`
- `src/rnn_lstm_sinusoid_demixing/shared/paths.py` — path helpers
- `src/rnn_lstm_sinusoid_demixing/data/signal_generator.py` — stub
- `src/rnn_lstm_sinusoid_demixing/data/noise.py` — `gaussian_noise` utility
- `src/rnn_lstm_sinusoid_demixing/data/dataset_builder.py` — stub
- `src/rnn_lstm_sinusoid_demixing/models/fully_connected.py` — stub `FullyConnectedModel`
- `src/rnn_lstm_sinusoid_demixing/models/rnn_model.py` — stub `RNNModel`
- `src/rnn_lstm_sinusoid_demixing/models/lstm_model.py` — stub `LSTMModel`
- `src/rnn_lstm_sinusoid_demixing/training/trainer.py` — stub `Trainer`
- `src/rnn_lstm_sinusoid_demixing/training/losses.py` — `mse_loss()`
- `src/rnn_lstm_sinusoid_demixing/evaluation/metrics.py` — stub `compute_mse`
- `src/rnn_lstm_sinusoid_demixing/evaluation/compare.py` — stub `compare_models`
- `src/rnn_lstm_sinusoid_demixing/visualization/plots.py` — stub plot functions
- `src/rnn_lstm_sinusoid_demixing/sdk/sdk.py` — stub `DemixingSDK`
- `config/default.json` — baseline experiment configuration
- `tests/unit/test_imports.py`, `test_config.py`, `test_paths.py` — smoke tests
- `results/.gitkeep`, `assets/.gitkeep`
- `docs/PRD_signal_generation.md`, `docs/PRD_dataset_builder.md`, `docs/PRD_model_comparison.md`, `docs/PRD_experiments.md` — stub sub-PRDs
- `docs/TODO.md`, `docs/PROMPTS.md` — updated
**Outcome:** Full project skeleton created. All subpackages import cleanly. Config dataclasses validate inputs. Path helpers resolve correctly. No ML logic implemented yet.
**Human Review:** Student should verify `uv run pytest tests/unit -v` and `uv run ruff check .` pass before committing.

## Prompt 5 — Branch and Pull Request Workflow

Purpose: Decide whether to continue development directly on `main` or use feature branches and pull requests for each major phase.

Summary:
Asked whether it is better to create a dedicated branch and pull request for each major project phase.

Outcome:
The project will continue with short-lived phase branches and pull requests. Each major phase will be implemented in a dedicated branch, pushed to GitHub, reviewed through a pull request, and merged into `main`.
