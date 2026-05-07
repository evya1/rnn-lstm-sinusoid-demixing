# TODO — RNN-LSTM Sinusoid Demixing

## Phase 1 — Repository Setup

- [x] Create GitHub repository.
- [x] Clone repository locally.
- [x] Initialize Python project with uv.
- [x] Add initial README.
- [x] Add `.gitignore`.
- [x] Add first signed commit.

Completed in commit: `1de5870`

## Phase 2 — Documentation Setup

Completed in commit: `c5aac32`

- [x] Create `docs/PRD.md`.
- [x] Create `docs/PLAN.md`.
- [x] Create `docs/TODO.md`.
- [x] Create `docs/PROMPTS.md`.
- [x] Review documentation and align it with assignment requirements.

## Phase 3 — Project Skeleton

Planned branch: `phase-03/project-skeleton`
Planned PR: TBD

- [x] Create source subpackages: `data`, `models`, `training`, `evaluation`, `visualization`, `shared`, `sdk`.
- [x] Create unit and integration test directories.
- [x] Add placeholder implementation files (stubs with docstrings).
- [x] Add project constants/configuration (`constants.py`, `shared/config.py`, `config/default.json`).
- [x] Add `torch` and `pytest-cov` to `pyproject.toml`.
- [x] Add `results/.gitkeep` and `assets/.gitkeep`.
- [x] Create stub sub-PRDs: `PRD_signal_generation.md`, `PRD_dataset_builder.md`, `PRD_model_comparison.md`, `PRD_experiments.md`.
- [x] Add smoke tests: `test_imports.py`, `test_config.py`, `test_paths.py`.
- [x] Flesh out `docs/PRD_signal_generation.md` before Phase 4.

## Phase 4 — Data Generation

Planned branch: `phase-04/data-generation`
Planned PR: TBD

- [x] Implement clean sinusoid generation.
- [x] Implement noise injection.
- [x] Implement noisy composite signal creation.
- [x] Add unit tests for signal generation.

## Phase 5 — Dataset Builder

Planned branch: `phase-05/dataset-builder`
Planned PR: TBD

- [x] Implement fixed-size context windows.
- [x] Implement one-hot selector construction.
- [x] Implement target clean-window extraction.
- [x] Add unit tests for dataset shapes and correctness.

## Phase 6 — Models

Planned branch: `phase-06/models`
Planned PR: TBD

- [x] Implement Fully Connected model.
- [x] Implement RNN model.
- [x] Implement LSTM model.
- [x] Add forward-pass tests for all models.
- [x] Add input preparation utilities (`models/input_prep.py`).
- [x] Add model factory (`models/factory.py`).

## Phase 7 — Training and Evaluation

Planned branch: `phase-07/training-evaluation`
Planned PR: TBD

- [x] Flesh out `docs/PRD_experiments.md` with full protocol.
- [x] Implement `data/dataloader.py`: `split_dataset` and `make_loader`.
- [x] Implement `training/trainer.py`: `Trainer.train_epoch`, `evaluate`, `fit`.
- [x] Implement `evaluation/metrics.py`: `compute_mse`.
- [x] Implement `evaluation/compare.py`: `compare_models`.
- [x] Add unit tests: `test_dataloader.py`, `test_metrics.py`, `test_compare.py`.
- [x] Add integration smoke test: `tests/integration/test_training.py`.
- [x] All 170 tests pass; ruff clean.

## Phase 8 — Visualization and Results

Planned branch: `phase-08/visualization-results`
Planned PR: TBD

- [x] Flesh out `docs/PRD_experiments.md` with visualization pipeline.
- [x] Implement `visualization/plots.py`: 4 plot functions.
- [x] Add `experiments/runner.py`: `run_single`, `run_noise_sweep`, `ModelResult`.
- [x] Implement `sdk/sdk.py` `DemixingSDK.run()` end-to-end pipeline.
- [x] Wire `main.py` CLI to `DemixingSDK().run()`.
- [x] Plot clean sinusoidal components (`results/signals_clean.png`).
- [x] Plot noisy composite signal (`results/signals_composite.png`, `signals_noisy.png`).
- [x] Plot prediction vs target examples for FC, RNN, LSTM.
- [x] Plot loss curves for FC, RNN, LSTM.
- [x] Plot MSE comparison across noise levels (`results/mse_vs_noise.png`).
- [x] Save `results/mse_summary.json` and `results/experiment_config.json`.
- [x] Add unit tests: `test_plots.py` (11 tests).
- [x] Add integration tests: `test_experiment.py` (8 tests).
- [x] Update README with embedded plots, MSE table, conclusions, limitations.
- [x] All 190 tests pass; ruff clean; 91% coverage.

## Phase 9 — Final Submission

Planned branch: `phase-09/final-submission`
Planned PR: TBD

- [ ] Update README with final explanation and results.
- [ ] Add generated plots required for the report.
- [ ] Verify `uv run pytest`.
- [ ] Verify `uv run ruff check .`.
- [ ] Push final commits to GitHub.
- [ ] Submit the required PDF form with the GitHub repository link.
