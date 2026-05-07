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

- [ ] Implement Fully Connected model.
- [ ] Implement RNN model.
- [ ] Implement LSTM model.
- [ ] Add forward-pass tests for all models.

## Phase 7 — Training and Evaluation

Planned branch: `phase-07/training-evaluation`
Planned PR: TBD

- [ ] Implement training loop.
- [ ] Implement MSE metric.
- [ ] Implement model comparison script.
- [ ] Add smoke test for a short training run.

## Phase 8 — Visualization and Results

Planned branch: `phase-08/visualization-results`
Planned PR: TBD

- [ ] Plot clean sinusoidal components.
- [ ] Plot noisy composite signal.
- [ ] Plot prediction vs target examples.
- [ ] Plot loss curves.
- [ ] Plot MSE comparison across models and noise levels.

## Phase 9 — Final Submission

Planned branch: `phase-09/final-submission`
Planned PR: TBD

- [ ] Update README with final explanation and results.
- [ ] Add generated plots required for the report.
- [ ] Verify `uv run pytest`.
- [ ] Verify `uv run ruff check .`.
- [ ] Push final commits to GitHub.
- [ ] Submit the required PDF form with the GitHub repository link.
