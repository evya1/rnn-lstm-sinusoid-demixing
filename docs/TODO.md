# TODO — RNN-LSTM Sinusoid Demixing

## Phase 1 — Repository Setup

- [x] Create GitHub repository.
- [x] Clone repository locally.
- [x] Initialize Python project with uv.
- [x] Add initial README.
- [x] Add `.gitignore`.
- [x] Add first signed commit.

## Phase 2 — Documentation Setup

- [x] Create `docs/PRD.md`.
- [x] Create `docs/PLAN.md`.
- [x] Create `docs/TODO.md`.
- [x] Create `docs/PROMPTS.md`.
- [ ] Review documentation and align it with assignment requirements.

## Phase 3 — Project Skeleton

- [ ] Create source subpackages: `data`, `models`, `training`, `evaluation`, `visualization`.
- [ ] Create unit and integration test directories.
- [ ] Add placeholder implementation files.
- [ ] Add project constants/configuration.

## Phase 4 — Data Generation

- [ ] Implement clean sinusoid generation.
- [ ] Implement noise injection.
- [ ] Implement noisy composite signal creation.
- [ ] Add unit tests for signal generation.

## Phase 5 — Dataset Builder

- [ ] Implement fixed-size context windows.
- [ ] Implement one-hot selector construction.
- [ ] Implement target clean-window extraction.
- [ ] Add unit tests for dataset shapes and correctness.

## Phase 6 — Models

- [ ] Implement Fully Connected model.
- [ ] Implement RNN model.
- [ ] Implement LSTM model.
- [ ] Add forward-pass tests for all models.

## Phase 7 — Training and Evaluation

- [ ] Implement training loop.
- [ ] Implement MSE metric.
- [ ] Implement model comparison script.
- [ ] Add smoke test for a short training run.

## Phase 8 — Visualization and Results

- [ ] Plot clean sinusoidal components.
- [ ] Plot noisy composite signal.
- [ ] Plot prediction vs target examples.
- [ ] Plot loss curves.
- [ ] Plot MSE comparison across models and noise levels.

## Phase 9 — Final Submission

- [ ] Update README with final explanation and results.
- [ ] Add generated plots required for the report.
- [ ] Verify `uv run pytest`.
- [ ] Verify `uv run ruff check .`.
- [ ] Push final commits to GitHub.
- [ ] Submit the required PDF form with the GitHub repository link.
- 