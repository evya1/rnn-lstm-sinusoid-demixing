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

## Prompt 5 — Branch and Pull Request Workflow

Purpose: Decide whether to continue development directly on `main` or use feature branches and pull requests for each major phase.

Summary:
Asked whether it is better to create a dedicated branch and pull request for each major project phase.

Outcome:
The project will continue with short-lived phase branches and pull requests. Each major phase will be implemented in a dedicated branch, pushed to GitHub, reviewed through a pull request, and merged into `main`.
