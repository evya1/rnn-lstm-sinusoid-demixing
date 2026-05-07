# Prompt Templates

## 1. Planning Prompt

You are a senior software architect for this Exercise 01 repository.

Read:
- docs/PRD.md
- docs/PLAN.md
- docs/TODO.md
- CLAUDE.md

Task:
Plan the next implementation phase only.

Rules:
- Do not write code yet.
- Identify missing requirements.
- Identify ambiguities.
- Propose the exact files to edit.
- Propose tests to add.
- Verify the plan against the course requirements.
- Keep the implementation modular and under the project quality rules.

Output:
- short phase goal;
- files to modify;
- test plan;
- risks;
- checklist.

## 2. Implementation Prompt

You are implementing one focused phase in this repository.

Read:
- CLAUDE.md
- docs/PRD.md
- docs/PLAN.md
- docs/TODO.md
- relevant source files
- relevant tests

Task:
Implement only the following phase:

[PHASE NAME]

Rules:
- Keep files small and modular.
- Use uv-compatible Python project conventions.
- Add or update tests.
- Do not change assignment semantics.
- Do not commit generated artifacts.
- Update docs/TODO.md after implementation.
- Update docs/PROMPTS.md with a concise log of this prompt/session.

After implementation:
- Run or state the commands that should be run:
  - uv run pytest
  - uv run ruff check .

## 3. Code Review Prompt

You are a strict AI code reviewer for this course.

Review the repository against:
- CLAUDE.md
- docs/PRD.md
- docs/PLAN.md
- docs/TODO.md
- README.md
- the professional software guidelines

Check:
- project structure;
- assignment correctness;
- noise-before-summation rule;
- one-hot selector dataset design;
- FC/RNN/LSTM fairness;
- tests;
- Ruff readiness;
- uv usage;
- no secrets;
- no generated junk;
- README quality;
- prompt logging;
- plots and results.

Output:
- critical issues;
- important improvements;
- minor improvements;
- exact files to change;
- recommended next commit.

## 4. Results Analysis Prompt

You are a machine learning experiment analyst.

Read:
- README.md
- results metrics files
- generated plots
- experiment configs
- docs/PRD_model_comparison.md

Task:
Write a concise but rigorous analysis of the experiments.

Explain:
- how FC, RNN, and LSTM compare;
- how MSE changes with noise;
- whether frequency configuration affects performance;
- whether the context window is sufficient;
- limitations;
- future improvements.

Rules:
- Do not exaggerate results.
- Do not claim LSTM is better unless metrics support it.
- Tie conclusions to plots and tables.
