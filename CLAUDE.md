# CLAUDE.md

This file provides operating instructions for Claude Code when working in this repository.

This repository is for Exercise 01 in the course **AI Agent Orchestration / Agentic AI Systems**.
The project must demonstrate professional software development through Vibe Coding, not only produce working code.

## 1. Non-Negotiable Course Requirements

Claude must preserve the following requirements throughout the project:

- The repository must contain a root `README.md`.
- The repository must contain `docs/PRD.md`, `docs/PLAN.md`, and `docs/TODO.md`.
- The repository should contain `docs/PROMPTS.md` to document the AI prompts and agent workflow used during development.
- The submission must include source code, prompts, diagrams/plots, documentation, and README content.
- The work must be suitable for GitHub submission and review by AI-based graders.
- Aim for at least 85% test coverage for core source code where practical.
- The project must follow the professional software guidelines:
  - modular project structure;
  - Python package organization under `src/`;
  - `uv` for dependency and environment management;
  - `pyproject.toml` and `uv.lock`;
  - tests under `tests/`;
  - configuration under `config/`;
  - generated results under `results/` and/or `assets/`;
  - no secrets or API keys in source code;
  - no hardcoded experimental parameters inside core logic when they belong in config;
  - clean linting with Ruff;
  - meaningful tests;
  - documented results and visualizations.

## 2. Development Lifecycle

Use the Vibe Coding lifecycle:

```text
Idea -> PRD -> PLAN -> TODO -> Verify -> Execute -> Test -> Document -> Push
```

Before implementing new functionality:

1. Check whether the requirement already appears in `docs/PRD.md`.
2. Check whether the architecture/design appears in `docs/PLAN.md`.
3. Check whether the implementation task appears in `docs/TODO.md`.
4. If a central mechanism is added, create or update a dedicated PRD, for example:

   * `docs/PRD_signal_generation.md`
   * `docs/PRD_dataset_builder.md`
   * `docs/PRD_model_comparison.md`
   * `docs/PRD_experiments.md`
5. Only then implement.

Do not jump directly into coding if the requirement is unclear.
First update the relevant documentation and ask for confirmation when the ambiguity affects the assignment semantics.

## 3. Commands

This project uses `uv`. Run commands through `uv run` unless the command is specifically a `uv` command.

```bash
# Install dependencies
uv sync --group dev

# Run all tests
uv run pytest

# Run unit tests
uv run pytest tests/unit

# Run integration tests
uv run pytest tests/integration

# Run a single test file
uv run pytest tests/unit/test_signal_generator.py

# Run linting
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Run the CLI entry point
uv run rnn-lstm-sinusoid-demixing

# Optional coverage command, if configured
uv run pytest --cov=src --cov-report=term-missing
```

Never introduce a second package manager such as `pipenv`, `poetry`, or raw `pip install` workflows unless the user explicitly decides to change the project standard.

## 4. Project Purpose

This is an ML/software-engineering project comparing **Fully Connected**, **RNN**, and **LSTM** models on a **sinusoid demixing** task.

The system generates four clean sinusoidal signals, creates noisy versions of them, sums the noisy signals into one composite signal, and trains models to reconstruct a selected clean component from a context window and a one-hot selector.

The task is not merely to achieve low loss. The project must demonstrate:

* correct synthetic-data generation;
* clean dataset construction;
* fair comparison between model families;
* documented experiments;
* visualized results;
* maintainable software architecture;
* professional GitHub submission quality.

## 5. Exercise 01 Semantics

The core mathematical object is a clean sinusoid:

```markdown
$S_i(t)=A_i\sin(2\pi f_i t+\phi_i)$
```

The baseline frequency configuration is:

```python
frequencies = [1, 3, 5, 7]
```

This baseline is natural for the assignment, but the code should allow alternative frequency configurations for experiments.

Important semantic rule:

> Noise is added to each sinusoid before summation.

That is, generate:

```markdown
$S_{1,\text{noisy}}, S_{2,\text{noisy}}, S_{3,\text{noisy}}, S_{4,\text{noisy}}$
```

and then:

```markdown
$\Sigma_{\text{noisy}} =
S_{1,\text{noisy}}+
S_{2,\text{noisy}}+
S_{3,\text{noisy}}+
S_{4,\text{noisy}}$
```

Do not implement the main version as “sum clean signals first, then add noise once” unless it is explicitly documented as an additional ablation experiment.

## 6. Dataset Design

Each training example contains:

* a fixed-length context window from the noisy composite signal;
* a one-hot selector vector `C` of length 4;
* a target clean window from the selected sinusoidal component.

Conceptually:

```markdown
Input:  $(\Sigma_{\text{noisy window}}, C)$
Target: $S_{j,\text{clean window}}$
```

where `j` is chosen by `C`.

Default assumptions:

```python
duration_seconds = 10
sampling_rate = 1000
num_samples = 10000
context_window = 10
num_components = 4
```

The full signal length is 10 seconds.
The model context window is 10 samples.
Do not confuse these two quantities.

For the Fully Connected model, the natural input is:

```text
window length 10 + selector length 4 = 14 input features
```

For RNN/LSTM, the natural representation is:

```text
(batch_size, sequence_length=10, features_per_timestep=5)
```

where each timestep contains:

```text
[sigma_t, c_1, c_2, c_3, c_4]
```

If a different selector-injection strategy is used, document it in `docs/PLAN.md` and `README.md`.

## 7. Planned Source Layout

Expected structure:

```text
src/rnn_lstm_sinusoid_demixing/
├── __init__.py
├── constants.py
├── main.py
├── sdk/
│   ├── __init__.py
│   └── sdk.py
├── data/
│   ├── signal_generator.py
│   ├── noise.py
│   └── dataset_builder.py
├── models/
│   ├── fully_connected.py
│   ├── rnn_model.py
│   └── lstm_model.py
├── training/
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   └── compare.py
├── visualization/
│   └── plots.py
└── shared/
    ├── config.py
    └── paths.py
```

Tests should live under:

```text
tests/
├── unit/
└── integration/
```

Documentation should live under:

```text
docs/
├── PRD.md
├── PLAN.md
├── TODO.md
├── PROMPTS.md
├── PRD_signal_generation.md
├── PRD_dataset_builder.md
├── PRD_model_comparison.md
└── PRD_experiments.md
```

## 8. Code Quality Rules

Follow these rules strictly:

* Keep every Python source file under 150 lines. This is a hard course requirement; split files by responsibility instead of exceeding it.
* If a file grows too large, split it by responsibility.
* Prefer small, typed, testable functions/classes.
* Use descriptive names.
* Avoid duplicated logic.
* Avoid hidden global state.
* Avoid hardcoded experiment parameters inside model/data logic.
* Keep config values in `config/*.json`, dataclasses, or typed config modules.
* Use `pathlib.Path` instead of fragile string paths.
* Use deterministic seeds where reproducibility matters.
* Do not commit large generated artifacts, checkpoints, caches, or temporary files.
* Do not add secrets, tokens, API keys, or personal credentials.

## 9. Testing Expectations

Use test-driven or test-near development.

At minimum, tests should verify:

* generated sinusoids have the expected shape;
* generated signals contain no `NaN` or infinite values;
* noisy signals preserve expected shape;
* the composite signal is the sum of noisy components;
* one-hot selectors are valid;
* dataset windows have correct input and target shapes;
* the target window corresponds to the selected clean component;
* FC/RNN/LSTM forward passes return the expected output shape;
* a short smoke-training run completes without crashing;
* evaluation metrics return valid numeric values.

Before considering a phase complete, run:

```bash
uv run pytest
uv run ruff check .
```

## 10. Experiment and Results Requirements

The README must not be only an installation document.
It must also summarize the scientific/engineering results.

Generate and document:

* clean component signals;
* noisy component signals;
* noisy composite signal;
* example input context window;
* example target clean window;
* prediction vs target for FC;
* prediction vs target for RNN;
* prediction vs target for LSTM;
* training and validation loss curves;
* test MSE table;
* MSE vs noise level plot;
* conclusions about model behavior.

Recommended baseline:

```python
frequencies = [1, 3, 5, 7]
noise_levels = [0.00, 0.01, 0.05, 0.10, 0.20]
context_window = 10
models = ["fully_connected", "rnn", "lstm"]
```

Recommended additional frequency scenarios:

```python
frequency_scenarios = {
    "baseline": [1, 3, 5, 7],
    "low_mixed": [0.5, 1, 3, 7],
    "wide_gap": [1, 5, 20, 40],
    "close_low": [1, 2, 3, 4],
}
```

Every experiment should have:

* config;
* command used to run it;
* saved metrics;
* saved plots;
* short interpretation.

## 11. README Expectations

The root `README.md` should include:

1. project title and short description;
2. assignment/course context;
3. problem statement;
4. installation instructions;
5. usage instructions;
6. configuration explanation;
7. data-generation explanation;
8. model architecture summary;
9. experiment protocol;
10. results with plots/tables;
11. conclusions;
12. limitations;
13. future improvements;
14. testing/linting commands;
15. AI usage disclosure;
16. credits/license if relevant.

## 12. Prompt Logging

Whenever Claude performs a meaningful development step, update `docs/PROMPTS.md`.

For each prompt/session, document:

```markdown
## Prompt N — Short Title

**Date:** YYYY-MM-DD  
**Tool/Agent:** Claude Code / ChatGPT / Gemini / NotebookLM  
**Purpose:** What this prompt was meant to achieve.  
**Prompt Summary:** Short paraphrase of the prompt.  
**Files Affected:** List of files changed or created.  
**Outcome:** What was actually produced.  
**Human Review:** What the student checked, changed, accepted, or rejected.
```

Do not fabricate prompt logs.
If the exact prompt is unavailable, write a faithful summary and mark it as a summary.

## 13. Git Workflow

Use short-lived phase branches:

```text
phase-01/project-setup
phase-02/docs-prd-plan-todo
phase-03/project-skeleton
phase-04/data-generation
phase-05/dataset-builder
phase-06/models
phase-07/training-evaluation
phase-08/visualization-results
phase-09/final-submission
```

Each phase should have:

* focused commits;
* tests where relevant;
* documentation updates;
* TODO status updates;
* a clear merge into `main`.

Avoid one giant final commit. The course explicitly values a visible development history.

## 14. Before Making Changes

Before editing code, Claude should inspect:

* `docs/PRD.md`
* `docs/PLAN.md`
* `docs/TODO.md`
* relevant source files
* relevant tests

Then Claude should state a short plan and implement only the requested phase or task.

## 15. Before Finishing a Task

Before reporting completion, Claude must check:

* Did the implementation match the PRD?
* Did the implementation match the PLAN?
* Was TODO updated?
* Were tests added or updated?
* Were commands run?
* Were results/plots documented if relevant?
* Was `docs/PROMPTS.md` updated if AI assistance was used?
* Did the change avoid generated artifacts and secrets?

## 16. Do Not Do These Things

Do not:

* replace the project with a notebook-only solution;
* hide core logic inside notebooks;
* commit model checkpoints by default;
* commit large generated files unless intentionally part of the report;
* hardcode absolute local paths;
* hardcode secrets;
* skip documentation;
* skip tests;
* silently change assignment semantics;
* implement noise after summation as the main pipeline;
* compare models using different datasets/splits unless documented as a deliberate experiment;
* report success without checking tests/linting when possible.

## 17. Preferred Implementation Style

Use:

* Pythonic, readable code;
* type hints;
* dataclasses for config where useful;
* `numpy` for signal generation;
* `torch` for FC/RNN/LSTM models;
* `matplotlib` for plots;
* `pytest` for tests;
* `ruff` for linting;
* `typer` or a simple CLI entrypoint for commands.

The code should be understandable by another student, by the professor, and by an automated AI reviewer.

## 18. Python Code Style and Project-Relevant Fluent Python Practices

Follow idiomatic, readable Python. Prefer simple, explicit, maintainable code over clever abstractions.

### Hard File-Size Rule

- Each `.py` file must stay under 150 lines of code.
- If a file approaches the limit, split it by responsibility before adding more logic.
- Do not bypass the limit by writing dense, unreadable code.
- Prefer more small modules over one large module.

### Readability First

- Code should be understandable by another student, the professor, and an automated AI reviewer.
- Prefer clear names over short names.
- Use small functions with one clear responsibility.
- Avoid “clever” Python features unless they clearly improve readability.
- Do not use metaprogramming, descriptors, custom protocols, operator overloading, or advanced dunder methods unless there is a direct, documented need.

### Type Hints

- Add type hints to public functions, config objects, dataset builders, model factories, and evaluation functions.
- Type hints improve documentation and static checking, but they do not replace tests.
- Use `numpy.typing.NDArray` where it clarifies numerical array inputs/outputs.
- Avoid overcomplicated type annotations that make the code harder to read.

### Dataclasses and Configuration

- Use `@dataclass(frozen=True)` for simple immutable configuration objects.
- Good use cases:
  - experiment configuration;
  - signal-generation parameters;
  - training parameters;
  - evaluation settings.
- Do not create “anemic” data classes everywhere. If behavior naturally belongs with the data, add methods or keep the logic in a clear service/function.
- Keep runtime experiment values in config files or config objects, not scattered as hardcoded constants.

### Mutability and Defaults

- Never use mutable default arguments such as `[]`, `{}`, or `set()` in function signatures.
- Use `None` as the default and create the object inside the function, or use `default_factory` in dataclasses.
- Avoid unexpected mutation of input arguments unless the function name and docstring clearly state that mutation is intended.
- Prefer returning new arrays/results from data-processing functions.

### Comprehensions and Generators

- Use list comprehensions when the goal is clearly to build a list.
- Do not use comprehensions only for side effects.
- Keep comprehensions short; if a comprehension becomes hard to read or spans several logical steps, rewrite it as a normal loop.
- Use generator expressions when streaming values or when building an intermediate list would waste memory.

### Numerical Python

- Prefer NumPy vectorized operations for signal generation, noise generation, metrics, and array transformations.
- Avoid explicit Python loops over samples when a clear NumPy expression is available.
- Keep PyTorch tensor logic inside model/training modules and NumPy logic inside data/evaluation/visualization modules unless conversion is necessary and documented.
- Be explicit about shapes in variable names, comments, or docstrings when useful.

### Functions, Classes, and Modules

- Use functions for stateless transformations such as generating signals, creating windows, computing MSE, and saving plots.
- Use classes when they represent real concepts with state or behavior, such as models, trainers, or SDK facades.
- Keep model definitions separate from training loops.
- Keep plotting separate from metric computation.
- Keep CLI orchestration separate from core logic.

### Docstrings and Comments

- Add concise docstrings to public functions/classes.
- A docstring should explain purpose, important arguments, return value, and shape assumptions when relevant.
- Do not comment obvious code.
- Use comments to explain non-obvious design decisions, mathematical assumptions, or assignment-specific semantics.

### Testing Discipline

- Tests must verify behavior, not implementation details.
- Prefer small unit tests for pure functions.
- Add integration/smoke tests for training and end-to-end flow.
- Type hints, Ruff, and clean architecture are not substitutes for tests.
- Aim for at least 85% coverage for core source code where practical.

### Error Handling and Validation

- Validate important input assumptions early:
  - positive sampling rate;
  - positive duration;
  - context window smaller than signal length;
  - exactly four components unless the config explicitly supports another number;
  - valid one-hot selector;
  - matching signal lengths.
- Raise clear `ValueError` messages for invalid user/config input.
- Do not silently continue after shape mismatches.

### Reproducibility

- Use deterministic seeds for experiments when comparing models.
- Save experiment configuration together with metrics and plots.
- Make every reported result reproducible from a documented command.

### Text Files, JSON, and Paths

- Use `pathlib.Path` for filesystem paths.
- When reading or writing text files, always pass an explicit encoding, preferably `encoding="utf-8"`.
- Use `Path.read_text(encoding="utf-8")` and `Path.write_text(..., encoding="utf-8")` for small text files.
- Use JSON/YAML/config files only for configuration and experiment metadata, not for hidden logic.
- Do not rely on platform default encodings.

### Object Design

- Prefer composition over inheritance unless inheritance is required by a framework.
- Deep inheritance hierarchies are not appropriate for this project.
- It is acceptable for PyTorch models to inherit from `torch.nn.Module`.
- Keep SDK/facade classes thin; they should orchestrate existing modules, not hide large amounts of logic.
- For custom classes that are not dataclasses, implement `__repr__` when it helps debugging or logging.
- Do not implement special methods merely to look “Pythonic”; use them only when they make the object behave naturally with Python syntax.

## 19. Commit Discipline

Every commit on a phase branch must follow these rules:

- **One logical change per commit.** Do not bundle unrelated changes (e.g., a new module + its tests + a docs update) into one commit.
- **Size constraint:** Aim for ≤ 100 lines changed and ≤ 2 files per commit where possible. If a single logical change unavoidably touches more, that is acceptable, but it must never be used as a shortcut to batch unrelated work.
- **Suggested sequence within a phase:**
  1. PRD / sub-PRD documentation update — docs files only.
  2. Source implementation — one module at a time, one commit per file where practical.
  3. Tests for that module — one test file per commit.
  4. TODO + PROMPTS update — docs files only, as the final commit.
- **Never squash all phase work into one commit.** The course grades on a visible, incremental development history. A clean `git log` is part of the submission.
- Commit messages must use the imperative mood, briefly state *what* changed and *why* (if non-obvious), and end with the co-author trailer.

## 20. Pull Request Standards

PR bodies must be professional, specific, and traceable. Every PR must include:

### Required sections

1. **Overview** — one paragraph summarising the phase goal and scope boundaries (what is *not* included).
2. **TODOs resolved** — quote the exact checkbox lines from `docs/TODO.md` that this PR closes, rendered as a checked list.
3. **Plan / PRD reference** — cite the governing section of `docs/PLAN.md` or the relevant sub-PRD (e.g., `docs/PRD_model_comparison.md §Models`).
4. **Commit log** — list each commit by short hash and one-line description, e.g.:
   ```
   abc1234  Add __repr__ to FullyConnectedModel
   def5678  Implement forward-pass tests for FC model
   ```
5. **Files changed** — for every file, state what changed and the key design decision behind it.
6. **Design decisions** — explain non-obvious implementation choices: why a particular API shape, data layout, or algorithm was chosen.
7. **Validation** — show the exact commands run and their results:
   ```
   uv run pytest tests/unit -v  →  N passed
   uv run ruff check .          →  All checks passed
   ```

The PR title must follow the pattern `Phase NN: <short noun phrase>`.
Do not write vague PR bodies (“added models”, “fixed stuff”). Cite filenames, line counts, commit hashes, and section references.
