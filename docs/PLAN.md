# Implementation Plan — RNN-LSTM Sinusoid Demixing

## 1. Development Strategy

The project will be developed in small, reviewable steps:

1. Repository and uv setup.
2. Documentation setup.
3. Project module skeleton.
4. Signal generation.
5. Dataset construction.
6. Model implementations.
7. Training loop.
8. Evaluation and visualization.
9. README update with results and analysis.

## 2. Planned Project Structure

```text
rnn-lstm-sinusoid-demixing/
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   └── PROMPTS.md
├── src/
│   └── rnn_lstm_sinusoid_demixing/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── visualization/
└── tests/
    ├── unit/
    └── integration/
```

## 3. Data Pipeline

The data pipeline will include:

1. Generate clean sinusoids.
2. Add noise to each sinusoidal component.
3. Sum noisy components into one composite signal.
4. Sample fixed-size windows from the composite signal.
5. Pair each input window with a one-hot selector.
6. Use the corresponding clean sinusoid window as the target.

## 4. Model Plan

### Fully Connected Baseline

A feed-forward model that receives a flattened input window concatenated with the one-hot selector.

### RNN Model

A recurrent model that receives the signal window as a sequence. The selector will be provided as part of the input features.

### LSTM Model

An LSTM model using the same input representation as the RNN, allowing comparison between simple recurrent memory and gated recurrent memory.

## 5. Evaluation Plan

The main metric will be mean squared error.

The project will include:

* Training loss curves.
* Validation/test MSE.
* Prediction vs target plots.
* Comparison across noise levels.
* Comparison across model types.

## 6. Risks and Mitigations

* If training is unstable, start with a small dataset and simple models.
* If LSTM/RNN models are too slow, reduce hidden size and number of epochs.
* If results are unclear, add controlled experiments with fixed random seeds.
* If plots are too noisy, visualize short windows rather than full signals.


## 7. Branch and Pull Request Workflow

Development will continue using short-lived feature branches and pull requests.

Each major project phase will be implemented in a dedicated branch. After the phase is complete, the branch will be pushed to GitHub and merged into `main` through a pull request.

Planned branch structure:

```text
phase-03/project-skeleton
phase-04/data-generation
phase-05/dataset-builder
phase-06/models
phase-07/training-evaluation
phase-08/visualization-results
phase-09/final-submission
```
