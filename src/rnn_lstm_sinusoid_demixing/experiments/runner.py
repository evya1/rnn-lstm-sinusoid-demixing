"""Experiment orchestration: train all three models and collect results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from rnn_lstm_sinusoid_demixing.data.dataloader import make_loader, split_dataset
from rnn_lstm_sinusoid_demixing.data.dataset_builder import build_dataset
from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
from rnn_lstm_sinusoid_demixing.evaluation.metrics import compute_mse
from rnn_lstm_sinusoid_demixing.models.factory import VALID_MODEL_TYPES, create_model
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig
from rnn_lstm_sinusoid_demixing.training.losses import mse_loss
from rnn_lstm_sinusoid_demixing.training.trainer import Trainer


@dataclass
class ModelResult:
    """Results for a single model on a single noise level."""

    history: dict[str, list[float]]
    test_mse: float
    sample_prediction: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_target: np.ndarray = field(default_factory=lambda: np.array([]))


def run_single(
    signal_config: SignalConfig,
    training_config: TrainingConfig,
) -> dict[str, ModelResult]:
    """Train all three models on one signal configuration.

    All models share the same dataset split and random seed for fair comparison
    (see PRD_experiments.md §Fairness Rules).

    Args:
        signal_config:   Defines signal parameters including noise_level.
        training_config: Defines batch_size, lr, num_epochs, random_seed.

    Returns:
        Dict mapping model_type -> ModelResult with history, test_mse,
        and a sample (prediction, target) window for plotting.
    """
    _, clean, _, composite = build_signals(signal_config)
    inputs, selectors, targets = build_dataset(
        composite, clean, signal_config.context_window, signal_config.num_components
    )
    splits = split_dataset(inputs, selectors, targets, random_seed=training_config.random_seed)

    results: dict[str, ModelResult] = {}
    for model_type in VALID_MODEL_TYPES:
        torch.manual_seed(training_config.random_seed)
        model = create_model(model_type, signal_config, training_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
        trainer = Trainer(model, optimizer, mse_loss())

        train_loader = make_loader(*splits["train"], model_type, training_config.batch_size)
        val_loader = make_loader(*splits["val"], model_type, training_config.batch_size, shuffle=False)
        history = trainer.fit(train_loader, val_loader, training_config.num_epochs)

        test_loader = make_loader(*splits["test"], model_type, training_config.batch_size, shuffle=False)
        preds, tgts = [], []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                preds.append(model(x).numpy())
                tgts.append(y.numpy())
        all_preds = np.concatenate(preds)
        all_tgts = np.concatenate(tgts)

        results[model_type] = ModelResult(
            history=history,
            test_mse=compute_mse(all_preds, all_tgts),
            sample_prediction=all_preds[0],
            sample_target=all_tgts[0],
        )
    return results


def run_noise_sweep(
    training_config: TrainingConfig,
    noise_levels: list[float],
    base_config: SignalConfig,
) -> dict[str, list[float]]:
    """Train all models across a range of noise levels.

    Args:
        training_config: Shared training parameters.
        noise_levels:    List of noise standard deviations to sweep over.
        base_config:     Template SignalConfig; noise_level is overridden per step.

    Returns:
        Dict mapping model_type -> list of test MSE values (one per noise level).
    """
    mse_by_model: dict[str, list[float]] = {mt: [] for mt in VALID_MODEL_TYPES}
    for noise_level in noise_levels:
        import dataclasses
        sc = dataclasses.replace(base_config, noise_level=noise_level)
        single = run_single(sc, training_config)
        for model_type in VALID_MODEL_TYPES:
            mse_by_model[model_type].append(single[model_type].test_mse)
    return mse_by_model
