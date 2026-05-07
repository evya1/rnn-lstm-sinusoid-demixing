"""Integration smoke test: full training loop for all three model types.

Runs 3 epochs on a tiny dataset to verify the Trainer, DataLoader,
and model families work together end-to-end without errors.
"""

import torch

from rnn_lstm_sinusoid_demixing.data.dataloader import make_loader, split_dataset
from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
from rnn_lstm_sinusoid_demixing.data.dataset_builder import build_dataset
from rnn_lstm_sinusoid_demixing.evaluation.compare import compare_models
from rnn_lstm_sinusoid_demixing.evaluation.metrics import compute_mse
from rnn_lstm_sinusoid_demixing.models.factory import VALID_MODEL_TYPES, create_model
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig
from rnn_lstm_sinusoid_demixing.training.losses import mse_loss
from rnn_lstm_sinusoid_demixing.training.trainer import Trainer

_SC = SignalConfig(sampling_rate=100, duration_seconds=1.0)
_TC = TrainingConfig(batch_size=16, num_epochs=3, learning_rate=0.01)

_, clean, _, composite = build_signals(_SC)
inputs, selectors, targets = build_dataset(composite, clean, _SC.context_window, _SC.num_components)
_SPLITS = split_dataset(inputs, selectors, targets, random_seed=_TC.random_seed)


def _make_loaders(model_type: str):
    train_inp, train_sel, train_tgt = _SPLITS["train"]
    val_inp, val_sel, val_tgt = _SPLITS["val"]
    train_loader = make_loader(train_inp, train_sel, train_tgt, model_type, _TC.batch_size)
    val_loader = make_loader(val_inp, val_sel, val_tgt, model_type, _TC.batch_size, shuffle=False)
    return train_loader, val_loader


class TestSmokeTraining:
    def test_fc_fit_completes(self) -> None:
        model = create_model("fc", _SC, _TC)
        optimizer = torch.optim.Adam(model.parameters(), lr=_TC.learning_rate)
        trainer = Trainer(model, optimizer, mse_loss())
        train_loader, val_loader = _make_loaders("fc")
        history = trainer.fit(train_loader, val_loader, num_epochs=_TC.num_epochs)
        assert len(history["train"]) == _TC.num_epochs
        assert len(history["val"]) == _TC.num_epochs

    def test_rnn_fit_completes(self) -> None:
        model = create_model("rnn", _SC, _TC)
        optimizer = torch.optim.Adam(model.parameters(), lr=_TC.learning_rate)
        trainer = Trainer(model, optimizer, mse_loss())
        train_loader, val_loader = _make_loaders("rnn")
        history = trainer.fit(train_loader, val_loader, num_epochs=_TC.num_epochs)
        assert len(history["train"]) == _TC.num_epochs

    def test_lstm_fit_completes(self) -> None:
        model = create_model("lstm", _SC, _TC)
        optimizer = torch.optim.Adam(model.parameters(), lr=_TC.learning_rate)
        trainer = Trainer(model, optimizer, mse_loss())
        train_loader, val_loader = _make_loaders("lstm")
        history = trainer.fit(train_loader, val_loader, num_epochs=_TC.num_epochs)
        assert len(history["train"]) == _TC.num_epochs

    def test_all_losses_are_finite(self) -> None:
        for model_type in VALID_MODEL_TYPES:
            model = create_model(model_type, _SC, _TC)
            optimizer = torch.optim.Adam(model.parameters(), lr=_TC.learning_rate)
            trainer = Trainer(model, optimizer, mse_loss())
            train_loader, val_loader = _make_loaders(model_type)
            history = trainer.fit(train_loader, val_loader, num_epochs=2)
            for loss in history["train"] + history["val"]:
                assert loss > 0 and loss < float("inf"), f"{model_type} loss not finite: {loss}"

    def test_compare_models_with_real_results(self) -> None:
        mse_scores: dict[str, float] = {}
        for model_type in VALID_MODEL_TYPES:
            model = create_model(model_type, _SC, _TC)
            optimizer = torch.optim.Adam(model.parameters(), lr=_TC.learning_rate)
            trainer = Trainer(model, optimizer, mse_loss())
            train_loader, val_loader = _make_loaders(model_type)
            trainer.fit(train_loader, val_loader, num_epochs=2)
            test_inp, test_sel, test_tgt = _SPLITS["test"]
            test_loader = make_loader(test_inp, test_sel, test_tgt, model_type, _TC.batch_size, shuffle=False)
            preds, tgts = [], []
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    preds.append(model(x).numpy())
                    tgts.append(y.numpy())
            import numpy as np
            mse_scores[model_type] = compute_mse(np.concatenate(preds), np.concatenate(tgts))
        ranking = compare_models(mse_scores)
        values = list(ranking.values())
        assert values == sorted(values)
        assert all(v > 0 for v in values)
