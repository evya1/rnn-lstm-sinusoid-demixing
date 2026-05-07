"""Integration smoke test: run_single and run_noise_sweep with tiny config."""

import numpy as np

from rnn_lstm_sinusoid_demixing.experiments.runner import (
    ModelResult,
    run_noise_sweep,
    run_single,
)
from rnn_lstm_sinusoid_demixing.models.factory import VALID_MODEL_TYPES
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig

_SC = SignalConfig(sampling_rate=100, duration_seconds=1.0)
_TC = TrainingConfig(batch_size=16, num_epochs=2, learning_rate=0.01)


class TestRunSingle:
    results = run_single(_SC, _TC)

    def test_returns_all_model_types(self) -> None:
        assert set(self.results.keys()) == set(VALID_MODEL_TYPES)

    def test_each_result_is_model_result(self) -> None:
        for res in self.results.values():
            assert isinstance(res, ModelResult)

    def test_history_has_correct_length(self) -> None:
        for res in self.results.values():
            assert len(res.history["train"]) == _TC.num_epochs
            assert len(res.history["val"]) == _TC.num_epochs

    def test_test_mse_is_finite_positive(self) -> None:
        for res in self.results.values():
            assert res.test_mse > 0
            assert np.isfinite(res.test_mse)

    def test_sample_shapes(self) -> None:
        for res in self.results.values():
            assert res.sample_prediction.shape == (_SC.context_window,)
            assert res.sample_target.shape == (_SC.context_window,)


class TestRunNoiseSweep:
    noise_levels = [0.0, 0.05, 0.10]
    mse_by_model = run_noise_sweep(_TC, noise_levels, _SC)

    def test_returns_all_model_types(self) -> None:
        assert set(self.mse_by_model.keys()) == set(VALID_MODEL_TYPES)

    def test_correct_number_of_values(self) -> None:
        for mse_list in self.mse_by_model.values():
            assert len(mse_list) == len(self.noise_levels)

    def test_all_mse_finite_positive(self) -> None:
        for model_type, mse_list in self.mse_by_model.items():
            for mse in mse_list:
                assert mse > 0 and np.isfinite(mse), f"{model_type}: MSE={mse}"
