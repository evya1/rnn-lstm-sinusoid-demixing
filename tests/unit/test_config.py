"""Unit tests for SignalConfig and TrainingConfig dataclasses."""

import pytest

from rnn_lstm_sinusoid_demixing import constants
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig


class TestSignalConfig:
    def test_defaults_match_constants(self) -> None:
        cfg = SignalConfig()
        assert cfg.frequencies == tuple(constants.DEFAULT_FREQUENCIES)
        assert cfg.sampling_rate == constants.DEFAULT_SAMPLING_RATE
        assert cfg.duration_seconds == constants.DEFAULT_DURATION_SECONDS
        assert cfg.num_components == constants.DEFAULT_NUM_COMPONENTS
        assert cfg.noise_level == constants.DEFAULT_NOISE_LEVEL
        assert cfg.context_window == constants.DEFAULT_CONTEXT_WINDOW
        assert cfg.random_seed == constants.DEFAULT_RANDOM_SEED

    def test_custom_values_accepted(self) -> None:
        cfg = SignalConfig(
            frequencies=(2.0, 4.0, 6.0, 8.0),
            sampling_rate=500,
            duration_seconds=5.0,
        )
        assert cfg.frequencies == (2.0, 4.0, 6.0, 8.0)
        assert cfg.sampling_rate == 500

    def test_invalid_sampling_rate(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate"):
            SignalConfig(sampling_rate=0)

    def test_invalid_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds"):
            SignalConfig(duration_seconds=-1.0)

    def test_frequency_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="num_components"):
            SignalConfig(frequencies=(1.0, 2.0), num_components=4)

    def test_negative_noise_level(self) -> None:
        with pytest.raises(ValueError, match="noise_level"):
            SignalConfig(noise_level=-0.1)

    def test_context_window_too_large(self) -> None:
        with pytest.raises(ValueError, match="context_window"):
            SignalConfig(
                sampling_rate=10,
                duration_seconds=1.0,
                context_window=100,
            )

    def test_immutable(self) -> None:
        cfg = SignalConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.sampling_rate = 999  # type: ignore[misc]


class TestTrainingConfig:
    def test_defaults_match_constants(self) -> None:
        cfg = TrainingConfig()
        assert cfg.batch_size == constants.DEFAULT_BATCH_SIZE
        assert cfg.learning_rate == constants.DEFAULT_LEARNING_RATE
        assert cfg.num_epochs == constants.DEFAULT_NUM_EPOCHS

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_invalid_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0.0)

    def test_invalid_split_sum(self) -> None:
        with pytest.raises(ValueError, match="validation_split"):
            TrainingConfig(validation_split=0.6, test_split=0.6)

    def test_immutable(self) -> None:
        cfg = TrainingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.batch_size = 999  # type: ignore[misc]
