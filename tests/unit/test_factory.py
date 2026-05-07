"""Unit tests for the model factory."""

import pytest
import torch

from rnn_lstm_sinusoid_demixing.models.factory import VALID_MODEL_TYPES, create_model
from rnn_lstm_sinusoid_demixing.models.fully_connected import FullyConnectedModel
from rnn_lstm_sinusoid_demixing.models.lstm_model import LSTMModel
from rnn_lstm_sinusoid_demixing.models.rnn_model import RNNModel
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig

_SC = SignalConfig(sampling_rate=100, duration_seconds=1.0)
_TC = TrainingConfig()
_W = _SC.context_window   # 10
_C = _SC.num_components   # 4
_B = 4


class TestCreateModel:
    def test_creates_fc_model(self) -> None:
        assert isinstance(create_model("fc", _SC, _TC), FullyConnectedModel)

    def test_creates_rnn_model(self) -> None:
        assert isinstance(create_model("rnn", _SC, _TC), RNNModel)

    def test_creates_lstm_model(self) -> None:
        assert isinstance(create_model("lstm", _SC, _TC), LSTMModel)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model("transformer", _SC, _TC)

    def test_valid_model_types_constant(self) -> None:
        assert set(VALID_MODEL_TYPES) == {"fc", "rnn", "lstm"}

    def test_fc_forward_pass(self) -> None:
        model = create_model("fc", _SC, _TC)
        x = torch.randn(_B, _W + _C)
        assert model(x).shape == (_B, _W)

    def test_rnn_forward_pass(self) -> None:
        model = create_model("rnn", _SC, _TC)
        x = torch.randn(_B, _W, 1 + _C)
        assert model(x).shape == (_B, _W)

    def test_lstm_forward_pass(self) -> None:
        model = create_model("lstm", _SC, _TC)
        x = torch.randn(_B, _W, 1 + _C)
        assert model(x).shape == (_B, _W)

    def test_fc_uses_config_hidden_size(self) -> None:
        model = create_model("fc", _SC, _TC)
        assert model.hidden_size == _TC.fc_hidden_size

    def test_rnn_uses_config_hidden_size(self) -> None:
        model = create_model("rnn", _SC, _TC)
        assert model.hidden_size == _TC.rnn_hidden_size
