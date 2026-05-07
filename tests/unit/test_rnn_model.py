"""Forward-pass tests for RNNModel."""

import torch

from rnn_lstm_sinusoid_demixing.models.rnn_model import RNNModel

_W = 10              # seq_len / window_size
_C = 4               # num_components
_INPUT_SIZE = 1 + _C # features per timestep: [sigma_t, c_1..c_4]
_H = 64
_B = 8


class TestRNNModel:
    _model = RNNModel(input_size=_INPUT_SIZE, hidden_size=_H)

    def test_instantiates(self) -> None:
        assert self._model is not None

    def test_repr_contains_class_name(self) -> None:
        assert "RNNModel" in repr(self._model)

    def test_repr_contains_hyperparams(self) -> None:
        r = repr(self._model)
        assert str(_INPUT_SIZE) in r and str(_H) in r

    def test_output_shape(self) -> None:
        x = torch.randn(_B, _W, _INPUT_SIZE)
        out = self._model(x)
        assert out.shape == (_B, _W)

    def test_output_shape_batch_size_one(self) -> None:
        x = torch.randn(1, _W, _INPUT_SIZE)
        assert self._model(x).shape == (1, _W)

    def test_output_is_finite(self) -> None:
        x = torch.randn(_B, _W, _INPUT_SIZE)
        assert torch.all(torch.isfinite(self._model(x)))

    def test_stored_hyperparams(self) -> None:
        assert self._model.input_size == _INPUT_SIZE
        assert self._model.hidden_size == _H
        assert self._model.num_layers == 1

    def test_custom_hidden_size(self) -> None:
        m = RNNModel(input_size=_INPUT_SIZE, hidden_size=32)
        assert m(torch.randn(_B, _W, _INPUT_SIZE)).shape == (_B, _W)
