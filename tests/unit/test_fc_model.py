"""Forward-pass tests for FullyConnectedModel."""

import torch

from rnn_lstm_sinusoid_demixing.models.fully_connected import FullyConnectedModel

_W = 10   # window_size
_C = 4    # num_components
_H = 64   # hidden_size
_B = 8    # batch size


class TestFullyConnectedModel:
    _model = FullyConnectedModel(window_size=_W, num_components=_C, hidden_size=_H)

    def test_instantiates(self) -> None:
        assert self._model is not None

    def test_repr_contains_class_name(self) -> None:
        assert "FullyConnectedModel" in repr(self._model)

    def test_repr_contains_hyperparams(self) -> None:
        r = repr(self._model)
        assert str(_W) in r and str(_C) in r and str(_H) in r

    def test_output_shape(self) -> None:
        x = torch.randn(_B, _W + _C)
        out = self._model(x)
        assert out.shape == (_B, _W)

    def test_output_shape_batch_size_one(self) -> None:
        x = torch.randn(1, _W + _C)
        assert self._model(x).shape == (1, _W)

    def test_output_is_finite(self) -> None:
        x = torch.randn(_B, _W + _C)
        assert torch.all(torch.isfinite(self._model(x)))

    def test_stored_hyperparams(self) -> None:
        assert self._model.window_size == _W
        assert self._model.num_components == _C
        assert self._model.hidden_size == _H

    def test_custom_hidden_size(self) -> None:
        m = FullyConnectedModel(window_size=_W, num_components=_C, hidden_size=32)
        out = m(torch.randn(_B, _W + _C))
        assert out.shape == (_B, _W)
