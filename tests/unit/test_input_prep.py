"""Unit tests for model input preparation utilities."""

import torch

from rnn_lstm_sinusoid_demixing.models.input_prep import prepare_fc_input, prepare_seq_input

_B = 8   # batch
_W = 10  # window / seq_len
_C = 4   # num_components

_WINDOWS   = torch.randn(_B, _W)
_SELECTORS = torch.zeros(_B, _C)
_SELECTORS[:, 1] = 1.0  # fixed one-hot for component 1


class TestPrepareFcInput:
    def test_output_shape(self) -> None:
        out = prepare_fc_input(_WINDOWS, _SELECTORS)
        assert out.shape == (_B, _W + _C)

    def test_window_values_preserved(self) -> None:
        out = prepare_fc_input(_WINDOWS, _SELECTORS)
        torch.testing.assert_close(out[:, :_W], _WINDOWS)

    def test_selector_values_preserved(self) -> None:
        out = prepare_fc_input(_WINDOWS, _SELECTORS)
        torch.testing.assert_close(out[:, _W:], _SELECTORS)

    def test_output_is_finite(self) -> None:
        assert torch.all(torch.isfinite(prepare_fc_input(_WINDOWS, _SELECTORS)))

    def test_batch_size_one(self) -> None:
        out = prepare_fc_input(_WINDOWS[:1], _SELECTORS[:1])
        assert out.shape == (1, _W + _C)


class TestPrepareSeqInput:
    def test_output_shape(self) -> None:
        out = prepare_seq_input(_WINDOWS, _SELECTORS)
        assert out.shape == (_B, _W, 1 + _C)

    def test_sigma_at_first_feature(self) -> None:
        out = prepare_seq_input(_WINDOWS, _SELECTORS)
        torch.testing.assert_close(out[:, :, 0], _WINDOWS)

    def test_selector_broadcast_across_timesteps(self) -> None:
        out = prepare_seq_input(_WINDOWS, _SELECTORS)
        for t in range(_W):
            torch.testing.assert_close(out[:, t, 1:], _SELECTORS)

    def test_output_is_finite(self) -> None:
        assert torch.all(torch.isfinite(prepare_seq_input(_WINDOWS, _SELECTORS)))

    def test_batch_size_one(self) -> None:
        out = prepare_seq_input(_WINDOWS[:1], _SELECTORS[:1])
        assert out.shape == (1, _W, 1 + _C)
