"""Unit tests for Phase 5 dataset builder."""

import numpy as np
import pytest

from rnn_lstm_sinusoid_demixing.data.dataset_builder import (
    build_dataset,
    extract_windows,
    make_one_hot,
)
from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig

_CFG = SignalConfig(sampling_rate=100, duration_seconds=1.0)
_W = _CFG.context_window  # 10
_N = _CFG.sampling_rate   # 100 samples
_C = _CFG.num_components  # 4
_NUM_WINDOWS = _N - _W + 1  # 91

_, _CLEAN, _NOISY, _COMPOSITE = build_signals(_CFG)


class TestMakeOneHot:
    def test_shape(self) -> None:
        assert make_one_hot(0, 4).shape == (4,)

    def test_dtype(self) -> None:
        assert make_one_hot(0, 4).dtype == np.float32

    def test_correct_index_is_one(self) -> None:
        for i in range(4):
            vec = make_one_hot(i, 4)
            assert vec[i] == 1.0

    def test_all_others_are_zero(self) -> None:
        vec = make_one_hot(2, 4)
        assert vec[0] == 0.0 and vec[1] == 0.0 and vec[3] == 0.0

    def test_sum_is_one(self) -> None:
        assert make_one_hot(1, 4).sum() == pytest.approx(1.0)

    def test_invalid_index_raises(self) -> None:
        with pytest.raises(ValueError):
            make_one_hot(4, 4)

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError):
            make_one_hot(-1, 4)


class TestExtractWindows:
    def test_shape(self) -> None:
        windows = extract_windows(_COMPOSITE, _W)
        assert windows.shape == (_NUM_WINDOWS, _W)

    def test_dtype(self) -> None:
        assert extract_windows(_COMPOSITE, _W).dtype == np.float32

    def test_first_window_matches_signal(self) -> None:
        windows = extract_windows(_COMPOSITE, _W)
        np.testing.assert_array_equal(windows[0], _COMPOSITE[:_W])

    def test_last_window_matches_signal(self) -> None:
        windows = extract_windows(_COMPOSITE, _W)
        np.testing.assert_array_equal(windows[-1], _COMPOSITE[-_W:])

    def test_no_nan_inf(self) -> None:
        assert np.all(np.isfinite(extract_windows(_COMPOSITE, _W)))

    def test_window_size_exceeds_length_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_windows(_COMPOSITE, len(_COMPOSITE) + 1)


class TestBuildDataset:
    inputs, selectors, targets = build_dataset(_COMPOSITE, _CLEAN, _W, _C)
    _num_examples = _NUM_WINDOWS * _C

    def test_inputs_shape(self) -> None:
        assert self.inputs.shape == (self._num_examples, _W)

    def test_selectors_shape(self) -> None:
        assert self.selectors.shape == (self._num_examples, _C)

    def test_targets_shape(self) -> None:
        assert self.targets.shape == (self._num_examples, _W)

    def test_inputs_dtype(self) -> None:
        assert self.inputs.dtype == np.float32

    def test_selectors_dtype(self) -> None:
        assert self.selectors.dtype == np.float32

    def test_targets_dtype(self) -> None:
        assert self.targets.dtype == np.float32

    def test_selectors_are_valid_one_hot(self) -> None:
        sums = self.selectors.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(_NUM_WINDOWS * _C))
        assert np.all((self.selectors == 0) | (self.selectors == 1))

    def test_targets_match_correct_component(self) -> None:
        # For component j, examples live at rows [j*num_windows .. (j+1)*num_windows)
        for j in range(_C):
            start = j * _NUM_WINDOWS
            end = (j + 1) * _NUM_WINDOWS
            expected = extract_windows(_CLEAN[j], _W)
            np.testing.assert_array_almost_equal(self.targets[start:end], expected)

    def test_no_nan_inf_inputs(self) -> None:
        assert np.all(np.isfinite(self.inputs))

    def test_no_nan_inf_targets(self) -> None:
        assert np.all(np.isfinite(self.targets))

    def test_wrong_num_components_raises(self) -> None:
        with pytest.raises(ValueError):
            build_dataset(_COMPOSITE, _CLEAN, _W, num_components=3)
