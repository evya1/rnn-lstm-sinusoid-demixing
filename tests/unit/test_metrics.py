"""Unit tests for evaluation/metrics.py."""

import numpy as np
import pytest

from rnn_lstm_sinusoid_demixing.evaluation.metrics import compute_mse


class TestComputeMSE:
    def test_identical_arrays_gives_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert compute_mse(a, a) == 0.0

    def test_known_value(self) -> None:
        pred = np.array([0.0, 0.0])
        tgt = np.array([1.0, 3.0])
        # MSE = (1 + 9) / 2 = 5.0
        assert compute_mse(pred, tgt) == pytest.approx(5.0)

    def test_returns_float(self) -> None:
        result = compute_mse(np.zeros(4), np.ones(4))
        assert isinstance(result, float)

    def test_result_is_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        assert compute_mse(a, b) >= 0.0

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        assert compute_mse(a, b) == pytest.approx(compute_mse(b, a))

    def test_2d_arrays(self) -> None:
        pred = np.zeros((4, 10))
        tgt = np.ones((4, 10))
        assert compute_mse(pred, tgt) == pytest.approx(1.0)

    def test_finite_result(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((100, 10)).astype(np.float32)
        b = rng.standard_normal((100, 10)).astype(np.float32)
        assert np.isfinite(compute_mse(a, b))
