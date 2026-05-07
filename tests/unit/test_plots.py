"""Unit tests for visualization/plots.py."""

import numpy as np

from rnn_lstm_sinusoid_demixing.visualization.plots import (
    plot_loss_curves,
    plot_mse_vs_noise,
    plot_prediction_vs_target,
    plot_signals,
)

_T = np.linspace(0, 1, 100, dtype=np.float32)
_SIG = np.sin(2 * np.pi * 3 * _T)
_HISTORY = {"train": [0.5, 0.4, 0.3, 0.2], "val": [0.55, 0.45, 0.35, 0.25]}
_NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
_MSE_BY_MODEL = {
    "fc": [0.01, 0.02, 0.04, 0.10],
    "rnn": [0.008, 0.015, 0.03, 0.08],
    "lstm": [0.007, 0.012, 0.025, 0.07],
}


class TestPlotSignals:
    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "signals.png"
        plot_signals(_T, {"sin": _SIG}, "Test", save_path=out)
        assert out.exists()

    def test_file_nonempty(self, tmp_path) -> None:
        out = tmp_path / "signals.png"
        plot_signals(_T, {"sin": _SIG}, "Test", save_path=out)
        assert out.stat().st_size > 0

    def test_multi_signal(self, tmp_path) -> None:
        out = tmp_path / "multi.png"
        plot_signals(_T, {"a": _SIG, "b": _SIG * 0.5}, "Multi", save_path=out)
        assert out.exists()


class TestPlotLossCurves:
    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "loss.png"
        plot_loss_curves(_HISTORY, "fc", save_path=out)
        assert out.exists()

    def test_file_nonempty(self, tmp_path) -> None:
        out = tmp_path / "loss.png"
        plot_loss_curves(_HISTORY, "lstm", save_path=out)
        assert out.stat().st_size > 0

    def test_single_epoch(self, tmp_path) -> None:
        out = tmp_path / "single_epoch.png"
        plot_loss_curves({"train": [0.5], "val": [0.6]}, "rnn", save_path=out)
        assert out.exists()


class TestPlotPredictionVsTarget:
    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "pred.png"
        plot_prediction_vs_target(_SIG, _SIG * 0.9, "fc", save_path=out)
        assert out.exists()

    def test_file_nonempty(self, tmp_path) -> None:
        out = tmp_path / "pred.png"
        plot_prediction_vs_target(_SIG, _SIG * 1.1, "lstm", save_path=out)
        assert out.stat().st_size > 0


class TestPlotMseVsNoise:
    def test_saves_file(self, tmp_path) -> None:
        out = tmp_path / "mse_noise.png"
        plot_mse_vs_noise(_NOISE_LEVELS, _MSE_BY_MODEL, save_path=out)
        assert out.exists()

    def test_file_nonempty(self, tmp_path) -> None:
        out = tmp_path / "mse_noise.png"
        plot_mse_vs_noise(_NOISE_LEVELS, _MSE_BY_MODEL, save_path=out)
        assert out.stat().st_size > 0

    def test_single_model(self, tmp_path) -> None:
        out = tmp_path / "single.png"
        plot_mse_vs_noise([0.0, 0.1], {"fc": [0.01, 0.05]}, save_path=out)
        assert out.exists()
