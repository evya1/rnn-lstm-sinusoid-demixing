"""High-level SDK facade for the sinusoid demixing pipeline."""

from __future__ import annotations

import json

from rnn_lstm_sinusoid_demixing import constants
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig
from rnn_lstm_sinusoid_demixing.shared.paths import results_dir


class DemixingSDK:
    """Thin facade that orchestrates data generation, training, and evaluation.

    Delegates all logic to data/, models/, training/, evaluation/,
    experiments/, and visualization/ modules.

    Args:
        signal_config: Signal and dataset configuration.
        training_config: Model training configuration.
    """

    def __init__(
        self,
        signal_config: SignalConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> None:
        self.signal_config = signal_config or SignalConfig()
        self.training_config = training_config or TrainingConfig()

    def run(self) -> dict:
        """Execute the full pipeline: generate → train → evaluate → plot.

        Saves all artefacts (PNG plots + JSON summary) to results/.

        Returns:
            Summary dict mapping model_type -> test_mse for the default noise level.
        """
        from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
        from rnn_lstm_sinusoid_demixing.experiments.runner import run_noise_sweep, run_single
        from rnn_lstm_sinusoid_demixing.visualization.plots import (
            plot_loss_curves,
            plot_mse_vs_noise,
            plot_prediction_vs_target,
            plot_signals,
        )

        rdir = results_dir()
        sc = self.signal_config
        tc = self.training_config

        # --- Signal overview plots ---
        time, clean, noisy, composite = build_signals(sc)
        plot_signals(time, {f"{sc.frequencies[i]} Hz": clean[i] for i in range(sc.num_components)}, "Clean Components", rdir / "signals_clean.png")
        plot_signals(time, {f"{sc.frequencies[i]} Hz (noisy)": noisy[i] for i in range(sc.num_components)}, "Noisy Components", rdir / "signals_noisy.png")
        plot_signals(time, {"Noisy composite": composite}, "Noisy Composite Signal", rdir / "signals_composite.png")

        # --- Train all three models at default noise level ---
        results = run_single(sc, tc)
        for model_type, res in results.items():
            plot_loss_curves(res.history, model_type, rdir / f"loss_curves_{model_type}.png")
            plot_prediction_vs_target(res.sample_target, res.sample_prediction, model_type, rdir / f"prediction_vs_target_{model_type}.png")

        # --- Noise sweep ---
        noise_levels = list(constants.DEFAULT_NOISE_LEVELS)
        mse_by_model = run_noise_sweep(tc, noise_levels, sc)
        plot_mse_vs_noise(noise_levels, mse_by_model, rdir / "mse_vs_noise.png")

        # --- Save JSON summaries ---
        summary = {mt: float(res.test_mse) for mt, res in results.items()}
        (rdir / "mse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        sweep_data = {
            "noise_levels": noise_levels,
            "mse_by_model": {mt: [float(v) for v in vals] for mt, vals in mse_by_model.items()},
        }
        (rdir / "mse_noise_sweep.json").write_text(json.dumps(sweep_data, indent=2), encoding="utf-8")
        _save_config(sc, tc, rdir)

        return summary


def _save_config(sc: SignalConfig, tc: TrainingConfig, rdir) -> None:
    config = {
        "signal": {"frequencies": list(sc.frequencies), "sampling_rate": sc.sampling_rate, "duration_seconds": sc.duration_seconds, "noise_level": sc.noise_level, "context_window": sc.context_window},
        "training": {"batch_size": tc.batch_size, "learning_rate": tc.learning_rate, "num_epochs": tc.num_epochs, "random_seed": tc.random_seed},
    }
    (rdir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
