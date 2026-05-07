"""High-level SDK facade for the sinusoid demixing pipeline."""

from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig


class DemixingSDK:
    """Thin facade that orchestrates data generation, training, and evaluation.

    This class is intended for programmatic use and CLI orchestration.
    It delegates all logic to the domain modules in data/, models/, training/,
    evaluation/, and visualization/.

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

    def run(self) -> None:
        """Execute the full pipeline: generate → train → evaluate → plot."""
        raise NotImplementedError("Phase 7: DemixingSDK.run")
