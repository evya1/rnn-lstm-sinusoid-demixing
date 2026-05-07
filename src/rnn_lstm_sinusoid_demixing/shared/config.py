"""Typed configuration dataclasses for signal generation and training."""

from dataclasses import dataclass, field

from rnn_lstm_sinusoid_demixing import constants


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for sinusoidal signal generation.

    Args:
        frequencies: Hz values for each sinusoidal component.
        sampling_rate: Samples per second.
        duration_seconds: Total signal length in seconds.
        num_components: Number of sinusoidal components (must be 4 by default).
        noise_level: Standard deviation of Gaussian noise added to each component.
        context_window: Number of samples per input window.
        random_seed: Seed for reproducibility.
    """

    frequencies: tuple[float, ...] = field(
        default_factory=lambda: tuple(constants.DEFAULT_FREQUENCIES)
    )
    sampling_rate: int = constants.DEFAULT_SAMPLING_RATE
    duration_seconds: float = constants.DEFAULT_DURATION_SECONDS
    num_components: int = constants.DEFAULT_NUM_COMPONENTS
    noise_level: float = constants.DEFAULT_NOISE_LEVEL
    context_window: int = constants.DEFAULT_CONTEXT_WINDOW
    random_seed: int = constants.DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        if self.num_components <= 0:
            raise ValueError("num_components must be positive.")
        if len(self.frequencies) != self.num_components:
            raise ValueError(
                f"len(frequencies) must equal num_components ({self.num_components})."
            )
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative.")
        num_samples = int(self.sampling_rate * self.duration_seconds)
        if self.context_window >= num_samples:
            raise ValueError("context_window must be smaller than total signal length.")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training.

    Args:
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        num_epochs: Number of training epochs.
        validation_split: Fraction of data reserved for validation.
        test_split: Fraction of data reserved for test.
        random_seed: Seed for reproducibility.
        fc_hidden_size: Hidden layer size for the FC model.
        rnn_hidden_size: Hidden size for the RNN model.
        lstm_hidden_size: Hidden size for the LSTM model.
    """

    batch_size: int = constants.DEFAULT_BATCH_SIZE
    learning_rate: float = constants.DEFAULT_LEARNING_RATE
    num_epochs: int = constants.DEFAULT_NUM_EPOCHS
    validation_split: float = constants.DEFAULT_VALIDATION_SPLIT
    test_split: float = constants.DEFAULT_TEST_SPLIT
    random_seed: int = constants.DEFAULT_RANDOM_SEED
    fc_hidden_size: int = constants.DEFAULT_FC_HIDDEN_SIZE
    rnn_hidden_size: int = constants.DEFAULT_RNN_HIDDEN_SIZE
    lstm_hidden_size: int = constants.DEFAULT_LSTM_HIDDEN_SIZE

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        total_split = self.validation_split + self.test_split
        if not (0 < total_split < 1):
            raise ValueError("validation_split + test_split must be in (0, 1).")
