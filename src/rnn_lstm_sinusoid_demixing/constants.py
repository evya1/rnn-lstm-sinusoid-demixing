"""Project-wide default constants.

These values mirror config/default.json and serve as typed fallbacks.
Configurable experiment parameters belong in config files or config dataclasses,
not scattered as magic numbers inside logic modules.
"""

# Signal generation defaults
DEFAULT_FREQUENCIES: list[float] = [1.0, 3.0, 5.0, 7.0]
DEFAULT_SAMPLING_RATE: int = 1000
DEFAULT_DURATION_SECONDS: float = 10.0
DEFAULT_NUM_COMPONENTS: int = 4

# Dataset defaults
DEFAULT_CONTEXT_WINDOW: int = 10
DEFAULT_NOISE_LEVEL: float = 0.1

# Training defaults
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_NUM_EPOCHS: int = 50
DEFAULT_VALIDATION_SPLIT: float = 0.1
DEFAULT_TEST_SPLIT: float = 0.1
DEFAULT_RANDOM_SEED: int = 42

# Experiment noise sweep
DEFAULT_NOISE_LEVELS: list[float] = [0.00, 0.01, 0.05, 0.10, 0.20]

# Model hidden sizes
DEFAULT_FC_HIDDEN_SIZE: int = 64
DEFAULT_RNN_HIDDEN_SIZE: int = 64
DEFAULT_LSTM_HIDDEN_SIZE: int = 64
