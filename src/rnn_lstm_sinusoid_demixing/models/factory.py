"""Model factory: create any model family from config.

The training loop calls create_model() instead of instantiating model
classes directly, so it never needs to know constructor argument details.
"""

import torch.nn as nn

from rnn_lstm_sinusoid_demixing.models.fully_connected import FullyConnectedModel
from rnn_lstm_sinusoid_demixing.models.lstm_model import LSTMModel
from rnn_lstm_sinusoid_demixing.models.rnn_model import RNNModel
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig

VALID_MODEL_TYPES = ("fc", "rnn", "lstm")


def create_model(
    model_type: str,
    signal_config: SignalConfig,
    training_config: TrainingConfig,
) -> nn.Module:
    """Instantiate a model from string type and config objects.

    Args:
        model_type: One of "fc", "rnn", "lstm".
        signal_config: Provides window_size and num_components.
        training_config: Provides per-model hidden_size defaults.

    Returns:
        Configured, untrained nn.Module.

    Raises:
        ValueError: If model_type is not in VALID_MODEL_TYPES.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {', '.join(VALID_MODEL_TYPES)}."
        )

    window_size = signal_config.context_window
    num_components = signal_config.num_components
    seq_input_size = 1 + num_components  # [sigma_t, c_1 .. c_C]

    if model_type == "fc":
        return FullyConnectedModel(
            window_size=window_size,
            num_components=num_components,
            hidden_size=training_config.fc_hidden_size,
        )
    if model_type == "rnn":
        return RNNModel(
            input_size=seq_input_size,
            hidden_size=training_config.rnn_hidden_size,
        )
    # model_type == "lstm"
    return LSTMModel(
        input_size=seq_input_size,
        hidden_size=training_config.lstm_hidden_size,
    )
