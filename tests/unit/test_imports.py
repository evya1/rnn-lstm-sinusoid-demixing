"""Smoke tests: verify all subpackages import without error."""


def test_import_package() -> None:
    import rnn_lstm_sinusoid_demixing  # noqa: F401


def test_import_constants() -> None:
    from rnn_lstm_sinusoid_demixing import constants  # noqa: F401


def test_import_shared_config() -> None:
    from rnn_lstm_sinusoid_demixing.shared import config  # noqa: F401


def test_import_shared_paths() -> None:
    from rnn_lstm_sinusoid_demixing.shared import paths  # noqa: F401


def test_import_data_signal_generator() -> None:
    from rnn_lstm_sinusoid_demixing.data import signal_generator  # noqa: F401


def test_import_data_noise() -> None:
    from rnn_lstm_sinusoid_demixing.data import noise  # noqa: F401


def test_import_data_dataset_builder() -> None:
    from rnn_lstm_sinusoid_demixing.data import dataset_builder  # noqa: F401


def test_import_models_fully_connected() -> None:
    from rnn_lstm_sinusoid_demixing.models import fully_connected  # noqa: F401


def test_import_models_rnn() -> None:
    from rnn_lstm_sinusoid_demixing.models import rnn_model  # noqa: F401


def test_import_models_lstm() -> None:
    from rnn_lstm_sinusoid_demixing.models import lstm_model  # noqa: F401


def test_import_training_trainer() -> None:
    from rnn_lstm_sinusoid_demixing.training import trainer  # noqa: F401


def test_import_training_losses() -> None:
    from rnn_lstm_sinusoid_demixing.training import losses  # noqa: F401


def test_import_evaluation_metrics() -> None:
    from rnn_lstm_sinusoid_demixing.evaluation import metrics  # noqa: F401


def test_import_evaluation_compare() -> None:
    from rnn_lstm_sinusoid_demixing.evaluation import compare  # noqa: F401


def test_import_visualization_plots() -> None:
    from rnn_lstm_sinusoid_demixing.visualization import plots  # noqa: F401


def test_import_models_input_prep() -> None:
    from rnn_lstm_sinusoid_demixing.models import input_prep  # noqa: F401


def test_import_models_factory() -> None:
    from rnn_lstm_sinusoid_demixing.models import factory  # noqa: F401


def test_import_data_dataloader() -> None:
    from rnn_lstm_sinusoid_demixing.data import dataloader  # noqa: F401


def test_import_experiments_runner() -> None:
    from rnn_lstm_sinusoid_demixing.experiments import runner  # noqa: F401


def test_import_sdk() -> None:
    from rnn_lstm_sinusoid_demixing.sdk import sdk  # noqa: F401
