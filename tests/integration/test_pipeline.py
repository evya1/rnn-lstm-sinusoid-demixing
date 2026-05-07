"""Integration smoke test: data generation → dataset → model forward pass.

Connects Phases 04, 05, and 06 in a single pipeline to verify that the
modules work together correctly end-to-end, not just in isolation.
"""

import numpy as np
import torch

from rnn_lstm_sinusoid_demixing.data.dataset_builder import build_dataset
from rnn_lstm_sinusoid_demixing.data.signal_generator import build_signals
from rnn_lstm_sinusoid_demixing.models.factory import VALID_MODEL_TYPES, create_model
from rnn_lstm_sinusoid_demixing.models.input_prep import prepare_fc_input, prepare_seq_input
from rnn_lstm_sinusoid_demixing.shared.config import SignalConfig, TrainingConfig

_SC = SignalConfig(sampling_rate=100, duration_seconds=1.0)
_TC = TrainingConfig()
_W = _SC.context_window   # 10
_C = _SC.num_components   # 4
_B = 4  # small batch for smoke testing


def _make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a real batch from the signal + dataset pipeline."""
    _, clean, _, composite = build_signals(_SC)
    inputs, selectors, targets = build_dataset(composite, clean, _W, _C)

    idx = np.random.default_rng(0).integers(0, len(inputs), size=_B)
    windows = torch.from_numpy(inputs[idx])
    sels = torch.from_numpy(selectors[idx])
    tgts = torch.from_numpy(targets[idx])
    return windows, sels, tgts


class TestEndToEndPipeline:
    windows, sels, tgts = _make_batch()

    def test_batch_shapes_from_pipeline(self) -> None:
        assert self.windows.shape == (_B, _W)
        assert self.sels.shape == (_B, _C)
        assert self.tgts.shape == (_B, _W)

    def test_fc_forward_pass_with_real_data(self) -> None:
        model = create_model("fc", _SC, _TC)
        x = prepare_fc_input(self.windows, self.sels)
        out = model(x)
        assert out.shape == (_B, _W)
        assert torch.all(torch.isfinite(out))

    def test_rnn_forward_pass_with_real_data(self) -> None:
        model = create_model("rnn", _SC, _TC)
        x = prepare_seq_input(self.windows, self.sels)
        out = model(x)
        assert out.shape == (_B, _W)
        assert torch.all(torch.isfinite(out))

    def test_lstm_forward_pass_with_real_data(self) -> None:
        model = create_model("lstm", _SC, _TC)
        x = prepare_seq_input(self.windows, self.sels)
        out = model(x)
        assert out.shape == (_B, _W)
        assert torch.all(torch.isfinite(out))

    def test_all_model_types_run(self) -> None:
        """Smoke: every model type in VALID_MODEL_TYPES completes a forward pass."""
        for model_type in VALID_MODEL_TYPES:
            model = create_model(model_type, _SC, _TC)
            if model_type == "fc":
                x = prepare_fc_input(self.windows, self.sels)
            else:
                x = prepare_seq_input(self.windows, self.sels)
            out = model(x)
            assert torch.all(torch.isfinite(out)), f"{model_type} produced non-finite output"

    def test_target_dtype_matches_model_output(self) -> None:
        model = create_model("fc", _SC, _TC)
        x = prepare_fc_input(self.windows, self.sels)
        out = model(x)
        assert out.dtype == self.tgts.dtype
