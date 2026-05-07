"""Unit tests for data/dataloader.py."""

import numpy as np
import pytest
import torch

from rnn_lstm_sinusoid_demixing.data.dataloader import make_loader, split_dataset

_N = 200
_W = 10
_C = 4
_RNG = np.random.default_rng(0)

_INPUTS = _RNG.random((_N, _W), dtype=np.float32)
_SELS = np.tile(np.eye(_C, dtype=np.float32), (_N // _C, 1))
_TARGETS = _RNG.random((_N, _W), dtype=np.float32)


class TestSplitDataset:
    splits = split_dataset(_INPUTS, _SELS, _TARGETS, val_split=0.1, test_split=0.1)

    def test_returns_three_keys(self) -> None:
        assert set(self.splits) == {"train", "val", "test"}

    def test_total_count_preserved(self) -> None:
        total = sum(len(v[0]) for v in self.splits.values())
        assert total == _N

    def test_train_is_largest(self) -> None:
        assert len(self.splits["train"][0]) > len(self.splits["val"][0])
        assert len(self.splits["train"][0]) > len(self.splits["test"][0])

    def test_approximate_val_ratio(self) -> None:
        n_val = len(self.splits["val"][0])
        assert abs(n_val / _N - 0.1) < 0.05

    def test_approximate_test_ratio(self) -> None:
        n_test = len(self.splits["test"][0])
        assert abs(n_test / _N - 0.1) < 0.05

    def test_deterministic_same_seed(self) -> None:
        s1 = split_dataset(_INPUTS, _SELS, _TARGETS, random_seed=7)
        s2 = split_dataset(_INPUTS, _SELS, _TARGETS, random_seed=7)
        np.testing.assert_array_equal(s1["train"][0], s2["train"][0])

    def test_different_seed_different_split(self) -> None:
        s1 = split_dataset(_INPUTS, _SELS, _TARGETS, random_seed=1)
        s2 = split_dataset(_INPUTS, _SELS, _TARGETS, random_seed=2)
        assert not np.array_equal(s1["train"][0], s2["train"][0])

    def test_invalid_splits_raise(self) -> None:
        with pytest.raises(ValueError):
            split_dataset(_INPUTS, _SELS, _TARGETS, val_split=0.6, test_split=0.6)


class TestMakeLoader:
    inp, sel, tgt = split_dataset(_INPUTS, _SELS, _TARGETS)["train"]

    def test_fc_loader_input_shape(self) -> None:
        loader = make_loader(self.inp, self.sel, self.tgt, "fc", batch_size=16)
        x, y = next(iter(loader))
        assert x.shape == (16, _W + _C)
        assert y.shape == (16, _W)

    def test_rnn_loader_input_shape(self) -> None:
        loader = make_loader(self.inp, self.sel, self.tgt, "rnn", batch_size=16)
        x, _ = next(iter(loader))
        assert x.shape == (16, _W, 1 + _C)

    def test_lstm_loader_input_shape(self) -> None:
        loader = make_loader(self.inp, self.sel, self.tgt, "lstm", batch_size=16)
        x, _ = next(iter(loader))
        assert x.shape == (16, _W, 1 + _C)

    def test_loader_yields_float32(self) -> None:
        loader = make_loader(self.inp, self.sel, self.tgt, "fc", batch_size=16)
        x, y = next(iter(loader))
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(ValueError):
            make_loader(self.inp, self.sel, self.tgt, "transformer", batch_size=16)

    def test_shuffle_false_is_deterministic(self) -> None:
        loader1 = make_loader(self.inp, self.sel, self.tgt, "fc", batch_size=16, shuffle=False)
        loader2 = make_loader(self.inp, self.sel, self.tgt, "fc", batch_size=16, shuffle=False)
        x1, _ = next(iter(loader1))
        x2, _ = next(iter(loader2))
        assert torch.equal(x1, x2)
