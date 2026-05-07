"""Dataset splitting and DataLoader construction for sinusoid demixing."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from rnn_lstm_sinusoid_demixing.models.input_prep import prepare_fc_input, prepare_seq_input

_VALID_MODEL_TYPES = ("fc", "rnn", "lstm")


def split_dataset(
    inputs: NDArray,
    selectors: NDArray,
    targets: NDArray,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
) -> dict[str, tuple[NDArray, NDArray, NDArray]]:
    """Shuffle and split arrays into train / val / test subsets.

    Args:
        inputs:      (N, window_size) composite windows.
        selectors:   (N, num_components) one-hot selectors.
        targets:     (N, window_size) clean component windows.
        val_split:   Fraction reserved for validation.
        test_split:  Fraction reserved for testing.
        random_seed: Seed for reproducible shuffling.

    Returns:
        Dict with keys 'train', 'val', 'test', each a tuple
        (inputs, selectors, targets) of shuffled arrays.
    """
    if not (0 < val_split + test_split < 1):
        raise ValueError("val_split + test_split must be in (0, 1).")

    n = len(inputs)
    idx = np.random.default_rng(random_seed).permutation(n)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }
    return {
        name: (inputs[i], selectors[i], targets[i]) for name, i in splits.items()
    }


def make_loader(
    inputs: NDArray,
    selectors: NDArray,
    targets: NDArray,
    model_type: str,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader that yields model-ready (input, target) batches.

    Applies prepare_fc_input or prepare_seq_input so the Trainer receives
    tensors it can pass directly to the model.

    Args:
        inputs:      (N, window_size) float32 composite windows.
        selectors:   (N, num_components) float32 one-hot selectors.
        targets:     (N, window_size) float32 clean windows.
        model_type:  One of 'fc', 'rnn', 'lstm'.
        batch_size:  Mini-batch size.
        shuffle:     Whether to shuffle each epoch.

    Returns:
        DataLoader yielding (model_input, target) tensor pairs.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected one of {_VALID_MODEL_TYPES}."
        )

    windows_t = torch.from_numpy(inputs)
    sels_t = torch.from_numpy(selectors)
    targets_t = torch.from_numpy(targets)

    if model_type == "fc":
        model_input = prepare_fc_input(windows_t, sels_t)
    else:
        model_input = prepare_seq_input(windows_t, sels_t)

    dataset = TensorDataset(model_input, targets_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
