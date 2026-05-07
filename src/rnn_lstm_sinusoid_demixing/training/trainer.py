"""Training loop for sinusoid demixing models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader


class Trainer:
    """Encapsulates the training and validation loop for a single model.

    Args:
        model: The PyTorch model to train.
        optimizer: Optimizer instance.
        loss_fn: Loss function (e.g. MSELoss).
        device: torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch and return mean loss.

        Args:
            loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Mean training loss for the epoch.
        """
        raise NotImplementedError("Phase 7: Trainer.train_epoch")

    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate on a dataloader and return mean loss.

        Args:
            loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Mean evaluation loss.
        """
        raise NotImplementedError("Phase 7: Trainer.evaluate")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> dict[str, list[float]]:
        """Train for num_epochs and return loss history.

        Returns:
            {'train': [...], 'val': [...]} loss per epoch.
        """
        raise NotImplementedError("Phase 7: Trainer.fit")
