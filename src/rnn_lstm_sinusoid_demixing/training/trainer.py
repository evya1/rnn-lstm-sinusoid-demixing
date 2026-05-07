"""Training loop for sinusoid demixing models."""

from __future__ import annotations

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
        self.model.to(device)

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch and return mean loss.

        Args:
            loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate on a dataloader and return mean loss.

        Args:
            loader: DataLoader yielding (inputs, targets) batches.

        Returns:
            Mean evaluation loss.
        """
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                total += self.loss_fn(self.model(x), y).item()
        return total / len(loader)

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
        history: dict[str, list[float]] = {"train": [], "val": []}
        for _ in range(num_epochs):
            history["train"].append(self.train_epoch(train_loader))
            history["val"].append(self.evaluate(val_loader))
        return history
