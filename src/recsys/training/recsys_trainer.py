"""Trainer concreto do modelo neural (implementa o Template Method BaseTrainer).

Treina com ``BCEWithLogitsLoss`` sobre feedback implícito, salva o melhor
checkpoint (menor perda de validação) e é device-agnostic.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from recsys.config import AppConfig
from recsys.training.trainer import BaseTrainer
from recsys.utils.device import resolve_device


class RecsysTrainer(BaseTrainer):
    """Laço de treino do :class:`RecsysMLP` com early stopping e checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: AppConfig,
        checkpoint_path: Path,
    ) -> None:
        """Configura modelo, otimizador, perda e device."""
        super().__init__(patience=cfg.training.patience)
        self._device = resolve_device(cfg.training.device)
        self._model = model.to(self._device)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._checkpoint_path = checkpoint_path
        self._best_val = math.inf

    def train_epoch(self) -> float:
        """Uma época de treino; retorna a perda média."""
        self._model.train()
        losses = [self._step(u, i, y) for u, i, y in self._train_loader]
        return sum(losses) / max(len(losses), 1)

    def _step(self, users: torch.Tensor, items: torch.Tensor, labels: torch.Tensor) -> float:
        """Um passo de otimização; retorna a perda do batch."""
        users, items, labels = self._to_device(users, items, labels)
        self._optimizer.zero_grad()
        loss = self._loss_fn(self._model(users, items), labels)
        loss.backward()
        self._optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def validate(self) -> float:
        """Perda média de validação; salva o melhor checkpoint."""
        self._model.eval()
        losses = []
        for users, items, labels in self._val_loader:
            users, items, labels = self._to_device(users, items, labels)
            losses.append(float(self._loss_fn(self._model(users, items), labels).item()))
        val_loss = sum(losses) / max(len(losses), 1)
        self._save_if_best(val_loss)
        return val_loss

    def _to_device(
        self, users: torch.Tensor, items: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move um batch para o device do modelo."""
        return users.to(self._device), items.to(self._device), labels.to(self._device)

    def _save_if_best(self, val_loss: float) -> None:
        """Persiste o ``state_dict`` quando a validação melhora."""
        if val_loss < self._best_val:
            self._best_val = val_loss
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), self._checkpoint_path)
