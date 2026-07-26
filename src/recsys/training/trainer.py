"""Laço de treino genérico (padrão Template Method).

:class:`BaseTrainer` fixa o esqueleto do treino (start → épocas com early
stopping → end); subclasses implementam apenas ``train_epoch`` e ``validate``.
Hooks opcionais permitem estender sem alterar o esqueleto.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Histórico de um treino."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf
    stopped_early: bool = False


class BaseTrainer(ABC):
    """Esqueleto de treino com early stopping por paciência."""

    def __init__(self, patience: int) -> None:
        """Inicializa o controle de early stopping.

        Args:
            patience: Épocas sem melhora de ``val_loss`` antes de parar.
        """
        self._patience = patience
        self._best_val = math.inf
        self._best_epoch = 0
        self._epochs_no_improve = 0

    def fit(self, epochs: int) -> TrainingHistory:
        """Executa o treino (Template Method).

        Args:
            epochs: Número máximo de épocas.

        Returns:
            O :class:`TrainingHistory` do treino.
        """
        self.on_train_start()
        history = TrainingHistory()
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            self.on_epoch_end(epoch, train_loss, val_loss)
            if self._should_stop(val_loss, epoch):
                history.stopped_early = True
                break
        history.best_epoch = self._best_epoch
        history.best_val_loss = self._best_val
        self.on_train_end()
        return history

    def _should_stop(self, val_loss: float, epoch: int) -> bool:
        """Atualiza o melhor resultado e decide se deve parar."""
        if val_loss < self._best_val:
            self._best_val = val_loss
            self._best_epoch = epoch
            self._epochs_no_improve = 0
            return False
        self._epochs_no_improve += 1
        return self._epochs_no_improve >= self._patience

    @abstractmethod
    def train_epoch(self) -> float:
        """Treina uma época e retorna a perda de treino."""

    @abstractmethod
    def validate(self) -> float:
        """Avalia no conjunto de validação e retorna a perda (métrica de parada)."""

    # Hooks opcionais do Template Method: vazios de propósito (no-op),
    # feitos para subclasses sobrescreverem. B027 é esperado aqui.
    def on_train_start(self) -> None:  # noqa: B027
        """Hook chamado uma vez, antes da primeira época (no-op por padrão)."""

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:  # noqa: B027
        """Hook chamado ao fim de cada época (no-op por padrão)."""

    def on_train_end(self) -> None:  # noqa: B027
        """Hook chamado uma vez, após a última época (no-op por padrão)."""
