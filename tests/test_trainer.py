"""Testes de src/recsys/training/trainer.py."""

from __future__ import annotations

from collections.abc import Sequence

from recsys.training.trainer import BaseTrainer


class FakeTrainer(BaseTrainer):
    """Trainer de teste que registra a ordem das chamadas."""

    def __init__(self, val_losses: Sequence[float], patience: int) -> None:
        super().__init__(patience)
        self._val_losses = list(val_losses)
        self._index = 0
        self.calls: list[str] = []

    def train_epoch(self) -> float:
        self.calls.append("train_epoch")
        return 1.0

    def validate(self) -> float:
        value = self._val_losses[self._index]
        self._index += 1
        self.calls.append("validate")
        return value

    def on_train_start(self) -> None:
        self.calls.append("on_train_start")

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.calls.append(f"on_epoch_end:{epoch}")

    def on_train_end(self) -> None:
        self.calls.append("on_train_end")


def test_call_order_without_early_stop() -> None:
    trainer = FakeTrainer(val_losses=[1.0, 0.9, 0.8], patience=2)
    history = trainer.fit(epochs=3)
    assert trainer.calls == [
        "on_train_start",
        "train_epoch",
        "validate",
        "on_epoch_end:1",
        "train_epoch",
        "validate",
        "on_epoch_end:2",
        "train_epoch",
        "validate",
        "on_epoch_end:3",
        "on_train_end",
    ]
    assert history.stopped_early is False
    assert history.best_epoch == 3
    assert history.best_val_loss == 0.8


def test_early_stopping_triggers() -> None:
    # melhora na época 2 (0.5); épocas 3 e 4 sem melhora -> para em 4 (patience=2).
    trainer = FakeTrainer(val_losses=[1.0, 0.5, 0.6, 0.7, 0.7], patience=2)
    history = trainer.fit(epochs=10)
    assert history.stopped_early is True
    assert history.best_epoch == 2
    assert history.best_val_loss == 0.5
    assert len(history.val_losses) == 4  # parou após a 4ª época
    assert trainer.calls[-1] == "on_train_end"


def test_hooks_are_optional_noop_by_default() -> None:
    class Minimal(BaseTrainer):
        def train_epoch(self) -> float:
            return 0.1

        def validate(self) -> float:
            return 0.1

    history = Minimal(patience=1).fit(epochs=1)
    assert history.val_losses == [0.1]
    assert history.best_epoch == 1
