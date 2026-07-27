"""Contrato comum dos recomendadores (usado na avaliação)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Recommender(Protocol):
    """Interface mínima: recomendar top-k itens para um usuário."""

    def recommend(self, user: int, k: int) -> list[int]:
        """Retorna os ``k`` itens recomendados para ``user``."""
        ...
