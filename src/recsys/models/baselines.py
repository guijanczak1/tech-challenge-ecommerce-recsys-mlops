"""Modelos baseline de recomendação (registrados na Factory).

Na Etapa 3 há o baseline de popularidade, que serve de referência para o
modelo neural (PyTorch) da Etapa 4.
"""

from __future__ import annotations

import pandas as pd

from recsys.models.factory import register_model


@register_model("popularity")
class PopularityRecommender:
    """Recomenda os itens mais populares (soma de pesos), iguais para todos.

    Baseline não personalizado: simples, forte e determinístico.
    """

    def __init__(self) -> None:
        """Inicializa sem itens (chame :meth:`fit` antes de recomendar)."""
        self._ranked_items: list[int] = []

    def fit(
        self, interactions: pd.DataFrame, item_col: str, weight_col: str
    ) -> PopularityRecommender:
        """Ordena os itens pela soma de pesos (mais popular primeiro)."""
        popularity = interactions.groupby(item_col)[weight_col].sum()
        self._ranked_items = popularity.sort_values(ascending=False).index.tolist()
        return self

    def recommend(self, k: int) -> list[int]:
        """Retorna os ``k`` itens mais populares."""
        return self._ranked_items[:k]

    @property
    def ranked_items(self) -> list[int]:
        """Itens ordenados por popularidade (após ``fit``)."""
        return self._ranked_items
