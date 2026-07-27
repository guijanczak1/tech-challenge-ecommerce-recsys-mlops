"""Modelos baseline de recomendação (registrados na Factory).

Referências para o modelo neural (Etapa 4): popularidade (não personalizado) e
fatoração de matriz com scikit-learn (TruncatedSVD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from recsys.models.factory import register_model


@register_model("popularity")
class PopularityRecommender:
    """Recomenda os itens mais populares (soma de pesos), iguais para todos."""

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

    def recommend(self, user: int, k: int) -> list[int]:
        """Top-k itens populares (o mesmo para qualquer ``user``)."""
        return self._ranked_items[:k]

    @property
    def ranked_items(self) -> list[int]:
        """Itens ordenados por popularidade (após ``fit``)."""
        return self._ranked_items


def _interaction_matrix(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    weight_col: str,
    shape: tuple[int, int],
) -> csr_matrix:
    """Monta a matriz esparsa usuário×item ponderada."""
    rows = interactions[user_col].to_numpy()
    cols = interactions[item_col].to_numpy()
    data = interactions[weight_col].to_numpy(dtype="float32")
    return csr_matrix((data, (rows, cols)), shape=shape)


@register_model("svd")
class SvdRecommender:
    """Fatoração de matriz por TruncatedSVD (baseline scikit-learn)."""

    def __init__(self, n_components: int = 32, seed: int = 42) -> None:
        """Guarda hiperparâmetros; fatores são aprendidos em :meth:`fit`."""
        self._n_components = n_components
        self._seed = seed
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str,
        item_col: str,
        weight_col: str,
        n_users: int,
        n_items: int,
    ) -> SvdRecommender:
        """Aprende fatores latentes de usuário e item a partir das interações."""
        matrix = _interaction_matrix(
            interactions, user_col, item_col, weight_col, (n_users, n_items)
        )
        components = max(min(self._n_components, min(matrix.shape) - 1), 1)
        svd = TruncatedSVD(n_components=components, random_state=self._seed)
        self._user_factors = svd.fit_transform(matrix)
        self._item_factors = svd.components_.T
        return self

    def recommend(self, user: int, k: int) -> list[int]:
        """Top-k itens por produto interno dos fatores de ``user``."""
        assert self._user_factors is not None and self._item_factors is not None
        scores = self._user_factors[user] @ self._item_factors.T
        return np.argsort(scores)[::-1][:k].tolist()
