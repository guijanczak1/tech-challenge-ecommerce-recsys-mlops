"""Métricas de recomendação top-k.

Na Etapa 3 há precision@k e recall@k; a Etapa 4 acrescenta NDCG@k e MAP@k
para a comparação com o modelo neural (≥ 4 métricas).
"""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fração dos ``k`` recomendados que são relevantes."""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fração dos itens relevantes que aparecem nos ``k`` recomendados."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & relevant) / len(relevant)
