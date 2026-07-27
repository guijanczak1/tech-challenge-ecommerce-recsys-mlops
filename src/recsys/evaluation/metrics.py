"""Métricas de recomendação top-k.

Quatro métricas por lista: precision@k, recall@k, NDCG@k e average
precision@k (agregada em MAP@k na avaliação).
"""

from __future__ import annotations

import math
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


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Ganho cumulativo descontado normalizado nos ``k`` recomendados."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Precisão média nos ``k`` primeiros (base do MAP@k)."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (rank + 1)
    denominator = min(k, len(relevant))
    return score / denominator if denominator > 0 else 0.0
