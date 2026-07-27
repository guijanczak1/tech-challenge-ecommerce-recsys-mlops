"""Recomendador neural: embeddings de usuário/item + MLP (PyTorch).

Registrado na Factory como ``mlp``. Produz um score por par (usuário, item);
``recommend`` retorna os top-k itens para um usuário.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from recsys.models.factory import register_model


@register_model("mlp")
class RecsysMLP(nn.Module):
    """Embeddings de usuário e item concatenados alimentando um MLP."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        """Cria as tabelas de embedding e a cabeça MLP."""
        super().__init__()
        self.n_items = n_items
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.mlp = self._build_mlp(2 * embedding_dim, hidden_dims, dropout)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
        """Empilha camadas Linear+ReLU+Dropout terminando num logit."""
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score (logit) para cada par (usuário, item) do batch."""
        features = torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)
        return self.mlp(features).squeeze(-1)

    @torch.no_grad()
    def recommend(self, user: int, k: int) -> list[int]:
        """Top-k itens de maior score para ``user``."""
        users = torch.full((self.n_items,), user, dtype=torch.long)
        items = torch.arange(self.n_items, dtype=torch.long)
        scores = self.forward(users, items)
        return torch.topk(scores, min(k, self.n_items)).indices.tolist()
