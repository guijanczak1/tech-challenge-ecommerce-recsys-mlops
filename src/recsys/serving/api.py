"""API de inferência (FastAPI) que serve o modelo em produção.

O modelo é carregado no startup a partir do MLflow Model Registry
(``models:/recsys-mlp@production``). ``create_app`` recebe um ``provider`` para
permitir injeção de um modelo dummy nos testes.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from recsys.data.catalog import product_name
from recsys.models.base import Recommender

REGISTERED_MODEL = "recsys-mlp"
_LOCAL_MODEL_PATH = Path("models/mlp.pt")


def production_model() -> Recommender:
    """Carrega o modelo de produção do MLflow Model Registry."""
    import mlflow.pytorch

    return mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL}@production")


def default_provider() -> Recommender:
    """Usa o modelo embutido (imagem self-contained) se existir; senão, o Registry."""
    if _LOCAL_MODEL_PATH.exists():
        from recsys.serving.loader import load_local_model

        return load_local_model()
    return production_model()


def _enrich(item_ids: list[int]) -> list[dict[str, Any]]:
    """Combina cada item_id com nome/categoria fictícios (catálogo de demo)."""
    return [{"id": item_id, **product_name(item_id)} for item_id in item_ids]


def _register_routes(app: FastAPI) -> None:
    """Registra os endpoints ``/health`` e ``/recommend`` na app."""

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @app.get("/recommend")
    def recommend(user: int = Query(..., ge=0), k: int = Query(default=10, ge=0)) -> dict[str, Any]:
        """Retorna os top-k itens recomendados para ``user``.

        ``name``/``category`` vêm de um catálogo **fictício** de demonstração
        — o RetailRocket (dataset real sugerido) não expõe nomes de produto,
        apenas IDs e propriedades hasheadas por anonimização.
        """
        try:
            item_ids = app.state.model.recommend(user, k)
        except IndexError as exc:
            raise HTTPException(
                status_code=404, detail=f"user {user} fora do catálogo conhecido"
            ) from exc
        return {"user": user, "k": k, "items": _enrich(item_ids)}


def create_app(provider: Callable[[], Recommender] = default_provider) -> FastAPI:
    """Cria a app FastAPI; o modelo é carregado no startup via ``provider``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        """Carrega o modelo uma vez no ciclo de vida da aplicação."""
        app.state.model = provider()
        yield

    app = FastAPI(title="recsys-api", lifespan=lifespan)
    _register_routes(app)
    return app


app = create_app()
