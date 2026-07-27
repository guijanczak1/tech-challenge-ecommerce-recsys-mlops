"""API de inferência (FastAPI) que serve o modelo em produção.

O modelo é carregado no startup a partir do MLflow Model Registry
(``models:/recsys-mlp@production``). ``create_app`` recebe um ``provider`` para
permitir injeção de um modelo dummy nos testes.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from recsys.models.base import Recommender

REGISTERED_MODEL = "recsys-mlp"


def production_model() -> Recommender:
    """Carrega o modelo de produção do MLflow Model Registry."""
    import mlflow.pytorch

    return mlflow.pytorch.load_model(f"models:/{REGISTERED_MODEL}@production")


def _register_routes(app: FastAPI) -> None:
    """Registra os endpoints ``/health`` e ``/recommend`` na app."""

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @app.get("/recommend")
    def recommend(user: int, k: int = 10) -> dict[str, Any]:
        """Retorna os top-k itens recomendados para ``user``."""
        items = app.state.model.recommend(user, k)
        return {"user": user, "k": k, "items": items}


def create_app(provider: Callable[[], Recommender] = production_model) -> FastAPI:
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
