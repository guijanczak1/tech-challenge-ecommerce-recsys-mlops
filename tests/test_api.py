"""Testes da API de inferência (com modelo dummy injetado)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from recsys.serving.api import create_app


class _DummyModel:
    def recommend(self, user: int, k: int) -> list[int]:
        return list(range(k))


def test_health() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_items() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend", params={"user": 7, "k": 3})
    body = response.json()
    assert response.status_code == 200
    assert body == {"user": 7, "k": 3, "items": [0, 1, 2]}


def test_recommend_requires_user() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend")  # falta 'user'
    assert response.status_code == 422  # validação do FastAPI
