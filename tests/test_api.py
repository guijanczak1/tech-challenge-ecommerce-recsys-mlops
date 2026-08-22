"""Testes da API de inferência (com modelo dummy injetado)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from recsys.data.catalog import product_name
from recsys.serving.api import create_app


class _DummyModel:
    def recommend(self, user: int, k: int) -> list[int]:
        if user >= 100:
            raise IndexError("index out of range in self")
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
    assert body["user"] == 7
    assert body["k"] == 3
    expected = [{"id": i, **product_name(i)} for i in range(3)]
    assert body["items"] == expected


def test_recommend_requires_user() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend")  # falta 'user'
    assert response.status_code == 422  # validação do FastAPI


def test_recommend_k_zero_returns_empty() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend", params={"user": 1, "k": 0})
    assert response.status_code == 200
    assert response.json() == {"user": 1, "k": 0, "items": []}


def test_recommend_negative_user_is_rejected() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend", params={"user": -1, "k": 5})
    assert response.status_code == 422


def test_recommend_negative_k_is_rejected() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend", params={"user": 1, "k": -1})
    assert response.status_code == 422


def test_recommend_user_out_of_range_returns_404() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend", params={"user": 999999, "k": 5})
    assert response.status_code == 404
    assert "fora do catálogo" in response.json()["detail"]


def test_recommend_items_are_enriched_with_name_and_category() -> None:
    with TestClient(create_app(provider=_DummyModel)) as client:
        response = client.get("/recommend", params={"user": 1, "k": 2})
    items = response.json()["items"]
    assert len(items) == 2
    for item in items:
        assert set(item) == {"id", "name", "category"}
