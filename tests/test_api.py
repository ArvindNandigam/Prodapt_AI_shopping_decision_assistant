"""Integration tests for the FastAPI endpoints."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from backend.main import app
from backend.models import ComparisonResult, RecommendedProduct


FAKE_PRODUCTS_DICT = [
    {
        "id": 1,
        "title": "Nike Running Shoe",
        "description": "Great shoe",
        "category": "sports-accessories",
        "brand": "Nike",
        "price": 89.99,
        "discountPercentage": 10.0,
        "rating": 4.5,
        "stock": 30,
        "availabilityStatus": "In Stock",
        "warrantyInformation": "1 year",
        "returnPolicy": "30 days",
        "tags": ["sports"],
        "sku": "NK001",
        "thumbnail": "",
        "reviews": [],
        "_score": 0.9,
    }
]

FAKE_COMPARISON = ComparisonResult(
    recommended=[
        RecommendedProduct(
            id=1,
            score=0.9,
            rank=1,
            reason="Best value",
            tradeoffs="Slightly pricier",
            pros_llm=["comfortable"],
            cons_llm=["limited colors"],
        )
    ],
    comparison_summary="Nike is the top pick for running.",
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestCategoriesEndpoint:
    def test_returns_categories_list(self, client):
        with patch("backend.main.catalog.get_categories", return_value=["smartphones", "laptops"]):
            resp = client.get("/api/categories")
        assert resp.status_code == 200
        assert "categories" in resp.json()


class TestSearchEndpoint:
    def test_basic_search(self, client):
        with (
            patch("backend.main.search", return_value=FAKE_PRODUCTS_DICT),
            patch("backend.main.compare", return_value=FAKE_COMPARISON),
            patch("backend.main.catalog.get_categories", return_value=["sports-accessories"]),
        ):
            resp = client.post("/api/search", json={
                "query": "running shoes",
                "category": "sports-accessories",
                "price_min": 0,
                "price_max": 200,
                "min_rating": 4.0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "summary" in data

    def test_invalid_price_range_returns_400(self, client):
        resp = client.post("/api/search", json={
            "price_min": 200,
            "price_max": 100,  # min > max
        })
        assert resp.status_code == 400

    def test_invalid_rating_returns_400(self, client):
        resp = client.post("/api/search", json={
            "min_rating": 6.0,  # > 5.0
        })
        assert resp.status_code == 400

    def test_unknown_category_returns_400(self, client):
        with patch("backend.main.catalog.get_categories", return_value=["smartphones"]):
            resp = client.post("/api/search", json={
                "category": "nonexistent-category",
            })
        assert resp.status_code == 400

    def test_empty_results_graceful(self, client):
        with (
            patch("backend.main.search", return_value=[]),
            patch("backend.main.catalog.get_categories", return_value=[]),
        ):
            resp = client.post("/api/search", json={"query": "nothing exists"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []


class TestChatEndpoint:
    def test_basic_chat(self, client):
        with patch("backend.main.chat", return_value=("Great choice!", "session-123")):
            resp = client.post("/api/chat", json={
                "message": "Which is better?",
                "session_id": "session-123",
                "context": [],
            })
        assert resp.status_code == 200
        assert "reply" in resp.json()

    def test_empty_message_returns_400(self, client):
        resp = client.post("/api/chat", json={
            "message": "   ",
            "session_id": "session-123",
        })
        assert resp.status_code == 400
