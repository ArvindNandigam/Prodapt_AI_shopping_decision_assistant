"""Unit tests for the search engine."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from backend.models import Product
from backend.search import search, _score_text_match


FAKE_PRODUCTS = [
    Product(
        id=1,
        title="Nike Running Shoes",
        description="Great cushioning for runners",
        category="sports-accessories",
        brand="Nike",
        price=89.99,
        discountPercentage=10.0,
        rating=4.5,
        stock=30,
        availabilityStatus="In Stock",
        returnPolicy="30 days",
    ),
    Product(
        id=2,
        title="Adidas Training Shoes",
        description="Durable gym shoes",
        category="sports-accessories",
        brand="Adidas",
        price=149.99,
        discountPercentage=5.0,
        rating=4.2,
        stock=5,
        availabilityStatus="Low Stock",
        returnPolicy="14 days",
    ),
    Product(
        id=3,
        title="Samsung Galaxy S24",
        description="Flagship smartphone",
        category="smartphones",
        brand="Samsung",
        price=899.99,
        discountPercentage=8.0,
        rating=4.7,
        stock=50,
        availabilityStatus="In Stock",
        returnPolicy="30 days",
    ),
]


@pytest.fixture(autouse=True)
def mock_catalog():
    with patch("backend.search.get_all_products", return_value=FAKE_PRODUCTS):
        yield


class TestSearch:
    def test_no_filters_returns_all(self):
        results = search()
        assert len(results) == 3

    def test_category_filter(self):
        results = search(category="smartphones")
        assert len(results) == 1
        assert results[0]["title"] == "Samsung Galaxy S24"

    def test_price_max_filter(self):
        results = search(price_max=100.0)
        # Only Nike shoes at $89.99 is within range
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_price_range_filter(self):
        results = search(price_min=100.0, price_max=200.0)
        assert len(results) == 1
        assert results[0]["id"] == 2

    def test_min_rating_filter(self):
        results = search(min_rating=4.5)
        ids = {r["id"] for r in results}
        assert 1 in ids   # Nike 4.5
        assert 3 in ids   # Samsung 4.7
        assert 2 not in ids  # Adidas 4.2

    def test_text_query_filter(self):
        results = search(query="running")
        assert results[0]["id"] == 1  # Nike matches "running"

    def test_text_query_no_match(self):
        results = search(query="zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_limit(self):
        results = search(limit=2)
        assert len(results) <= 2

    def test_unknown_category_returns_empty(self):
        results = search(category="unknown-category")
        assert results == []

    def test_results_contain_required_fields(self):
        results = search()
        for r in results:
            assert "id" in r
            assert "title" in r
            assert "price" in r
            assert "rating" in r
            assert "_score" in r


class TestTextMatch:
    def test_full_match(self):
        p = FAKE_PRODUCTS[0]
        score = _score_text_match(p, "nike running")
        assert score == 1.0

    def test_partial_match(self):
        p = FAKE_PRODUCTS[0]
        score = _score_text_match(p, "nike blah")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        p = FAKE_PRODUCTS[0]
        score = _score_text_match(p, "zzz_nope")
        assert score == 0.0

    def test_empty_query(self):
        p = FAKE_PRODUCTS[0]
        score = _score_text_match(p, "")
        assert score == 1.0
