"""Unit tests for the catalog loader — PII stripping and data loading."""
from __future__ import annotations

import pytest
from backend.models import Product


SAMPLE_RAW_PRODUCT = {
    "id": 1,
    "title": "Test Shoe",
    "description": "A test product",
    "category": "sports-accessories",
    "brand": "TestBrand",
    "price": 79.99,
    "discountPercentage": 5.0,
    "rating": 4.3,
    "stock": 10,
    "availabilityStatus": "In Stock",
    "warrantyInformation": "1 year warranty",
    "shippingInformation": "Ships in 1-2 days",
    "returnPolicy": "30 days return policy",
    "tags": ["sports", "shoes"],
    "sku": "TS001",
    "thumbnail": "https://example.com/img.jpg",
    "reviews": [
        {
            "rating": 5,
            "comment": "Great product!",
            "reviewerName": "John Doe",          # PII — must be stripped
            "reviewerEmail": "john@example.com", # PII — must be stripped
        },
        {
            "rating": 4,
            "comment": "Pretty good.",
            "reviewerName": "Jane Smith",
            "reviewerEmail": "jane@example.com",
        },
        {
            "rating": 3,
            "comment": "Third review — should be excluded.",
            "reviewerName": "Bob",
            "reviewerEmail": "bob@example.com",
        },
    ],
}


class TestProductFromDummyjson:
    def test_pii_stripped_from_reviews(self):
        """reviewerName and reviewerEmail must never appear in the parsed product."""
        product = Product.from_dummyjson(SAMPLE_RAW_PRODUCT)
        product_dict = product.model_dump()
        product_json = str(product_dict)

        assert "John Doe" not in product_json
        assert "john@example.com" not in product_json
        assert "Jane Smith" not in product_json
        assert "jane@example.com" not in product_json

    def test_review_content_retained(self):
        """Review rating and comment (non-PII) should be kept."""
        product = Product.from_dummyjson(SAMPLE_RAW_PRODUCT)
        assert len(product.reviews) == 2  # capped at 2
        assert product.reviews[0].comment == "Great product!"
        assert product.reviews[0].rating == 5

    def test_max_two_reviews(self):
        """Only the first 2 reviews are kept."""
        product = Product.from_dummyjson(SAMPLE_RAW_PRODUCT)
        assert len(product.reviews) <= 2

    def test_core_fields_mapped(self):
        product = Product.from_dummyjson(SAMPLE_RAW_PRODUCT)
        assert product.id == 1
        assert product.title == "Test Shoe"
        assert product.price == 79.99
        assert product.rating == 4.3
        assert product.brand == "TestBrand"
        assert product.category == "sports-accessories"

    def test_missing_optional_fields_default(self):
        minimal = {"id": 99, "title": "Minimal", "price": 9.99}
        product = Product.from_dummyjson(minimal)
        assert product.id == 99
        assert product.rating == 0.0
        assert product.reviews == []
        assert product.tags == []
