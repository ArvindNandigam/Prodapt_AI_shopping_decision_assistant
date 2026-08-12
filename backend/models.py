"""Pydantic data models for the shopping assistant."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Product catalog models (dummyjson shape)
# ---------------------------------------------------------------------------

class Review(BaseModel):
    """Only non-PII fields are retained — reviewer name/email are stripped."""
    rating: float
    comment: str


class Product(BaseModel):
    id: int
    title: str
    description: str = ""
    category: str = ""
    brand: str = ""
    price: float
    discountPercentage: float = 0.0
    rating: float = 0.0
    stock: int = 0
    availabilityStatus: str = ""
    warrantyInformation: str = ""
    shippingInformation: str = ""
    returnPolicy: str = ""
    tags: list[str] = Field(default_factory=list)
    sku: str = ""
    thumbnail: str = ""
    reviews: list[Review] = Field(default_factory=list)
    cached_at: str = ""

    @classmethod
    def from_dummyjson(cls, raw: dict[str, Any]) -> "Product":
        """Build a Product, stripping PII from reviews."""
        clean_reviews = [
            Review(
                rating=r.get("rating", 0),
                comment=r.get("comment", ""),
            )
            for r in raw.get("reviews", [])[:2]  # cap at 2 reviews per prompt
        ]
        return cls(
            id=raw["id"],
            title=raw.get("title", ""),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            brand=raw.get("brand", ""),
            price=raw.get("price", 0.0),
            discountPercentage=raw.get("discountPercentage", 0.0),
            rating=raw.get("rating", 0.0),
            stock=raw.get("stock", 0),
            availabilityStatus=raw.get("availabilityStatus", ""),
            warrantyInformation=raw.get("warrantyInformation", ""),
            shippingInformation=raw.get("shippingInformation", ""),
            returnPolicy=raw.get("returnPolicy", ""),
            tags=raw.get("tags", []),
            sku=raw.get("sku", ""),
            thumbnail=raw.get("thumbnail", ""),
            reviews=clean_reviews,
        )


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = ""
    category: str = ""
    price_min: float = 0.0
    price_max: float = 1_000_000.0
    min_rating: float = 0.0
    session_id: str = ""
    limit: int = 10


class RecommendedProduct(BaseModel):
    id: int
    score: float
    rank: int
    reason: str
    tradeoffs: str
    why_not: str = ""
    pros_llm: list[str] = Field(default_factory=list)
    cons_llm: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    summary: str
    applied_filters: dict[str, Any]
    errors: str | None = None


class ChatRequest(BaseModel):
    message: str = ""
    session_id: str = ""
    context: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ComparisonResult(BaseModel):
    recommended: list[RecommendedProduct]
    comparison_summary: str
