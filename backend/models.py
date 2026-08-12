"""Pydantic data models for the shopping assistant."""
from __future__ import annotations

from typing import Any, List
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Product catalog models (dummyjson shape)
# ---------------------------------------------------------------------------

class Review(BaseModel):
    """Only non‑PII fields are retained — reviewer name/email are stripped.
    The API schema requires a placeholder reviewer name.
    """
    rating: float
    comment: str
    # Placeholder to satisfy schema while keeping privacy
    reviewer: str = "Verified Buyer"

class Dimensions(BaseModel):
    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0

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
    tags: List[str] = Field(default_factory=list)
    sku: str = ""
    thumbnail: str = ""
    reviews: List[Review] = Field(default_factory=list)
    weight: float = 0.0
    dimensions: Dimensions = Field(default_factory=Dimensions)
    images: List[str] = Field(default_factory=list)
    minimumOrderQuantity: int = 1
    cached_at: str = ""

    @classmethod
    def from_dummyjson(cls, raw: dict[str, Any]) -> "Product":
        """Build a Product, stripping PII from reviews."""
        clean_reviews = [
            Review(
                rating=r.get("rating", 0),
                comment=r.get("comment", ""),
            )
            for r in raw.get("reviews", [])[:2]
        ]
        dims_raw = raw.get("dimensions", {})
        dimensions = Dimensions(
            width=dims_raw.get("width", 0.0),
            height=dims_raw.get("height", 0.0),
            depth=dims_raw.get("depth", 0.0),
        )
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
            weight=raw.get("weight", 0.0),
            dimensions=dimensions,
            images=raw.get("images", []),
            minimumOrderQuantity=raw.get("minimumOrderQuantity", 1),
        )

    def to_schema_dict(self) -> dict[str, Any]:
        """Serialize to the exact API schema required by the frontend.
        Converts camelCase fields to snake_case, casts types, and adds derived values.
        """
        return {
            "id": str(self.id),
            "name": self.title,
            "brand": self.brand,
            "sku": self.sku,
            "category": self.category,
            "tags": self.tags,
            "price": self.price,
            "discount_percentage": self.discountPercentage,
            "rating": self.rating,
            "review_count": len(self.reviews),
            "top_reviews": [
                {"rating": rev.rating, "comment": rev.comment, "reviewer": rev.reviewer}
                for rev in self.reviews
            ],
            "in_stock": self.stock > 0,
            "stock_count": self.stock,
            "availability_status": self.availabilityStatus,
            "minimum_order_quantity": self.minimumOrderQuantity,
            "warranty": self.warrantyInformation,
            "shipping": self.shippingInformation,
            "return_policy": self.returnPolicy,
            "weight": self.weight,
            "dimensions": {
                "width": self.dimensions.width,
                "height": self.dimensions.height,
                "depth": self.dimensions.depth,
            },
            "description": self.description,
            "image": self.thumbnail,
            "images": self.images,
            "source": "dummyjson",
            "url": None,
        }

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

class SearchResponse(BaseModel):
    query: str
    filters_applied: dict[str, Any]
    total_matches: int
    products: list[dict[str, Any]]
    errors: str | None = None

class ChatRequest(BaseModel):
    message: str
    session_id: str
    context: list[dict[str, str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class RecommendedProduct(BaseModel):
    id: int
    score: float
    rank: int
    reason: str
    tradeoffs: str
    pros_llm: list[str] = Field(default_factory=list)
    cons_llm: list[str] = Field(default_factory=list)

class ComparisonResult(BaseModel):
    recommended: list[RecommendedProduct]
    comparison_summary: str
