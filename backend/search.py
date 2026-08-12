"""Search engine — structured filter + full-text search over the cached catalog.

No external service needed at 194 products; all filtering is local.
"""
from __future__ import annotations

import re
from typing import Any

from backend.catalog import get_all_products
from backend.models import Product


def _score_text_match(product: Product, query: str) -> float:
    """Return a simple relevance score (0–1) for full-text matching.

    Matches against title, description, category, brand, and tags.
    """
    if not query:
        return 1.0

    q = query.lower()
    terms = re.split(r"\s+", q.strip())

    fields = [
        product.title.lower(),
        product.description.lower(),
        product.category.lower(),
        product.brand.lower(),
        " ".join(product.tags).lower(),
    ]
    combined = " ".join(fields)

    matches = sum(1 for term in terms if term in combined)
    return matches / len(terms) if terms else 1.0


def search(
    query: str = "",
    category: str = "",
    price_min: float = 0.0,
    price_max: float = 1_000_000.0,
    min_rating: float = 0.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Filter the cached catalog and return the top-K products as dicts.

    Scoring: products that match all filter criteria are sorted by
    (text_relevance * 0.5) + (normalised_rating * 0.3) + (discount_bonus * 0.2)
    so the most relevant AND well-rated items surface first.
    """
    products = get_all_products()

    # 1. Hard filters
    filtered: list[tuple[Product, float]] = []
    for p in products:
        if category and p.category != category:
            continue
        if p.price < price_min or p.price > price_max:
            continue
        if p.rating < min_rating:
            continue

        text_score = _score_text_match(p, query)
        # Only include products with some text relevance (or no query)
        if query and text_score == 0.0:
            continue

        # Composite score
        rating_norm = p.rating / 5.0
        discount_norm = min(p.discountPercentage / 50.0, 1.0)  # cap at 50% discount
        composite = (text_score * 0.5) + (rating_norm * 0.3) + (discount_norm * 0.2)

        filtered.append((p, composite))

    # 2. Sort by composite score descending
    filtered.sort(key=lambda x: x[1], reverse=True)

    # 3. Return top-K as plain dicts (for JSON serialisation)
    return [
        {
            **p.model_dump(exclude={"reviews"}),
            "reviews": [r.model_dump() for r in p.reviews],
            "_score": round(score, 4),
        }
        for p, score in filtered[:limit]
    ]
