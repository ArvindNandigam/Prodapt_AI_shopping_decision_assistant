"""LLM Comparator Skill — LangChain v1 + Azure OpenAI with deterministic pre-scoring.

Architecture:
  - Deterministic scoring (relevance/quality/value/reviews/shipping/warranty/returns/stock)
  - LangChain prompt + AzureChatOpenAI + JsonOutputParser for explanation/ranking refinement
  - Strict JSON schema expected from LLM
  - Graceful fallback to deterministic ranking when LLM fails
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import AzureChatOpenAI

from backend.config import settings
from backend.models import ComparisonResult, RecommendedProduct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic ranking weights (sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "relevance": 0.25,
    "quality": 0.20,
    "value": 0.15,
    "review_signal": 0.10,
    "shipping": 0.10,
    "warranty": 0.08,
    "return_policy": 0.08,
    "stock_reliability": 0.04,
}

_POS_REVIEW_WORDS = {
    "great",
    "comfortable",
    "durable",
    "lightweight",
    "good",
    "excellent",
    "supportive",
    "breathable",
    "soft",
    "fit",
}
_NEG_REVIEW_WORDS = {
    "poor",
    "bad",
    "tight",
    "heavy",
    "broke",
    "uncomfortable",
    "slow",
    "hard",
    "small",
    "large",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a product comparison expert.

Your job:
1) Evaluate candidate products against user intent and constraints.
2) Use the provided deterministic scores as the baseline ranking.
3) Return clear pros/cons, tradeoffs, and a concise summary.

Critical rules:
- Base every claim only on provided data; never invent specs or policies.
- Respect final_score and score_breakdown as primary ranking signals.
- You may make only SMALL ranking adjustments for intent relevance or tie-breakers.
- Do not invert ranking significantly without explicit justification in reason.
- If two products are close, say so and explain the tie-breaker.
- Flag any product that violates hard constraints (if present).
- Output strictly valid JSON matching the exact schema.
- comparison_summary: 2-3 sentences.
- reason/tradeoffs: under 40 words each.
- No text outside the JSON object."""

HUMAN_PROMPT = """User intent: "{user_query}"

Hard constraints:
  - Category: {category}
  - Price range: ${price_min} – ${price_max}
  - Minimum rating: {min_rating}

Candidates (JSON array):
{candidates_json}

Respond with ONLY this JSON structure (no markdown, no extra keys):
{{
  "recommended": [
    {{
      "id": <int or string>,
      "score": <float 0.0-1.0>,
      "rank": <int starting at 1>,
      "reason": "<under 40 words>",
      "tradeoffs": "<under 40 words>",
      "pros_llm": ["<string>", ...],
      "cons_llm": ["<string>", ...]
    }}
  ],
  "comparison_summary": "<2-3 sentences>"
}}"""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:  # noqa: BLE001
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:  # noqa: BLE001
        return default


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _get_first(p: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    return default


def _parse_days(text: str) -> int | None:
    # e.g., "Ships in 3 days", "delivery in 1 day"
    m = re.search(r"(\d+)\s*day", (text or "").lower())
    return int(m.group(1)) if m else None


def _parse_months(text: str) -> int | None:
    # e.g., "1 month warranty", "12 months warranty", "2 year warranty"
    t = (text or "").lower()
    m = re.search(r"(\d+)\s*month", t)
    if m:
        return int(m.group(1))
    y = re.search(r"(\d+)\s*year", t)
    if y:
        return int(y.group(1)) * 12
    return None


def _parse_return_days(text: str) -> int | None:
    # e.g., "30 days return policy"
    m = re.search(r"(\d+)\s*day", (text or "").lower())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Deterministic component scorers (0..100)
# ---------------------------------------------------------------------------


def _review_signal_score(reviews: list[dict[str, Any]]) -> float:
    if not reviews:
        return 50.0

    ratings = [_safe_float(r.get("rating"), 0.0) for r in reviews]
    avg_rating = (sum(ratings) / len(ratings)) if ratings else 0.0
    base = (avg_rating / 5.0) * 70.0  # max 70

    pos = 0
    neg = 0
    for r in reviews:
        words = _tokenize(str(r.get("comment") or ""))
        pos += len(words & _POS_REVIEW_WORDS)
        neg += len(words & _NEG_REVIEW_WORDS)

    sentiment = _clamp(30 + (pos - neg) * 8, 0, 30)  # 0..30
    return _clamp(base + sentiment)


def _shipping_score(shipping_text: str) -> float:
    days = _parse_days(shipping_text or "")
    if days is None:
        return 60.0
    if days <= 1:
        return 95.0
    if days <= 3:
        return 75.0
    if days <= 5:
        return 60.0
    return 40.0


def _warranty_score(warranty_text: str) -> float:
    months = _parse_months(warranty_text or "")
    if months is None:
        return 50.0
    if months >= 12:
        return 95.0
    if months >= 6:
        return 75.0
    if months >= 3:
        return 60.0
    return 35.0


def _return_policy_score(return_text: str) -> float:
    days = _parse_return_days(return_text or "")
    if days is None:
        return 50.0
    if days >= 30:
        return 95.0
    if days >= 14:
        return 75.0
    if days >= 7:
        return 60.0
    return 35.0


def _relevance_score(product: dict[str, Any], user_query: str) -> float:
    q = _tokenize(user_query)
    if not q:
        return 70.0

    title = _get_first(product, ["title", "name"], "")
    brand = _get_first(product, ["brand"], "")
    category = _get_first(product, ["category"], "")
    tags = product.get("tags") or []
    tags_text = " ".join(t for t in tags if isinstance(t, str))

    hay = _tokenize(f"{title} {brand} {category} {tags_text}")
    overlap = len(q & hay)
    return _clamp((overlap / max(len(q), 1)) * 100.0)


def _quality_score(rating: float, review_count: int) -> float:
    rating_part = _clamp((rating / 5.0) * 100.0)
    confidence = _clamp((math.log1p(max(review_count, 0)) / math.log1p(1000)) * 100.0)
    return _clamp(0.75 * rating_part + 0.25 * confidence)


def _value_score(price: float, discount_pct: float, min_price: float, max_price: float) -> float:
    if max_price <= min_price:
        price_pos = 50.0
    else:
        # Lower price in range gets higher value score
        price_pos = _clamp((1.0 - ((price - min_price) / (max_price - min_price))) * 100.0)

    discount_boost = _clamp(discount_pct * 2.0, 0.0, 20.0)
    return _clamp(price_pos * 0.85 + discount_boost)


def _stock_score(in_stock: bool, stock_count: int) -> float:
    if not in_stock:
        return 10.0
    if stock_count >= 50:
        return 95.0
    if stock_count >= 20:
        return 80.0
    if stock_count >= 5:
        return 60.0
    return 40.0


# ---------------------------------------------------------------------------
# Product scoring + enrichment
# ---------------------------------------------------------------------------


def score_product(
    p: dict[str, Any],
    user_query: str,
    price_min: float,
    price_max: float,
) -> dict[str, Any]:
    rating = _safe_float(_get_first(p, ["rating"], 0.0))
    review_count = _safe_int(_get_first(p, ["review_count", "reviewCount"], 0))
    price = _safe_float(_get_first(p, ["price"], 0.0))
    discount = _safe_float(_get_first(p, ["discount_percentage", "discountPercentage"], 0.0))

    reviews = _get_first(p, ["top_reviews", "reviews"], []) or []
    shipping_text = str(_get_first(p, ["shipping", "shippingInformation"], "") or "")
    warranty_text = str(_get_first(p, ["warranty", "warrantyInformation"], "") or "")
    return_text = str(_get_first(p, ["return_policy", "returnPolicy"], "") or "")

    in_stock = bool(_get_first(p, ["in_stock"], _safe_int(_get_first(p, ["stock"], 0)) > 0))
    stock_count = _safe_int(_get_first(p, ["stock_count", "stock"], 0))

    breakdown = {
        "relevance": _relevance_score(p, user_query),
        "quality": _quality_score(rating, review_count),
        "value": _value_score(price, discount, price_min, price_max),
        "review_signal": _review_signal_score(reviews),
        "shipping": _shipping_score(shipping_text),
        "warranty": _warranty_score(warranty_text),
        "return_policy": _return_policy_score(return_text),
        "stock_reliability": _stock_score(in_stock, stock_count),
    }

    final_score_100 = sum(breakdown[k] * _WEIGHTS[k] for k in _WEIGHTS)
    return {
        "final_score": round(final_score_100 / 100.0, 4),  # 0..1 scale
        "score_breakdown": {k: round(v, 2) for k, v in breakdown.items()},
    }


def score_candidates(
    candidates: list[dict[str, Any]],
    user_query: str,
    price_min: float,
    price_max: float,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for p in candidates:
        cp = dict(p)
        cp.update(score_product(cp, user_query, price_min, price_max))
        enriched.append(cp)

    return sorted(enriched, key=lambda x: x.get("final_score", 0.0), reverse=True)


# ---------------------------------------------------------------------------
# LangChain chain construction
# ---------------------------------------------------------------------------


def _build_chain():
    llm = AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0.2,
        max_tokens=2048,
        timeout=30,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ]
    )

    parser = JsonOutputParser()
    return prompt | llm | parser


_chain = None


def _get_chain():
    global _chain  # noqa: PLW0603
    if _chain is None:
        _chain = _build_chain()
    return _chain


# ---------------------------------------------------------------------------
# Candidate trimming (send only needed fields)
# ---------------------------------------------------------------------------


def _trim_candidate(p: dict[str, Any]) -> dict[str, Any]:
    product_id = _get_first(p, ["id"])
    title = _get_first(p, ["title", "name"], "")
    return {
        "id": product_id,
        "title": title,
        "brand": _get_first(p, ["brand"], ""),
        "category": _get_first(p, ["category"], ""),
        "tags": p.get("tags") or [],
        "price": _get_first(p, ["price"]),
        "discount_percentage": _get_first(p, ["discount_percentage", "discountPercentage"], 0.0),
        "rating": _get_first(p, ["rating"]),
        "review_count": _get_first(p, ["review_count", "reviewCount"], 0),
        "shipping": _get_first(p, ["shipping", "shippingInformation"], ""),
        "warranty": _get_first(p, ["warranty", "warrantyInformation"], ""),
        "return_policy": _get_first(p, ["return_policy", "returnPolicy"], ""),
        "in_stock": _get_first(p, ["in_stock"], _safe_int(_get_first(p, ["stock"], 0)) > 0),
        "stock_count": _get_first(p, ["stock_count", "stock"], 0),
        "reviews": [
            {"rating": r.get("rating"), "comment": r.get("comment")}
            for r in (_get_first(p, ["top_reviews", "reviews"], []) or [])[:3]
        ],
        "final_score": p.get("final_score"),
        "score_breakdown": p.get("score_breakdown"),
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def compare(
    candidates: list[dict[str, Any]],
    user_query: str,
    category: str = "",
    price_min: float = 0.0,
    price_max: float = 1_000_000.0,
    min_rating: float = 0.0,
) -> ComparisonResult:
    """Rank and explain candidates using deterministic + LLM comparator skill."""
    if not candidates:
        return ComparisonResult(
            recommended=[],
            comparison_summary="No candidates to compare.",
        )

    # Deterministic baseline ranking first
    scored_candidates = score_candidates(
        candidates=candidates,
        user_query=user_query or "best product",
        price_min=price_min,
        price_max=price_max,
    )

    # If LLM disabled, return deterministic fallback immediately
    if not settings.use_llm:
        return _fallback_result(scored_candidates)

    trimmed = [_trim_candidate(p) for p in scored_candidates]
    candidates_json = json.dumps(trimmed, indent=2)

    try:
        chain = _get_chain()
        raw = chain.invoke(
            {
                "user_query": user_query or "best product",
                "category": category or "any",
                "price_min": price_min,
                "price_max": price_max,
                "min_rating": min_rating,
                "candidates_json": candidates_json,
            }
        )

        return _parse_llm_output(raw, scored_candidates)

    except OutputParserException as exc:
        logger.warning("LLM output parse failed: %s — using deterministic fallback", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM comparator error: %s — using deterministic fallback", exc)

    return _fallback_result(scored_candidates)


# ---------------------------------------------------------------------------
# Parsing & fallback
# ---------------------------------------------------------------------------


def _fallback_result(scored_candidates: list[dict[str, Any]]) -> ComparisonResult:
    """Deterministic fallback: rank by final_score and provide simple reasons."""
    recommended: list[RecommendedProduct] = []
    for i, p in enumerate(scored_candidates):
        score = _safe_float(p.get("final_score"), 0.0)
        breakdown = p.get("score_breakdown", {}) or {}
        reason = (
            f"Strong overall fit (relevance {breakdown.get('relevance', 'N/A')}, "
            f"quality {breakdown.get('quality', 'N/A')}, value {breakdown.get('value', 'N/A')})."
        )
        tradeoffs = (
            "Shipping/warranty/returns may vary; review product policy details before purchase."
        )
        recommended.append(
            RecommendedProduct(
                id=p["id"],
                score=round(score, 4),
                rank=i + 1,
                reason=reason[:120],
                tradeoffs=tradeoffs[:120],
                pros_llm=[],
                cons_llm=[],
            )
        )

    return ComparisonResult(
        recommended=recommended,
        comparison_summary=(
            "AI comparison is unavailable, so results use deterministic scoring across "
            "relevance, quality, value, reviews, shipping, warranty, return policy, and stock."
        ),
    )


def _parse_llm_output(raw: Any, candidates: list[dict[str, Any]]) -> ComparisonResult:
    """Parse + validate LLM JSON output into ComparisonResult."""
    if not isinstance(raw, dict):
        raise OutputParserException(f"Expected dict, got {type(raw)}")

    candidate_ids = {p["id"] for p in candidates}
    recommended_raw = raw.get("recommended", [])
    summary = str(raw.get("comparison_summary", "") or "").strip()

    recommended: list[RecommendedProduct] = []
    for i, item in enumerate(recommended_raw):
        pid = item.get("id")
        if pid not in candidate_ids:
            logger.warning("LLM returned unknown product id %s — skipping", pid)
            continue

        score = _safe_float(item.get("score"), 0.0)
        score = max(0.0, min(1.0, score))

        rank = _safe_int(item.get("rank"), i + 1)
        if rank < 1:
            rank = i + 1

        reason = str(item.get("reason", "") or "").strip()
        tradeoffs = str(item.get("tradeoffs", "") or "").strip()
        pros = item.get("pros_llm", []) or []
        cons = item.get("cons_llm", []) or []

        if not isinstance(pros, list):
            pros = []
        if not isinstance(cons, list):
            cons = []

        recommended.append(
            RecommendedProduct(
                id=pid,
                score=round(score, 4),
                rank=rank,
                reason=reason,
                tradeoffs=tradeoffs,
                pros_llm=[str(x) for x in pros][:5],
                cons_llm=[str(x) for x in cons][:5],
            )
        )

    if not recommended:
        raise OutputParserException("LLM returned no valid recommended products")

    # Ensure deterministic ordering by rank, then score desc for ties
    recommended.sort(key=lambda x: (x.rank, -x.score))

    if not summary:
        summary = "Compared candidates by fit, quality, value, and purchase-risk factors."

    return ComparisonResult(recommended=recommended, comparison_summary=summary)
