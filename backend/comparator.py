"""LLM Comparator Skill — LangChain v1 + Azure OpenAI (LLM-only comparison).

Architecture:
  - Uses LangChain ChatPromptTemplate + AzureChatOpenAI + JsonOutputParser
  - Supports optional user_preferences from backend
  - Strict JSON schema expected from LLM
  - Graceful fallback if LLM fails
"""

from __future__ import annotations

import json
import logging
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
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a product comparison expert.

Goal:
Compare candidate products against the user's intent, constraints, and preferences.
Return useful recommendations with clear pros/cons and tradeoffs.

Rules:
- Use only fields provided in input candidates; never invent facts.
- Consider rating, review_count, top_reviews/reviews, shipping, warranty, return_policy,
  availability/in_stock/stock_count, price, and discount when available.
- Respect hard constraints as highest priority. If any candidate appears to violate them, flag it.
- If user_preferences are provided, prioritize preference fit as a tie-breaker after hard constraints.
- If products are close, say so and explain tie-breaker briefly.
- Keep comparison_summary to 2-3 sentences.
- Keep reason/tradeoffs concise (under 40 words each).
- Output strictly valid JSON and nothing else."""

HUMAN_PROMPT = """User intent: "{user_query}"

Hard constraints:
  - Category: {category}
  - Price range: ${price_min} – ${price_max}
  - Minimum rating: {min_rating}

User preferences (optional):
{user_preferences_json}

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
# Candidate trimming — keep only relevant fields
# ---------------------------------------------------------------------------


def _get_first(p: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in p and p[k] is not None:
            return p[k]
    return default


def _trim_candidate(p: dict[str, Any]) -> dict[str, Any]:
    def _clean_reviews(reviews: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for review in reviews or []:
            if not isinstance(review, dict):
                continue
            filtered = {
                key: value
                for key, value in review.items()
                if key not in {"reviewerName", "reviewerEmail", "name", "email", "phone", "address"}
            }
            if filtered:
                cleaned.append(filtered)
        return cleaned[:3]

    title = _get_first(p, ["title", "name"], "")
    return {
        "id": _get_first(p, ["id"]),
        "title": title,
        "name": title,
        "brand": _get_first(p, ["brand"], ""),
        "sku": _get_first(p, ["sku"], None),
        "category": _get_first(p, ["category"], ""),
        "tags": p.get("tags") or [],
        "price": _get_first(p, ["price"], None),
        "discountPercentage": _get_first(p, ["discountPercentage", "discount_percentage"], None),
        "discount_percentage": _get_first(p, ["discountPercentage", "discount_percentage"], None),
        "rating": _get_first(p, ["rating"], None),
        "review_count": _get_first(p, ["review_count", "reviewCount"], 0),
        "top_reviews": _clean_reviews((p.get("top_reviews") or p.get("reviews") or [])[:3]),
        "reviews": _clean_reviews((p.get("top_reviews") or p.get("reviews") or [])[:3]),
        "in_stock": _get_first(p, ["in_stock"], None),
        "stock": _get_first(p, ["stock", "stock_count"], None),
        "stock_count": _get_first(p, ["stock", "stock_count"], None),
        "availabilityStatus": _get_first(p, ["availabilityStatus", "availability_status"], None),
        "availability_status": _get_first(p, ["availabilityStatus", "availability_status"], None),
        "warrantyInformation": _get_first(p, ["warrantyInformation", "warranty"], None),
        "warranty": _get_first(p, ["warrantyInformation", "warranty"], None),
        "shippingInformation": _get_first(p, ["shippingInformation", "shipping"], None),
        "shipping": _get_first(p, ["shippingInformation", "shipping"], None),
        "returnPolicy": _get_first(p, ["returnPolicy", "return_policy"], None),
        "return_policy": _get_first(p, ["returnPolicy", "return_policy"], None),
        "description": _get_first(p, ["description"], None),
        "source": _get_first(p, ["source"], None),
        "url": _get_first(p, ["url"], None),
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
    user_preferences: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Rank and explain candidates using LLM comparator skill (no deterministic ranking)."""
    if not candidates:
        return ComparisonResult(
            recommended=[],
            comparison_summary="No candidates to compare.",
        )

    if not settings.llm_enabled:
        logger.warning(
            "LLM disabled or Azure OpenAI configuration is missing; using fallback comparison."
        )
        return _fallback_result(candidates)

    trimmed = [_trim_candidate(p) for p in candidates]
    candidates_json = json.dumps(trimmed, indent=2)
    user_preferences_json = json.dumps(user_preferences or {}, indent=2)

    try:
        chain = _get_chain()
        raw = chain.invoke(
            {
                "user_query": user_query or "best product",
                "category": category or "any",
                "price_min": price_min,
                "price_max": price_max,
                "min_rating": min_rating,
                "user_preferences_json": user_preferences_json,
                "candidates_json": candidates_json,
            }
        )
        return _parse_llm_output(raw, candidates)

    except OutputParserException as exc:
        logger.warning("LLM output parse failed: %s — using fallback", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM comparator error: %s — using fallback", exc)

    return _fallback_result(candidates)


# ---------------------------------------------------------------------------
# Fallback & parsing helpers
# ---------------------------------------------------------------------------


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


def _fallback_result(candidates: list[dict[str, Any]]) -> ComparisonResult:
    """Graceful non-LLM fallback: preserve input order with neutral explanation."""
    recommended = []
    for i, p in enumerate(candidates):
        rating = _safe_float(p.get("rating"), 0.0)
        score = round(max(0.0, min(1.0, rating / 5.0)), 2)

        recommended.append(
            RecommendedProduct(
                id=p["id"],
                score=score,
                rank=i + 1,
                reason="AI comparison unavailable; product included from filtered candidate set.",
                tradeoffs="Detailed comparison could not be generated at this time.",
                advantages=[],
                disadvantages=[],
                pros_llm=[],
                cons_llm=[],
            )
        )

    return ComparisonResult(
        recommended=recommended,
        comparison_summary="AI comparison is currently unavailable. Showing filtered candidates in input order.",
    )


def _parse_llm_output(raw: Any, candidates: list[dict[str, Any]]) -> ComparisonResult:
    """Parse and validate LLM JSON output into ComparisonResult."""
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

        advantages = item.get("advantages", []) or item.get("pros_llm", []) or []
        disadvantages = item.get("disadvantages", []) or item.get("cons_llm", []) or []
        if not isinstance(advantages, list):
            advantages = []
        if not isinstance(disadvantages, list):
            disadvantages = []

        normalized_advantages = [str(x) for x in advantages][:5]
        normalized_disadvantages = [str(x) for x in disadvantages][:5]

        recommended.append(
            RecommendedProduct(
                id=pid,
                score=round(score, 4),
                rank=rank,
                reason=reason,
                tradeoffs=tradeoffs,
                advantages=normalized_advantages,
                disadvantages=normalized_disadvantages,
                pros_llm=normalized_advantages,
                cons_llm=normalized_disadvantages,
            )
        )

    if not recommended:
        raise OutputParserException("LLM returned no valid recommended products")

    # Stable ordering by rank, then score desc
    recommended.sort(key=lambda x: (x.rank, -x.score))

    if not summary:
        summary = "Compared products using intent, constraints, preferences, and available product metadata."

    return ComparisonResult(recommended=recommended, comparison_summary=summary)
