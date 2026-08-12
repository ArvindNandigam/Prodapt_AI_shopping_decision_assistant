"""LLM Comparator Skill — uses LangChain v1 + Azure OpenAI to rank products.

Architecture:
  - Uses LangChain's ChatPromptTemplate + AzureChatOpenAI + JsonOutputParser
  - Strict JSON schema enforced; invalid output triggers graceful fallback
  - System prompt is static (cacheable); user prompt is dynamic per request
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from backend.config import settings
from backend.models import ComparisonResult, RecommendedProduct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a product comparison expert. Your job is to evaluate a list of candidate \
products against the user's stated intent and constraints, rank them, and provide \
clear, honest pros/cons with a final recommendation.

Rules:
- Base every claim on data provided; do not invent specs.
- If two products are close, say so and explain the tiebreaker.
- Flag any product that violates a hard constraint (price, min rating) — even if \
  it appeared in the candidate list.
- Weigh stock/availability and return policy alongside price and rating — a \
  cheaper item that is low-stock or has a short return window is a real tradeoff, \
  not a strictly better pick.
- Output strictly valid JSON matching the provided schema.
- Keep the comparison_summary to 2-3 sentences; keep each reason under 40 words.
- Do not include any text outside the JSON object."""

HUMAN_PROMPT = """User intent: "{user_query}"

Hard constraints:
  - Category: {category}
  - Price range: ${price_min} – ${price_max}
  - Minimum rating: {min_rating}

Candidates (JSON array — fields sourced from dummyjson):
{candidates_json}

Respond with ONLY this JSON structure (no markdown, no extra keys):
{{
  "recommended": [
    {{
      "id": <int>,
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
    """Build and return the LangChain LCEL chain."""
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

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
    ])

    parser = JsonOutputParser()

    return prompt | llm | parser


_chain = None  # Lazy-initialised to avoid import-time side effects


def _get_chain():
    global _chain  # noqa: PLW0603
    if _chain is None:
        _chain = _build_chain()
    return _chain


# ---------------------------------------------------------------------------
# Candidate trimming — only send fields the LLM needs, strip PII
# ---------------------------------------------------------------------------

def _trim_candidate(p: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal projection of a product dict for the LLM prompt."""
    return {
        "id": p.get("id"),
        "title": p.get("title"),
        "brand": p.get("brand"),
        "price": p.get("price"),
        "discountPercentage": p.get("discountPercentage"),
        "rating": p.get("rating"),
        "stock": p.get("stock"),
        "availabilityStatus": p.get("availabilityStatus"),
        "warrantyInformation": p.get("warrantyInformation"),
        "returnPolicy": p.get("returnPolicy"),
        "reviews": [
            # PII already stripped in catalog loader; belt-and-suspenders here
            {"rating": r.get("rating"), "comment": r.get("comment")}
            for r in (p.get("reviews") or [])[:2]
        ],
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
    """Rank and explain candidates using the LLM comparator skill.

    Returns a ComparisonResult. On any LLM failure, falls back to a
    rating-sorted result with a generic summary.
    """
    if not candidates:
        return ComparisonResult(recommended=[], comparison_summary="No candidates to compare.")

    if not settings.use_llm:
        return _fallback_result(candidates)

    trimmed = [_trim_candidate(p) for p in candidates]
    candidates_json = json.dumps(trimmed, indent=2)

    try:
        chain = _get_chain()
        raw = chain.invoke({
            "user_query": user_query or "best product",
            "category": category or "any",
            "price_min": price_min,
            "price_max": price_max,
            "min_rating": min_rating,
            "candidates_json": candidates_json,
        })

        # Validate and coerce to ComparisonResult
        return _parse_llm_output(raw, candidates)

    except OutputParserException as exc:
        logger.warning("LLM output parse failed: %s — using fallback", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM comparator error: %s — using fallback", exc)

    return _fallback_result(candidates)


# ---------------------------------------------------------------------------
# Fallback & parsing helpers
# ---------------------------------------------------------------------------

def _fallback_result(candidates: list[dict[str, Any]]) -> ComparisonResult:
    """Return candidates sorted by rating as a graceful fallback."""
    sorted_candidates = sorted(candidates, key=lambda p: p.get("rating", 0), reverse=True)
    recommended = [
        RecommendedProduct(
            id=p["id"],
            score=round(p.get("rating", 0) / 5.0, 2),
            rank=i + 1,
            reason=f"Ranked by rating ({p.get('rating', 'N/A')})",
            tradeoffs="AI comparison unavailable — showing top-rated products.",
            pros_llm=[],
            cons_llm=[],
        )
        for i, p in enumerate(sorted_candidates)
    ]
    return ComparisonResult(
        recommended=recommended,
        comparison_summary="AI comparison is currently unavailable. Results are sorted by rating.",
    )


def _parse_llm_output(raw: Any, candidates: list[dict[str, Any]]) -> ComparisonResult:
    """Parse the LLM JSON output into a ComparisonResult, with validation."""
    if not isinstance(raw, dict):
        raise OutputParserException(f"Expected dict, got {type(raw)}")

    candidate_ids = {p["id"] for p in candidates}
    recommended_raw = raw.get("recommended", [])
    summary = raw.get("comparison_summary", "")

    recommended = []
    for i, item in enumerate(recommended_raw):
        pid = item.get("id")
        if pid not in candidate_ids:
            logger.warning("LLM returned unknown product id %s — skipping", pid)
            continue
        recommended.append(
            RecommendedProduct(
                id=pid,
                score=float(item.get("score", 0.5)),
                rank=int(item.get("rank", i + 1)),
                reason=str(item.get("reason", "")),
                tradeoffs=str(item.get("tradeoffs", "")),
                pros_llm=list(item.get("pros_llm", [])),
                cons_llm=list(item.get("cons_llm", [])),
            )
        )

    if not recommended:
        raise OutputParserException("LLM returned no valid recommended products")

    return ComparisonResult(recommended=recommended, comparison_summary=summary)
