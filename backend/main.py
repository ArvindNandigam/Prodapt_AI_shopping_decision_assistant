"""FastAPI application — main entry point.

Endpoints:
  GET  /health              — liveness check
  GET  /api/categories      — list available product categories
  POST /api/search          — search + AI compare
  POST /api/chat            — conversational follow-up
  POST /api/catalog/refresh — manually trigger catalog re-sync
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import catalog
from backend.comparator import compare
from backend.chat import chat, create_session
from backend.models import SearchRequest, SearchResponse, ChatRequest, ChatResponse
from backend.search import search
from backend.security import (
    MAX_TOOL_CALLS,
    audit_logger,
    sanitize_input,
    session_store,
    validate_search_request,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — sync catalog on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — syncing product catalog from dummyjson…")
    catalog.sync_catalog()
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Shopping Decision Assistant",
    version="0.1.0",
    description="Helps users compare products with AI-powered ranking and rationale.",
    lifespan=lifespan,
)

_rate_limit_store: dict[str, tuple[int, float]] = {}


def _check_rate_limit(client_id: str) -> None:
    now = __import__("time").time()
    window = 60.0
    limit = 60
    current = _rate_limit_store.get(client_id, (0, now))
    requests, first_seen = current
    if now - first_seen > window:
        requests = 0
        first_seen = now
    requests += 1
    _rate_limit_store[client_id] = (requests, first_seen)
    if requests > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    products = catalog.get_all_products()
    return {
        "status": "ok",
        "catalog_size": len(products),
        "catalog_stale": catalog.is_stale(),
    }


@app.get("/api/categories")
async def get_categories() -> dict[str, list[str]]:
    return {"categories": catalog.get_categories()}


@app.post("/api/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest, request: Request) -> SearchResponse:
    """Search the catalog and return AI-ranked recommendations."""
    client_id = request.client.host if request.client else "anonymous"
    _check_rate_limit(client_id)

    try:
        validated = validate_search_request(
            query=req.query,
            category=req.category,
            price_min=req.price_min,
            price_max=req.price_max,
            min_rating=req.min_rating,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = req.session_id or create_session()
    session_store.update_session(
        session_id,
        {
            "query": validated["query"],
            "budget": validated["price_max"],
            "preferences": {"category": validated["category"]},
            "rating_requirements": validated["min_rating"],
        },
    )

    trace_id = audit_logger.create_trace(session_id)
    audit_logger.record_event(
        trace_id=trace_id,
        session_id=session_id,
        status="started",
        step="request_validation",
        latency_ms=0,
        validation_result="accepted",
    )

    if validated["category"]:
        valid_categories = catalog.get_categories()
        if valid_categories and validated["category"] not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Unknown category '{validated['category']}'.",
                    "valid_categories": valid_categories,
                },
            )

    candidates = search(
        query=validated["query"],
        category=validated["category"],
        price_min=validated["price_min"],
        price_max=validated["price_max"],
        min_rating=validated["min_rating"],
        limit=validated["limit"],
    )
    audit_logger.record_event(
        trace_id=trace_id,
        session_id=session_id,
        status="ok" if candidates else "empty",
        step="search",
        latency_ms=0,
        product_ids=[int(item["id"]) for item in candidates[:MAX_TOOL_CALLS]],
        validation_result="candidate_set_ready",
    )

    if not candidates:
        return SearchResponse(
            results=[],
            summary="No products match your filters. Try widening your price range or lowering the minimum rating.",
            applied_filters={
                "query": validated["query"],
                "category": validated["category"],
                "price_min": validated["price_min"],
                "price_max": validated["price_max"],
                "min_rating": validated["min_rating"],
            },
            errors=None,
        )

    llm_error: str | None = None
    try:
        comparison = compare(
            candidates=candidates,
            user_query=validated["query"],
            category=validated["category"],
            price_min=validated["price_min"],
            price_max=validated["price_max"],
            min_rating=validated["min_rating"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Comparator failed: %s", exc)
        llm_error = "AI comparison unavailable — showing top results by rating."
        comparison = None  # type: ignore[assignment]

    if comparison:
        rec_map = {r.id: r for r in comparison.recommended}
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            rec = rec_map.get(candidate["id"])
            enriched.append({
                **candidate,
                "rank": rec.rank if rec else 999,
                "ai_score": rec.score if rec else None,
                "reason": rec.reason if rec else None,
                "tradeoffs": rec.tradeoffs if rec else None,
                "why_not": rec.why_not if rec else "",
                "pros_llm": rec.pros_llm if rec else [],
                "cons_llm": rec.cons_llm if rec else [],
            })
        enriched.sort(key=lambda x: x.get("rank", 999))
        summary = comparison.comparison_summary
    else:
        enriched = sorted(candidates, key=lambda x: x.get("rating", 0), reverse=True)
        for i, item in enumerate(enriched):
            item["rank"] = i + 1
            item["reason"] = "Strong match based on rating and budget fit."
            item["tradeoffs"] = "Limited AI comparison is available; verify specific product details."
            item["why_not"] = "Unknown or unavailable alternative"
            item["pros_llm"] = []
            item["cons_llm"] = []
        summary = f"Top {len(enriched)} matches"
        if validated["category"]:
            summary += f" in {validated['category']}"
        if validated["price_max"] < 1_000_000:
            summary += f" between ${validated['price_min']:.0f}–${validated['price_max']:.0f}"
        if validated["min_rating"]:
            summary += f" with rating ≥ {validated['min_rating']}"

    audit_logger.record_event(
        trace_id=trace_id,
        session_id=session_id,
        status="ok"
        if comparison is not None or enriched
        else "fallback",
        step="recommendation",
        latency_ms=0,
        product_ids=[int(item["id"]) for item in enriched[:MAX_TOOL_CALLS]],
        recommendation_result="returned",
    )

    return SearchResponse(
        results=enriched,
        summary=summary,
        applied_filters={
            "query": validated["query"],
            "category": validated["category"],
            "price_min": validated["price_min"],
            "price_max": validated["price_max"],
            "min_rating": validated["min_rating"],
        },
        errors=llm_error,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Handle a conversational message, maintaining session context."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    reply, session_id = chat(
        message=req.message,
        session_id=req.session_id,
        product_context=req.context,  # type: ignore[arg-type]
    )
    return ChatResponse(reply=reply, session_id=session_id)


@app.post("/api/catalog/refresh")
async def refresh_catalog() -> dict[str, str]:
    """Manually trigger a catalog re-sync from dummyjson."""
    catalog.sync_catalog()
    return {"status": "refreshed", "catalog_size": str(len(catalog.get_all_products()))}
