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
async def search_endpoint(req: SearchRequest) -> SearchResponse:
    """Search the catalog and return AI-ranked recommendations."""

    # Validate category
    if req.category:
        valid_categories = catalog.get_categories()
        if valid_categories and req.category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Unknown category '{req.category}'.",
                    "valid_categories": valid_categories,
                },
            )

    # Validate price range
    if req.price_min < 0 or req.price_max < 0:
        raise HTTPException(status_code=400, detail="Price values must be non-negative.")
    if req.price_min > req.price_max:
        raise HTTPException(status_code=400, detail="price_min must be <= price_max.")
    if not 0.0 <= req.min_rating <= 5.0:
        raise HTTPException(status_code=400, detail="min_rating must be between 0.0 and 5.0.")

    candidates = search(
        query=req.query,
        category=req.category,
        price_min=req.price_min,
        price_max=req.price_max,
        min_rating=req.min_rating,
    )

    if not candidates:
        return SearchResponse(
            results=[],
            summary="No products match your filters. Try widening your price range or lowering the minimum rating.",
            applied_filters={
                "query": req.query,
                "category": req.category,
                "price_min": req.price_min,
                "price_max": req.price_max,
                "min_rating": req.min_rating,
            },
            errors=None,
        )

    # AI comparison
    llm_error: str | None = None
    try:
        comparison = compare(
            candidates=candidates,
            user_query=req.query,
            category=req.category,
            price_min=req.price_min,
            price_max=req.price_max,
            min_rating=req.min_rating,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Comparator failed: %s", exc)
        llm_error = "AI comparison unavailable — showing top results by rating."
        comparison = None  # type: ignore[assignment]

    # Build enriched results
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
                "pros_llm": rec.pros_llm if rec else [],
                "cons_llm": rec.cons_llm if rec else [],
            })
        # Sort by rank
        enriched.sort(key=lambda x: x.get("rank", 999))
        summary = comparison.comparison_summary
    else:
        enriched = sorted(candidates, key=lambda x: x.get("rating", 0), reverse=True)
        for i, item in enumerate(enriched):
            item["rank"] = i + 1
        summary = f"Top {len(enriched)} matches"
        if req.category:
            summary += f" in {req.category}"
        if req.price_max < 1_000_000:
            summary += f" between ${req.price_min:.0f}–${req.price_max:.0f}"
        if req.min_rating:
            summary += f" with rating ≥ {req.min_rating}"

    return SearchResponse(
        results=enriched,
        summary=summary,
        applied_filters={
            "query": req.query,
            "category": req.category,
            "price_min": req.price_min,
            "price_max": req.price_max,
            "min_rating": req.min_rating,
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
