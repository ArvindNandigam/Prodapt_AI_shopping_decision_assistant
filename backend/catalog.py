"""Product catalog loader — fetches from dummyjson once at startup, caches in memory.

PII guarantee: reviewerName and reviewerEmail are stripped by Product.from_dummyjson()
before anything is stored in the cache.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.models import Product

logger = logging.getLogger(__name__)

_SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "catalog_snapshot.json"

# ---------------------------------------------------------------------------
# Catalog Cache (singleton-like module state)
# ---------------------------------------------------------------------------
_products: list[Product] = []
_categories: list[str] = []
_last_synced_at: float = 0.0


def get_all_products() -> list[Product]:
    return _products


def get_categories() -> list[str]:
    return _categories


def is_stale() -> bool:
    return (time.time() - _last_synced_at) > settings.catalog_refresh_interval_seconds


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_all_products() -> list[dict]:
    url = f"{settings.dummyjson_base_url}/products?limit=0"
    logger.info("Fetching catalog from %s", url)
    with httpx.Client(timeout=15.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json().get("products", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_category_list() -> list[str]:
    url = f"{settings.dummyjson_base_url}/products/category-list"
    logger.info("Fetching category list from %s", url)
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _save_snapshot(raw_products: list[dict]) -> None:
    """Persist a fallback snapshot to disk."""
    try:
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(json.dumps(raw_products, indent=2))
        logger.info("Saved catalog snapshot to %s", _SNAPSHOT_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save snapshot: %s", exc)


def _load_snapshot() -> Optional[list[dict]]:
    """Load the last known-good snapshot from disk."""
    if _SNAPSHOT_PATH.exists():
        try:
            data = json.loads(_SNAPSHOT_PATH.read_text())
            logger.info("Loaded fallback catalog snapshot (%d products)", len(data))
            return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load snapshot: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public sync entry point
# ---------------------------------------------------------------------------

def sync_catalog() -> None:
    """Fetch catalog from dummyjson and populate the in-memory cache.

    Falls back to the last on-disk snapshot if the API is unreachable.
    """
    global _products, _categories, _last_synced_at  # noqa: PLW0603

    raw_products: Optional[list[dict]] = None

    try:
        raw_products = _fetch_all_products()
        _save_snapshot(raw_products)
    except Exception as exc:  # noqa: BLE001
        logger.error("dummyjson fetch failed: %s — trying snapshot fallback", exc)
        raw_products = _load_snapshot()

    if raw_products is None:
        logger.critical("No product data available — catalog is empty")
        return

    _products = [Product.from_dummyjson(p) for p in raw_products]
    _last_synced_at = time.time()

    try:
        _categories = _fetch_category_list()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Category list fetch failed, deriving from catalog: %s", exc)
        _categories = sorted({p.category for p in _products if p.category})

    logger.info(
        "Catalog ready: %d products across %d categories",
        len(_products),
        len(_categories),
    )
