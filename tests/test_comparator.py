"""Unit tests for the LLM comparator — fallback logic and output parsing."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from backend.comparator import _fallback_result, _parse_llm_output, _trim_candidate
from backend.models import ComparisonResult
from langchain_core.exceptions import OutputParserException


FAKE_CANDIDATES = [
    {
        "id": 1,
        "title": "Nike Shoe",
        "brand": "Nike",
        "price": 89.99,
        "discountPercentage": 10.0,
        "rating": 4.5,
        "stock": 30,
        "availabilityStatus": "In Stock",
        "warrantyInformation": "1 year",
        "returnPolicy": "30 days",
        "reviews": [{"rating": 5, "comment": "Great!"}],
    },
    {
        "id": 2,
        "title": "Adidas Shoe",
        "brand": "Adidas",
        "price": 69.99,
        "discountPercentage": 5.0,
        "rating": 4.1,
        "stock": 5,
        "availabilityStatus": "Low Stock",
        "warrantyInformation": "6 months",
        "returnPolicy": "14 days",
        "reviews": [],
    },
]


class TestFallbackResult:
    def test_returns_comparison_result(self):
        result = _fallback_result(FAKE_CANDIDATES)
        assert isinstance(result, ComparisonResult)

    def test_sorted_by_rating_descending(self):
        result = _fallback_result(FAKE_CANDIDATES)
        assert result.recommended[0].id == 1  # Nike has higher rating (4.5 > 4.1)
        assert result.recommended[0].rank == 1

    def test_fallback_summary_indicates_unavailability(self):
        result = _fallback_result(FAKE_CANDIDATES)
        assert "unavailable" in result.comparison_summary.lower()

    def test_empty_candidates(self):
        result = _fallback_result([])
        assert result.recommended == []


class TestParseOutput:
    def test_valid_output(self):
        raw = {
            "recommended": [
                {
                    "id": 1,
                    "score": 0.9,
                    "rank": 1,
                    "reason": "Best value",
                    "tradeoffs": "Pricier",
                    "pros_llm": ["great cushioning"],
                    "cons_llm": ["heavy"],
                },
                {
                    "id": 2,
                    "score": 0.7,
                    "rank": 2,
                    "reason": "Budget option",
                    "tradeoffs": "Lower stock",
                    "pros_llm": ["affordable"],
                    "cons_llm": ["shorter warranty"],
                },
            ],
            "comparison_summary": "Nike leads on rating; Adidas wins on price.",
        }
        result = _parse_llm_output(raw, FAKE_CANDIDATES)
        assert len(result.recommended) == 2
        assert result.recommended[0].id == 1
        assert result.recommended[0].score == 0.9
        assert result.comparison_summary == "Nike leads on rating; Adidas wins on price."

    def test_unknown_product_id_skipped(self):
        raw = {
            "recommended": [
                {
                    "id": 999,  # not in candidates
                    "score": 0.8,
                    "rank": 1,
                    "reason": "Unknown",
                    "tradeoffs": "",
                    "pros_llm": [],
                    "cons_llm": [],
                }
            ],
            "comparison_summary": "Summary",
        }
        with pytest.raises(OutputParserException):
            _parse_llm_output(raw, FAKE_CANDIDATES)

    def test_non_dict_raises(self):
        with pytest.raises(OutputParserException):
            _parse_llm_output("not a dict", FAKE_CANDIDATES)


class TestTrimCandidate:
    def test_pii_not_in_trimmed(self):
        raw = {**FAKE_CANDIDATES[0], "reviews": [{"rating": 5, "comment": "Good", "reviewerName": "Alice"}]}
        trimmed = _trim_candidate(raw)
        assert "reviewerName" not in str(trimmed)
        assert "Alice" not in str(trimmed)

    def test_required_fields_present(self):
        trimmed = _trim_candidate(FAKE_CANDIDATES[0])
        for field in ["id", "title", "price", "rating", "stock", "returnPolicy"]:
            assert field in trimmed
