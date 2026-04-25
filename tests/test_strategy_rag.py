"""
Unit tests for strategy_rag.py and game_guardrails.py.

All tests run without a live Ollama instance — LLM calls are mocked
via patch on strategy_rag.ollama or patch.object on the instance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

import pytest

from strategy_rag import StrategyRAG, STRATEGY_KB, TOP_K_CHUNKS
from game_guardrails import (
    QueryValidator,
    EmptyQueryError,
    QueryTooShortError,
    QueryTooLongError,
    InvalidHintOutputError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rag():
    r = StrategyRAG(model="test-model")
    r.build_index()
    return r


@pytest.fixture
def validator():
    return QueryValidator()


# ---------------------------------------------------------------------------
# 1. KB chunking produces chunks
# ---------------------------------------------------------------------------

def test_kb_chunking_produces_chunks():
    r = StrategyRAG()
    chunks = r._chunk_kb(STRATEGY_KB)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) > 0
        assert c.index >= 0


# ---------------------------------------------------------------------------
# 2. Retrieve returns top_k with descending scores
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k_with_scores(rag):
    result = rag.retrieve("binary search midpoint", top_k=2)
    assert len(result.chunks) == 2
    assert len(result.scores) == 2
    assert result.scores[0] >= result.scores[1]


# ---------------------------------------------------------------------------
# 3. Retrieve raises RuntimeError if not indexed
# ---------------------------------------------------------------------------

def test_retrieve_raises_if_not_indexed():
    r = StrategyRAG()
    with pytest.raises(RuntimeError, match="build_index"):
        r.retrieve("any query")


# ---------------------------------------------------------------------------
# 4. Retrieve raises ValueError for blank query
# ---------------------------------------------------------------------------

def test_retrieve_raises_on_empty_query(rag):
    with pytest.raises(ValueError, match="empty"):
        rag.retrieve("   ")


# ---------------------------------------------------------------------------
# 5. generate_advice calls ollama.chat exactly once
# ---------------------------------------------------------------------------

def test_generate_advice_calls_ollama(rag):
    retrieval = rag.retrieve("best first guess", top_k=1)
    mock_resp = {"message": {"content": "Guess 50 using binary search [STRATEGY 1]."}}

    with patch("strategy_rag.ollama") as mock_ollama:
        mock_ollama.chat.return_value = mock_resp
        response = rag.generate_advice("best first guess", retrieval)

    mock_ollama.chat.assert_called_once()
    assert len(response.advice) > 0


# ---------------------------------------------------------------------------
# 6. generate_advice returns StrategyResponse with confidence >= 0
# ---------------------------------------------------------------------------

def test_generate_advice_returns_strategy_response(rag):
    retrieval = rag.retrieve("midpoint strategy", top_k=2)
    mock_resp = {"message": {"content": "Use 50 as your first guess [STRATEGY 1]."}}

    with patch("strategy_rag.ollama") as mock_ollama:
        mock_ollama.chat.return_value = mock_resp
        response = rag.generate_advice("midpoint strategy", retrieval)

    assert isinstance(response.advice, str)
    assert isinstance(response.confidence_score, float)
    assert response.confidence_score >= 0.0
    assert len(response.retrieved_chunks) >= 1


# ---------------------------------------------------------------------------
# 7. QueryValidator rejects empty query
# ---------------------------------------------------------------------------

def test_query_validator_rejects_empty_query(validator):
    with pytest.raises(EmptyQueryError):
        validator.validate_query("   ")


# ---------------------------------------------------------------------------
# 8. QueryValidator rejects too-short query
# ---------------------------------------------------------------------------

def test_query_validator_rejects_too_short(validator):
    with pytest.raises(QueryTooShortError):
        validator.validate_query("hi")


# ---------------------------------------------------------------------------
# 9. QueryValidator rejects too-long query
# ---------------------------------------------------------------------------

def test_query_validator_rejects_too_long(validator):
    with pytest.raises(QueryTooLongError):
        validator.validate_query("x" * 501)


# ---------------------------------------------------------------------------
# 10. validate_hint_output passes correct direction keywords
# ---------------------------------------------------------------------------

def test_validate_hint_output_passes_correct_direction(validator):
    # "Too High" → hint must contain a LOWER keyword — should NOT raise
    validator.validate_hint_output("Go lower! You're close.", "Too High")
    # "Too Low" → hint must contain a HIGHER keyword — should NOT raise
    validator.validate_hint_output("Try going higher next time!", "Too Low")


# ---------------------------------------------------------------------------
# 11. validate_hint_output raises for wrong direction
# ---------------------------------------------------------------------------

def test_validate_hint_output_raises_wrong_direction(validator):
    # outcome is "Too High" but hint says "higher" — wrong direction
    with pytest.raises(InvalidHintOutputError):
        validator.validate_hint_output("Go higher!", "Too High")
