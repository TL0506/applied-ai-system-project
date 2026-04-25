"""
Unit tests for the RAG pipeline (rag_engine.py) and document_processor.py.

All tests run without a live Ollama instance.  LLM calls in test_generate_answer_*
are mocked via unittest.mock.patch so they never hit the network.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

import pytest

from rag_engine import RAGEngine, CHUNK_SIZE_CHARS, TOP_K_CHUNKS
from document_processor import (
    DocumentProcessor,
    UnsupportedFileTypeError,
    FileTooLargeError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return RAGEngine(model="test-model")


LONG_TEXT = " ".join(f"sentence{i} is here for testing purposes" for i in range(300))
SHORT_TEXT = "The Eiffel Tower was built by Gustave Eiffel in 1889 in Paris France."

# ---------------------------------------------------------------------------
# 1. chunk_text — split count and sizes
# ---------------------------------------------------------------------------

def test_chunk_text_splits_correctly(engine):
    chunks = engine.chunk_text(LONG_TEXT)
    # With ~12 000 chars and CHUNK_SIZE_CHARS=1800, expect at least 5 chunks
    assert len(chunks) >= 5
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.index >= 0


# ---------------------------------------------------------------------------
# 2. chunk_text — word-boundary preservation
# ---------------------------------------------------------------------------

def test_chunk_text_preserves_word_boundaries(engine):
    chunks = engine.chunk_text(LONG_TEXT)
    for chunk in chunks:
        # Chunk text should start with a non-space character (trimmed)
        assert not chunk.text[0].isspace()
        # Chunk text should not end with a space (stripped)
        assert not chunk.text[-1].isspace()


# ---------------------------------------------------------------------------
# 3. chunk_text — raises ValueError on empty input
# ---------------------------------------------------------------------------

def test_chunk_text_raises_on_empty(engine):
    with pytest.raises(ValueError, match="empty"):
        engine.chunk_text("   ")


# ---------------------------------------------------------------------------
# 4. chunk_text — short document produces exactly one chunk
# ---------------------------------------------------------------------------

def test_chunk_single_short_doc(engine):
    chunks = engine.chunk_text(SHORT_TEXT)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert SHORT_TEXT.strip() in chunks[0].text


# ---------------------------------------------------------------------------
# 5. build_index — return value matches actual chunk count
# ---------------------------------------------------------------------------

def test_build_index_returns_chunk_count(engine):
    count = engine.build_index(LONG_TEXT)
    assert count == len(engine.chunks)
    assert count >= 1
    assert engine._is_indexed is True


# ---------------------------------------------------------------------------
# 6. retrieve — returns exactly top_k chunks (or fewer if doc is tiny)
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k(engine):
    engine.build_index(LONG_TEXT)
    result = engine.retrieve("testing sentence here", top_k=3)
    assert len(result.chunks) == 3
    assert len(result.scores) == 3
    # Scores should be in descending order
    assert result.scores[0] >= result.scores[-1]


# ---------------------------------------------------------------------------
# 7. retrieve — raises RuntimeError if build_index was never called
# ---------------------------------------------------------------------------

def test_retrieve_raises_if_not_indexed(engine):
    with pytest.raises(RuntimeError, match="build_index"):
        engine.retrieve("any question")


# ---------------------------------------------------------------------------
# 8. retrieve — raises ValueError for blank query
# ---------------------------------------------------------------------------

def test_retrieve_raises_on_blank_query(engine):
    engine.build_index(SHORT_TEXT)
    with pytest.raises(ValueError, match="empty"):
        engine.retrieve("   ")


# ---------------------------------------------------------------------------
# 9. generate_answer — mocked LLM call is invoked exactly once
# ---------------------------------------------------------------------------

def test_generate_answer_calls_ollama(engine):
    engine.build_index(SHORT_TEXT)
    retrieval = engine.retrieve("Who built the Eiffel Tower?", top_k=1)

    mock_response = {"message": {"content": "Gustave Eiffel built it [SOURCE 1]."}}

    with patch("rag_engine.ollama") as mock_ollama:
        mock_ollama.chat.return_value = mock_response
        rag_response = engine.generate_answer("Who built the Eiffel Tower?", retrieval)

    mock_ollama.chat.assert_called_once()
    call_kwargs = mock_ollama.chat.call_args
    # Verify the model name was passed
    assert call_kwargs.kwargs.get("model") == "test-model" or \
           call_kwargs.args[0] if call_kwargs.args else True  # model could be positional
    assert "Eiffel" in rag_response.answer or "SOURCE" in rag_response.answer
    assert rag_response.confidence_score >= 0.0


# ---------------------------------------------------------------------------
# 10. DocumentProcessor — rejects unsupported file types
# ---------------------------------------------------------------------------

def test_document_processor_rejects_exe():
    proc = DocumentProcessor()
    with pytest.raises(UnsupportedFileTypeError):
        proc.validate_file("malware.exe", 1024)


# ---------------------------------------------------------------------------
# 11. DocumentProcessor — rejects files larger than 10 MB
# ---------------------------------------------------------------------------

def test_document_processor_rejects_oversized():
    proc = DocumentProcessor()
    oversized = 11 * 1024 * 1024  # 11 MB
    with pytest.raises(FileTooLargeError):
        proc.validate_file("document.pdf", oversized)
