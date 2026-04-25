"""
Unit tests for ai_engine.py.

All tests run without a live Ollama instance — LLM calls are mocked
via patch.object on ai._call_ollama or patch on ai_engine.ollama.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

import pytest

from ai_engine import AIEngine, SolverStep

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ai():
    return AIEngine(model="test-model")


# ---------------------------------------------------------------------------
# Auto-solver — correctness
# ---------------------------------------------------------------------------

def test_auto_solve_finds_easy_secret(ai):
    """Solver must reach Win for secret=10, range 1–20."""
    with patch.object(ai, "_call_ollama", return_value="ok"):
        steps = ai.auto_solve(secret=10, low=1, high=20, max_attempts=5)
    assert steps[-1].outcome == "Win"
    assert steps[-1].guess == 10


def test_auto_solve_finds_secret_at_boundary_low(ai):
    """Solver handles secret at the lowest boundary."""
    with patch.object(ai, "_call_ollama", return_value="ok"):
        steps = ai.auto_solve(secret=1, low=1, high=100, max_attempts=10)
    assert steps[-1].outcome == "Win"


def test_auto_solve_finds_secret_at_boundary_high(ai):
    """Solver handles secret at the highest boundary."""
    with patch.object(ai, "_call_ollama", return_value="ok"):
        steps = ai.auto_solve(secret=100, low=1, high=100, max_attempts=10)
    assert steps[-1].outcome == "Win"


def test_auto_solve_single_value_range(ai):
    """When low == high == secret, solver solves in exactly 1 step."""
    with patch.object(ai, "_call_ollama", return_value="ok"):
        steps = ai.auto_solve(secret=7, low=7, high=7, max_attempts=5)
    assert len(steps) == 1
    assert steps[0].outcome == "Win"


# ---------------------------------------------------------------------------
# Auto-solver — binary search correctness
# ---------------------------------------------------------------------------

def test_auto_solve_uses_midpoint_each_step(ai):
    """Every guess must equal (low + high) // 2 of the current search range."""
    with patch.object(ai, "_call_ollama", return_value="ok"):
        steps = ai.auto_solve(secret=42, low=1, high=100, max_attempts=7)

    cur_low, cur_high = 1, 100
    for step in steps:
        assert step.guess == (cur_low + cur_high) // 2, (
            f"Step {step.step}: expected guess {(cur_low + cur_high)//2}, got {step.guess}"
        )
        if step.outcome == "Too Low":
            cur_low = step.guess + 1
        elif step.outcome == "Too High":
            cur_high = step.guess - 1


def test_auto_solve_efficient_step_count(ai):
    """Binary search on 1–100 should always solve in ≤7 steps."""
    with patch.object(ai, "_call_ollama", return_value="ok"):
        for secret in range(1, 101, 5):
            steps = ai.auto_solve(secret, 1, 100, max_attempts=10)
            assert len(steps) <= 7, f"Secret {secret} took {len(steps)} steps"


# ---------------------------------------------------------------------------
# AI Hints
# ---------------------------------------------------------------------------

def test_generate_hint_returns_string(ai):
    """generate_hint returns a non-empty string when Ollama succeeds."""
    with patch.object(ai, "_call_ollama", return_value="Go lower! You're getting warmer."):
        hint = ai.generate_hint(75, "Too High", 1, 100, 3)
    assert isinstance(hint, str)
    assert len(hint) > 0


def test_generate_hint_fallback_on_connection_error(ai):
    """generate_hint falls back gracefully if Ollama is unavailable."""
    with patch.object(ai, "_call_ollama", side_effect=ConnectionError("refused")):
        hint = ai.generate_hint(50, "Too Low", 1, 100, 2)
    assert isinstance(hint, str)
    assert len(hint) > 0
    assert "higher" in hint.lower() or "unavailable" in hint.lower()


# ---------------------------------------------------------------------------
# Game Analysis
# ---------------------------------------------------------------------------

def test_analyze_game_calls_ollama(ai):
    """analyze_game must invoke _call_ollama exactly once."""
    with patch.object(ai, "_call_ollama", return_value="Great strategy!") as mock_call:
        result = ai.analyze_game(
            guess_history=[50, 75, 62],
            secret=60,
            difficulty="Normal",
            outcome="lost",
            attempt_limit=8,
        )
    mock_call.assert_called_once()
    assert result == "Great strategy!"


def test_analyze_game_returns_string(ai):
    """analyze_game always returns a string."""
    with patch.object(ai, "_call_ollama", return_value="Nice try!"):
        result = ai.analyze_game([10, 20], 15, "Easy", "won", 6)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _call_ollama internals
# ---------------------------------------------------------------------------

def test_call_ollama_raises_connection_error(ai):
    """_call_ollama wraps connection failures as ConnectionError."""
    mock_response = MagicMock()
    mock_response.side_effect = Exception("connection refused")

    with patch("ai_engine.ollama") as mock_ollama:
        mock_ollama.chat.side_effect = Exception("connection refused")
        with pytest.raises(ConnectionError, match="Ollama"):
            ai._call_ollama("test prompt")
