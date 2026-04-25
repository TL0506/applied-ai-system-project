import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2"

try:
    import ollama as _ollama_module
except ImportError:
    _ollama_module = None

# Module-level reference so tests can patch 'ai_engine.ollama'
ollama = _ollama_module


@dataclass
class SolverStep:
    step: int
    guess: int
    outcome: str          # "Win", "Too High", "Too Low"
    search_range: tuple   # (low, high) before this guess
    explanation: str      # LLM-generated narration
    attempts_so_far: list


class AIEngine:

    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        logger.info("AIEngine initialized: model=%s", model)

    # ------------------------------------------------------------------
    # AI Hints
    # ------------------------------------------------------------------

    def generate_hint(self, guess: int, outcome: str, low: int, high: int, attempt: int) -> str:
        """Generate a fun, contextual hint after a wrong guess.
        Falls back to a simple hint if Ollama is unavailable.
        """
        direction = "lower" if outcome == "Too High" else "higher"
        prompt = (
            f"You are the host of a fun number guessing game. "
            f"The player guessed {guess}. The secret is somewhere between {low} and {high}. "
            f"The guess was too {'high' if outcome == 'Too High' else 'low'} — "
            f"they need to go {direction}. "
            f"This is attempt number {attempt}. "
            f"Give a short, fun, encouraging hint (1-2 sentences). "
            f"Include the direction ({direction}) but do NOT reveal the number. "
            f"Keep it playful and game-show-style."
        )
        try:
            return self._call_ollama(prompt, max_tokens=80)
        except Exception as e:
            logger.warning("Hint generation failed, using fallback: %s", e)
            emoji = "📉" if outcome == "Too High" else "📈"
            return f"{emoji} Go {direction.upper()}! (AI hints unavailable — is Ollama running?)"

    # ------------------------------------------------------------------
    # Game Analysis
    # ------------------------------------------------------------------

    def analyze_game(
        self,
        guess_history: list,
        secret: int,
        difficulty: str,
        outcome: str,
        attempt_limit: int,
    ) -> str:
        """Analyze a completed game session and provide coaching feedback."""
        int_guesses = [g for g in guess_history if isinstance(g, int)]
        history_str = " → ".join(str(g) for g in int_guesses) if int_guesses else "no valid guesses"

        prompt = (
            f"A player just finished a number guessing game.\n"
            f"Difficulty: {difficulty}\n"
            f"Secret number: {secret}\n"
            f"Guesses (in order): {history_str}\n"
            f"Outcome: {outcome} in {len(int_guesses)} guess(es) "
            f"(limit was {attempt_limit})\n\n"
            f"Analyze the player's guessing strategy in 3-4 sentences. Cover:\n"
            f"1. Was their search pattern efficient? (Binary search halves the range each step)\n"
            f"2. What was their best or worst guess?\n"
            f"3. One specific tip to improve next time.\n"
            f"Be encouraging but honest."
        )
        return self._call_ollama(prompt, max_tokens=300)

    # ------------------------------------------------------------------
    # Agentic Auto-Solver
    # ------------------------------------------------------------------

    def auto_solve(self, secret: int, low: int, high: int, max_attempts: int) -> list:
        """Agentic binary-search solver.

        Each step follows the Plan → Act → Check → Reflect loop:
          PLAN:    Pick the midpoint of the current search range.
          ACT:     Guess that number.
          CHECK:   Compare to the secret; narrow the range.
          REFLECT: Generate a natural-language explanation via LLM.

        Returns a list of SolverStep objects.
        """
        steps = []
        current_low, current_high = low, high
        attempts_so_far: list = []

        for step_num in range(1, max_attempts + 1):
            # PLAN
            guess = (current_low + current_high) // 2

            # ACT + CHECK
            if guess < secret:
                outcome = "Too Low"
            elif guess > secret:
                outcome = "Too High"
            else:
                outcome = "Win"

            attempts_so_far.append(guess)

            # REFLECT
            explanation = self._explain_solver_step(
                step_num, guess, outcome, current_low, current_high,
                secret if outcome == "Win" else None,
            )

            steps.append(
                SolverStep(
                    step=step_num,
                    guess=guess,
                    outcome=outcome,
                    search_range=(current_low, current_high),
                    explanation=explanation,
                    attempts_so_far=list(attempts_so_far),
                )
            )

            if outcome == "Win":
                break
            elif outcome == "Too High":
                current_high = guess - 1
            else:
                current_low = guess + 1

        return steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _explain_solver_step(
        self,
        step: int,
        guess: int,
        outcome: str,
        low: int,
        high: int,
        secret: Optional[int] = None,
    ) -> str:
        """LLM narration of one solver step; falls back to a template string."""
        if outcome == "Win":
            prompt = (
                f"An AI just solved a number guessing game in {step} step(s)! "
                f"It searched between {low} and {high}, guessed {guess}, and that was correct. "
                f"Explain in one enthusiastic sentence why binary search is so efficient."
            )
        else:
            new_low = guess + 1 if outcome == "Too Low" else low
            new_high = guess - 1 if outcome == "Too High" else high
            remaining = max(0, new_high - new_low + 1)
            prompt = (
                f"Step {step} of an AI number-guessing solver.\n"
                f"Search range: {low}–{high}. AI chose midpoint: {guess}.\n"
                f"Result: {outcome}. New range: {new_low}–{new_high} "
                f"({remaining} number(s) left).\n"
                f"Narrate this step in 1-2 sentences as if explaining the AI's reasoning."
            )
        try:
            return self._call_ollama(prompt, max_tokens=80)
        except Exception as e:
            logger.warning("Step explanation failed: %s", e)
            return self._fallback_explanation(step, guess, outcome, low, high)

    def _fallback_explanation(
        self, step: int, guess: int, outcome: str, low: int, high: int
    ) -> str:
        if outcome == "Win":
            return f"Found it! Binary search converged to {guess} in {step} step(s)."
        direction = "lower" if outcome == "Too High" else "higher"
        new_low = guess + 1 if outcome == "Too Low" else low
        new_high = guess - 1 if outcome == "Too High" else high
        remaining = max(0, new_high - new_low + 1)
        return (
            f"Step {step}: guessed {guess} (midpoint of {low}–{high}). "
            f"{outcome} → searching {direction}. {remaining} number(s) remain."
        )

    def _call_ollama(self, prompt: str, max_tokens: int = 256) -> str:
        """Send a prompt to Ollama and return the response text."""
        global ollama
        if ollama is None:
            raise RuntimeError(
                "The 'ollama' package is not installed. Run: pip install ollama"
            )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens},
            )
            if isinstance(response, dict):
                return response["message"]["content"].strip()
            return response.message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("connection", "refused", "connect error", "cannot connect")):
                raise ConnectionError(
                    "Cannot connect to Ollama. Make sure it is running: `ollama serve`"
                ) from e
            if "model" in err_str and ("not found" in err_str or "pull" in err_str):
                raise RuntimeError(
                    f"Model '{self.model}' not found. Run: `ollama pull {self.model}`"
                ) from e
            raise
