import logging
import logging.handlers
import os
import random

# Logging — guard prevents duplicate handlers on Streamlit reruns
_root = logging.getLogger()
if not _root.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _fh = logging.handlers.RotatingFileHandler(
        "app.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    _sh = logging.StreamHandler()
    _fh.setFormatter(_fmt)
    _sh.setFormatter(_fmt)
    _root.addHandler(_fh)
    _root.addHandler(_sh)
    _root.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from logic_utils import check_guess, parse_guess, get_range_for_difficulty, update_score
from ai_engine import AIEngine
from strategy_rag import StrategyRAG
from game_guardrails import QueryValidator, EmptyQueryError, QueryTooShortError, QueryTooLongError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Game Glitch Investigator",
    page_icon="🎮",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Singleton AIEngine
# ---------------------------------------------------------------------------

@st.cache_resource
def get_ai_engine() -> AIEngine:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    logger.info("AIEngine created: model=%s", model)
    return AIEngine(model=model, host=host)


ai = get_ai_engine()


@st.cache_resource
def get_strategy_rag() -> StrategyRAG:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    rag = StrategyRAG(model=model)
    rag.build_index()
    logger.info("StrategyRAG indexed: %d chunks", len(rag.chunks))
    return rag


strategy_rag_inst = get_strategy_rag()
validator = QueryValidator()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    difficulty = st.selectbox("Difficulty", ["Easy", "Normal", "Hard"], index=1)

    attempt_limit_map = {"Easy": 6, "Normal": 8, "Hard": 5}
    attempt_limit = attempt_limit_map[difficulty]
    low, high = get_range_for_difficulty(difficulty)

    st.caption(f"Range: {low}–{high}  |  Attempts: {attempt_limit}")
    st.divider()

    ai_hints_on = st.toggle(
        "🤖 AI Hints",
        value=False,
        help="Let Ollama generate creative hints instead of plain Higher/Lower messages.",
    )
    st.caption(f"Model: `{os.getenv('OLLAMA_MODEL', 'llama3.2')}`")
    st.caption("Runs locally — no API key needed.")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "secret" not in st.session_state:
    st.session_state.secret = random.randint(low, high)
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "status" not in st.session_state:
    st.session_state.status = "playing"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_difficulty" not in st.session_state:
    st.session_state.last_difficulty = difficulty
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "solver_steps" not in st.session_state:
    st.session_state.solver_steps = None
if "solver_secret" not in st.session_state:
    st.session_state.solver_secret = None

# Reset game when difficulty changes
if st.session_state.last_difficulty != difficulty:
    new_low, new_high = get_range_for_difficulty(difficulty)
    st.session_state.secret = random.randint(new_low, new_high)
    st.session_state.attempts = 0
    st.session_state.score = 0
    st.session_state.status = "playing"
    st.session_state.history = []
    st.session_state.analysis_result = None
    st.session_state.last_difficulty = difficulty
    logger.info("Difficulty changed to %s; game reset", difficulty)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_play, tab_solver, tab_analysis, tab_strategy, tab_reliability = st.tabs(
    ["🎮 Play", "🤖 Auto-Solver", "📊 Analysis", "💡 Strategy Advisor", "🧪 Reliability"]
)

# ===========================================================================
# TAB 1 — Play
# ===========================================================================

with tab_play:
    st.title("🎮 Game Glitch Investigator")
    st.caption("An AI-generated guessing game. Something might still be off.")

    st.info(
        f"Guess a number between **{low}** and **{high}**. "
        f"Attempts left: **{attempt_limit - st.session_state.attempts}**"
    )

    with st.expander("🔍 Developer Debug Info"):
        st.write("Secret:", st.session_state.secret)
        st.write("Attempts:", st.session_state.attempts)
        st.write("Score:", st.session_state.score)
        st.write("Difficulty:", difficulty)
        st.write("History:", st.session_state.history)

    raw_guess = st.text_input("Enter your guess:", key=f"guess_input_{difficulty}")

    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Submit Guess 🚀")
    with col2:
        new_game = st.button("New Game 🔁")

    # New game reset
    if new_game:
        st.session_state.secret = random.randint(low, high)
        st.session_state.attempts = 0
        st.session_state.score = 0
        st.session_state.history = []
        st.session_state.status = "playing"
        st.session_state.analysis_result = None
        logger.info("New game started: difficulty=%s", difficulty)
        st.success("New game started!")
        st.rerun()

    # Game-over guard
    if st.session_state.status != "playing":
        if st.session_state.status == "won":
            st.success(
                f"🎉 You won! The secret was **{st.session_state.secret}**. "
                f"Final score: **{st.session_state.score}**"
            )
        else:
            st.error(
                f"💀 Out of attempts! The secret was **{st.session_state.secret}**. "
                f"Score: **{st.session_state.score}**"
            )
        st.caption("Start a new game or check the Analysis tab for feedback.")
        st.stop()

    # Submit guess
    if submit:
        st.session_state.attempts += 1
        ok, guess_int, err = parse_guess(raw_guess, low, high)

        if not ok:
            st.session_state.history.append(raw_guess)
            st.error(err)
            logger.info("Invalid guess: %s", raw_guess)
        else:
            st.session_state.history.append(guess_int)
            outcome, simple_message = check_guess(guess_int, st.session_state.secret)

            st.session_state.score = update_score(
                current_score=st.session_state.score,
                outcome=outcome,
                attempt_number=st.session_state.attempts,
            )

            if outcome == "Win":
                st.balloons()
                st.session_state.status = "won"
                st.success(
                    f"🎉 Correct! The secret was **{st.session_state.secret}**. "
                    f"Final score: **{st.session_state.score}**"
                )
                logger.info(
                    "Game won: secret=%d, attempts=%d, score=%d",
                    st.session_state.secret, st.session_state.attempts, st.session_state.score,
                )
            else:
                # Show hint (AI or simple)
                if ai_hints_on:
                    with st.spinner("🤖 Generating hint..."):
                        hint = ai.generate_hint(
                            guess=guess_int,
                            outcome=outcome,
                            low=low,
                            high=high,
                            attempt=st.session_state.attempts,
                        )
                    st.warning(hint)
                    logger.info("AI hint generated for guess=%d outcome=%s", guess_int, outcome)
                else:
                    st.warning(simple_message)

                if st.session_state.attempts >= attempt_limit:
                    st.session_state.status = "lost"
                    st.error(
                        f"💀 Out of attempts! The secret was **{st.session_state.secret}**. "
                        f"Score: **{st.session_state.score}**"
                    )
                    logger.info(
                        "Game lost: secret=%d, attempts=%d",
                        st.session_state.secret, st.session_state.attempts,
                    )

    # Guess history
    if st.session_state.history:
        st.divider()
        st.caption("Guess history: " + "  →  ".join(str(g) for g in st.session_state.history))
        st.caption(f"Score: {st.session_state.score}")

# ===========================================================================
# TAB 2 — Auto-Solver (Agentic Workflow)
# ===========================================================================

with tab_solver:
    st.header("🤖 AI Auto-Solver")
    st.markdown(
        "Watch an AI agent solve the guessing game using **binary search**. "
        "Each step follows the **Plan → Act → Check → Reflect** loop."
    )
    st.markdown(
        "- **PLAN**: Choose the midpoint of the remaining search range  \n"
        "- **ACT**: Guess that number  \n"
        "- **CHECK**: Compare to the secret; narrow the range  \n"
        "- **REFLECT**: The AI explains its reasoning"
    )

    solver_col1, solver_col2 = st.columns(2)
    with solver_col1:
        solver_difficulty = st.selectbox(
            "Puzzle difficulty", ["Easy", "Normal", "Hard"], index=1, key="solver_diff"
        )
    with solver_col2:
        s_low, s_high = get_range_for_difficulty(solver_difficulty)
        s_limit = attempt_limit_map[solver_difficulty]
        st.metric("Search range", f"{s_low}–{s_high}")

    gen_col, solve_col = st.columns(2)
    with gen_col:
        if st.button("🎲 Generate New Puzzle"):
            st.session_state.solver_secret = random.randint(s_low, s_high)
            st.session_state.solver_steps = None
            logger.info("Solver puzzle generated: secret=%d", st.session_state.solver_secret)
            st.rerun()

    if st.session_state.solver_secret is not None:
        st.info(f"Secret number: **{st.session_state.solver_secret}** (range {s_low}–{s_high})")

        with solve_col:
            if st.button("▶ Watch AI Solve"):
                with st.spinner("AI is solving the puzzle step by step..."):
                    steps = ai.auto_solve(
                        secret=st.session_state.solver_secret,
                        low=s_low,
                        high=s_high,
                        max_attempts=s_limit,
                    )
                    st.session_state.solver_steps = steps
                    logger.info(
                        "Auto-solver finished: %d steps, outcome=%s",
                        len(steps),
                        steps[-1].outcome if steps else "none",
                    )

        if st.session_state.solver_steps:
            steps = st.session_state.solver_steps
            st.divider()
            won = steps[-1].outcome == "Win"
            if won:
                st.success(f"✅ Solved in **{len(steps)}** step(s)!")
            else:
                st.error("❌ Could not solve within the attempt limit.")

            for s in steps:
                icon = "✅" if s.outcome == "Win" else ("📉" if s.outcome == "Too High" else "📈")
                with st.expander(
                    f"Step {s.step}: guessed **{s.guess}** — {icon} {s.outcome}",
                    expanded=True,
                ):
                    st.markdown(f"**Search range:** {s.search_range[0]}–{s.search_range[1]}")
                    st.markdown(f"**Guess (midpoint):** {s.guess}")
                    st.markdown(f"**Result:** {s.outcome}")
                    st.markdown(f"**AI says:** _{s.explanation}_")
    else:
        st.info("Click **Generate New Puzzle** to create a puzzle for the AI to solve.")

# ===========================================================================
# TAB 3 — Analysis
# ===========================================================================

with tab_analysis:
    st.header("📊 Game Analysis")
    st.markdown(
        "After finishing a game, the AI analyzes your guessing strategy "
        "and gives you coaching feedback."
    )

    game_done = st.session_state.status in ("won", "lost")
    int_guesses = [g for g in st.session_state.history if isinstance(g, int)]

    if not game_done:
        st.info("Finish a game in the **Play** tab first, then come back here.")
    elif not int_guesses:
        st.warning("No valid guesses recorded — play at least one round.")
    else:
        st.markdown(f"**Game outcome:** {st.session_state.status.upper()}")
        st.markdown(f"**Secret was:** {st.session_state.secret}")
        st.markdown(f"**Your guesses:** " + " → ".join(str(g) for g in int_guesses))
        st.markdown(f"**Final score:** {st.session_state.score}")

        if st.button("🔍 Analyze My Strategy"):
            with st.spinner("AI is analyzing your game..."):
                try:
                    analysis = ai.analyze_game(
                        guess_history=st.session_state.history,
                        secret=st.session_state.secret,
                        difficulty=difficulty,
                        outcome=st.session_state.status,
                        attempt_limit=attempt_limit,
                    )
                    st.session_state.analysis_result = analysis
                    logger.info("Game analysis generated: %d chars", len(analysis))
                except ConnectionError as e:
                    st.error(f"🔌 {e}")
                    logger.error("Ollama connection error during analysis: %s", e)
                except RuntimeError as e:
                    st.error(f"⚠️ {e}")
                    logger.error("Runtime error during analysis: %s", e)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    logger.exception("Unexpected error during analysis")

        if st.session_state.analysis_result:
            st.divider()
            st.subheader("AI Coaching Feedback")
            st.markdown(st.session_state.analysis_result)

# ===========================================================================
# TAB 4 — Strategy Advisor (RAG over built-in game strategy knowledge base)
# ===========================================================================

with tab_strategy:
    st.header("💡 Strategy Advisor")
    st.markdown(
        "Ask a strategy question or describe your current game situation. "
        "The advisor retrieves relevant strategy excerpts and generates grounded advice."
    )
    st.caption(
        "Example questions:  "
        '"I keep guessing randomly — what should I do?" · '
        '"Secret is between 30 and 60, I guessed 45, too high" · '
        '"I have 2 attempts left and the range is 20–35"'
    )

    query_input = st.text_input(
        "Describe your situation or ask a strategy question:",
        key="strategy_query",
        placeholder="e.g. I keep guessing randomly with no strategy",
    )

    if st.button("💡 Get Strategy Advice"):
        if not query_input:
            st.warning("Please enter a question or describe your situation.")
        else:
            try:
                validator.validate_query(query_input)
                safe_query = validator.sanitize_query(query_input)

                with st.spinner("Retrieving strategy knowledge..."):
                    response = strategy_rag_inst.advise(safe_query)

                st.subheader("Strategy Advice")
                st.markdown(response.advice)

                st.metric("Retrieval Confidence", f"{response.confidence_score:.2f}")

                with st.expander("📚 Retrieved Strategy Excerpts"):
                    for i, (chunk, score) in enumerate(
                        zip(response.retrieved_chunks, response.retrieval_scores), 1
                    ):
                        st.markdown(f"**[STRATEGY {i}]** (score: {score:.3f})")
                        st.caption(chunk.text)
                        st.divider()

                logger.info(
                    "Strategy advice generated: query='%s...', confidence=%.2f",
                    safe_query[:40],
                    response.confidence_score,
                )

            except EmptyQueryError as e:
                st.error(f"Empty query: {e}")
            except QueryTooShortError as e:
                st.error(f"Query too short: {e}")
            except QueryTooLongError as e:
                st.error(f"Query too long: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logger.exception("Strategy Advisor error")

# ===========================================================================
# TAB 5 — Reliability
# ===========================================================================

with tab_reliability:
    st.header("🧪 Reliability Tests")
    st.markdown(
        "Runs automated checks to verify the AI components work correctly. "
        "Tests 1–3 run without Ollama. Tests 4–5 require Ollama to be running."
    )

    def _run_reliability_tests() -> list:
        results = []

        # --- TEST 1: Auto-solver always finds the secret (no Ollama needed) ---
        try:
            all_found = True
            for test_secret in [1, 50, 100, 13, 77]:
                from unittest.mock import patch as _patch
                with _patch.object(ai, "_call_ollama", return_value="ok"):
                    steps = ai.auto_solve(test_secret, 1, 100, 7)
                if not steps or steps[-1].outcome != "Win":
                    all_found = False
                    break
            results.append({
                "id": "T1",
                "name": "Auto-solver correctness",
                "passed": all_found,
                "note": "Tested secrets: 1, 13, 50, 77, 100" if all_found else "Failed to find one of the secrets",
            })
        except Exception as e:
            results.append({"id": "T1", "name": "Auto-solver correctness", "passed": False, "note": str(e)})

        # --- TEST 2: Auto-solver uses binary search (midpoint each step) ---
        try:
            from unittest.mock import patch as _patch
            with _patch.object(ai, "_call_ollama", return_value="ok"):
                steps = ai.auto_solve(42, 1, 100, 7)
            binary_ok = True
            cur_low, cur_high = 1, 100
            for step in steps:
                expected_mid = (cur_low + cur_high) // 2
                if step.guess != expected_mid:
                    binary_ok = False
                    break
                if step.outcome == "Too Low":
                    cur_low = step.guess + 1
                elif step.outcome == "Too High":
                    cur_high = step.guess - 1
            results.append({
                "id": "T2",
                "name": "Binary search midpoint accuracy",
                "passed": binary_ok,
                "note": "Each guess matches (low+high)//2" if binary_ok else "Midpoint mismatch detected",
            })
        except Exception as e:
            results.append({"id": "T2", "name": "Binary search midpoint accuracy", "passed": False, "note": str(e)})

        # --- TEST 3: Solver efficiency (≤7 steps for range 1–100) ---
        try:
            max_steps = 0
            from unittest.mock import patch as _patch
            for test_secret in range(1, 101, 10):
                with _patch.object(ai, "_call_ollama", return_value="ok"):
                    steps = ai.auto_solve(test_secret, 1, 100, 10)
                max_steps = max(max_steps, len(steps))
            efficient = max_steps <= 7
            results.append({
                "id": "T3",
                "name": "Solver efficiency (≤7 steps for 1–100)",
                "passed": efficient,
                "note": f"Max steps observed: {max_steps}" + (" ✓" if efficient else " — expected ≤7"),
            })
        except Exception as e:
            results.append({"id": "T3", "name": "Solver efficiency", "passed": False, "note": str(e)})

        # --- TEST 4: AI hint contains direction keyword (requires Ollama) ---
        try:
            hint = ai.generate_hint(75, "Too High", 1, 100, 3)
            direction_ok = any(word in hint.lower() for word in ("lower", "less", "down", "smaller", "below"))
            results.append({
                "id": "T4",
                "name": "AI hint direction accuracy",
                "passed": direction_ok,
                "note": f'Hint: "{hint[:80]}..."' if len(hint) > 80 else f'Hint: "{hint}"',
            })
        except ConnectionError:
            results.append({
                "id": "T4",
                "name": "AI hint direction accuracy",
                "passed": False,
                "note": "Ollama not running — start with `ollama serve`",
            })
        except Exception as e:
            results.append({"id": "T4", "name": "AI hint direction accuracy", "passed": False, "note": str(e)})

        # --- TEST 5: AI hint fallback on connection error ---
        try:
            from unittest.mock import patch as _patch
            mock_err = ConnectionError("refused")
            with _patch.object(ai, "_call_ollama", side_effect=mock_err):
                fallback = ai.generate_hint(50, "Too Low", 1, 100, 2)
            fallback_ok = isinstance(fallback, str) and len(fallback) > 0
            results.append({
                "id": "T5",
                "name": "Hint fallback on connection error",
                "passed": fallback_ok,
                "note": f'Fallback: "{fallback}"',
            })
        except Exception as e:
            results.append({"id": "T5", "name": "Hint fallback on connection error", "passed": False, "note": str(e)})

        return results

    if st.button("▶ Run Reliability Tests"):
        with st.spinner("Running tests..."):
            test_results = _run_reliability_tests()

        passed = sum(1 for r in test_results if r["passed"])
        total = len(test_results)

        if passed == total:
            st.success(f"✅ {passed}/{total} tests passed")
        else:
            st.warning(f"⚠️ {passed}/{total} tests passed")

        for r in test_results:
            icon = "✅" if r["passed"] else "❌"
            with st.expander(f"{icon} {r['id']}: {r['name']}"):
                st.markdown(f"**Result:** {'PASS' if r['passed'] else 'FAIL'}")
                st.markdown(f"**Notes:** {r['note']}")

        logger.info("Reliability tests: %d/%d passed", passed, total)
