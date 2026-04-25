# 🎮 Game Glitch Investigator: The Impossible Guesser — AI Enhanced

## Original Project (Modules 1–3)

**Game Glitch Investigator** was a Streamlit number-guessing game intentionally seeded with six bugs (swapped hint messages, broken scoring, no input validation, broken New Game reset, etc.). The goal was to find and fix every bug using AI assistance (Claude), refactor the logic into `logic_utils.py`, and verify correctness with 11 pytest cases. All original files are preserved: `logic_utils.py` and `tests/test_game_logic.py` still pass.

---

## What's New: AI-Enhanced Features

The game now includes three AI-powered features, all running locally via **Ollama** — no API key, no account, no cost.

| Feature | What It Does | Advanced AI Type |
|---------|-------------|-----------------|
| 🤖 AI Hints | Ollama generates creative, contextual hints instead of plain "Too High/Too Low" | Agentic (LLM in the loop) |
| 🤖 Auto-Solver | An AI agent solves the puzzle step-by-step using Plan → Act → Check → Reflect | **Agentic Workflow** |
| 📊 Game Analysis | After each game, AI reviews your guesses and gives strategy coaching | Explain / Classify |
| 🧪 Reliability Tab | Built-in test suite verifies AI components with automated checks | **Reliability System** |

---

## System Diagram

```
User
 |
 v
[Streamlit UI — app.py]
 |          |             |           |
🎮 Play   🤖 Auto-Solver  📊 Analysis  🧪 Reliability
 |          |             |
 v          v             v
[logic_utils.py]    [ai_engine.py — AIEngine]
 parse_guess()           |
 check_guess()      generate_hint()  ──► [Ollama LLM (llama3.2)]
 update_score()     analyze_game()   ──► [Ollama LLM]
                    auto_solve()     ──► Plan→Act→Check→Reflect loop
                         |                 ↓ each step explained by LLM
                    [app.log]  rotating file — all events logged
                         |
                    [Reliability tab]
                      5 automated checks → pass/fail report
```

**Agentic Auto-Solver loop (one iteration):**
1. **PLAN** — choose midpoint of the current search range (binary search)
2. **ACT** — submit that guess
3. **CHECK** — compare to secret; narrow range up or down
4. **REFLECT** — Ollama generates a 1-2 sentence narration of the step
5. Repeat until `Win` or max attempts reached

---

## Setup Instructions

### Prerequisites
- **Python 3.12** (not 3.15 — numpy has no wheel for the 3.15 alpha)
- [Ollama](https://ollama.com/download) — free local LLM runner for Windows

### Steps

```bash
# 1. Install Ollama from https://ollama.com/download (auto-starts on Windows)

# 2. Download the model (one-time, ~2 GB)
ollama pull llama3.2

# 3. Clone the repo and enter the project folder
git clone <repo-url>
cd applied-ai-system-project/applied-ai-system-project

# 4. Install Python dependencies
py -3.12 -m pip install -r requirements.txt

# 5. (Optional) Edit .env to change model
copy .env.example .env   # then edit OLLAMA_MODEL if desired

# 6. Run the app
py -3.12 -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Run Tests (no Ollama needed)

```bash
py -3.12 -m pytest tests/ -v
```

---

## Sample Interactions

### 1. Play with AI Hints ON

Toggle **🤖 AI Hints** in the sidebar. After each wrong guess:

> Player guesses **73** (secret is 42)
>
> 🤖 *"Whoa, that's way too adventurous! Try aiming quite a bit lower — you've got this!"*

Without AI Hints, the same guess shows the plain message:
> 📉 Go LOWER!

### 2. Watch the Auto-Solver (Normal difficulty, secret = 42)

```
Step 1: guessed 50 — 📉 Too High
  Search range: 1–100 | AI: "Splitting the range in half, I land at 50 — but it's too high,
  so I cut my search space to the lower half: 1–49."

Step 2: guessed 25 — 📈 Too Low
  Search range: 1–49 | AI: "25 is my new midpoint, but too low — now I focus on 26–49."

Step 3: guessed 37 — 📈 Too Low
  Search range: 26–49 | ...

Step 4: guessed 43 — 📉 Too High
  Search range: 37–49 | ...

Step 5: guessed 40 — 📈 Too Low
  Search range: 37–42 | ...

Step 6: guessed 41 — 📈 Too Low
  Search range: 40–42 | ...

Step 7: guessed 42 — ✅ Win!
  AI: "Binary search is optimal — each guess cuts the possibilities in half!"
```

✅ Solved in 7 steps!

### 3. Game Analysis (after losing)

> Your guesses: 10 → 20 → 15 → 18 → 17
> Secret was: 17
>
> 🤖 *"You showed some binary search instinct by starting at 10 and jumping to 20, effectively
> bounding the range early. Your best guess was 18 — just one off! Next time, split remaining
> ranges more precisely: after learning the secret is between 15 and 20, try 17 or 18 first
> rather than 17 last."*

---

## Architecture Overview

| Component | File | Role |
|-----------|------|------|
| Game UI | `app.py` | 4-tab Streamlit app: Play, Auto-Solver, Analysis, Reliability |
| AI features | `ai_engine.py` | `generate_hint`, `auto_solve`, `analyze_game`, `_call_ollama` |
| Game logic | `logic_utils.py` | Original: `parse_guess`, `check_guess`, `update_score`, `get_range_for_difficulty` |
| Game tests | `tests/test_game_logic.py` | 11 original tests — all still passing |
| AI tests | `tests/test_ai_engine.py` | 11 unit tests — Ollama mocked, no live calls |
| Config | `.env` | `OLLAMA_MODEL`, `OLLAMA_HOST` |
| Log | `app.log` | Rotating file: all game events, hints, solver runs, errors |

---

## Design Decisions

**Why an agentic auto-solver instead of just showing an answer?**
The assignment requires a system that "plans and completes a step-by-step task." Binary search maps perfectly onto the Plan → Act → Check → Reflect agent loop. Each iteration is one agent step; the LLM narrates the reasoning, making the agentic pattern transparent to the user rather than a black box.

**Why Ollama instead of a cloud API?**
No API key, no rate limits, works offline. The trade-off is a ~2 GB one-time model download and somewhat slower generation vs. cloud APIs. For a class project that "runs correctly and reproducibly," removing external dependencies is the right call.

**Why keep AI Hints optional?**
Making AI hints a toggleable feature means the game is always playable, even if Ollama is not running. The `generate_hint` method also has a built-in fallback — it silently returns a simple hint string if the LLM call fails. This is the guardrail: the AI enhances the experience but never breaks it.

**Why not use the LLM to decide the guesses in auto_solve?**
Having the LLM pick guesses would introduce randomness, making the solver sometimes suboptimal or incorrect. Binary search is provably optimal for this problem. The LLM's role is *narration* (Reflect), not *strategy* (Plan). This is a deliberate separation of deterministic logic from generative language.

---

## Testing Summary

| Suite | Tests | Passed | What it covers |
|-------|-------|--------|---------------|
| Game logic (original) | 11 | 11 | All 6 original bugs verified fixed |
| AI engine (new) | 11 | 11 | Solver correctness, binary search, hints, analysis, error handling |
| **Total** | **22** | **22** | Run: `py -3.12 -m pytest tests/ -v` |

**In-app Reliability tab (5 live checks, no Ollama needed for T1–T3):**

| Test | What it checks | Requires Ollama |
|------|---------------|----------------|
| T1 | Auto-solver finds 5 different secrets | No |
| T2 | Every guess equals the exact midpoint | No |
| T3 | Solves 1–100 in ≤7 steps | No |
| T4 | AI hint mentions correct direction (lower/higher) | Yes |
| T5 | Hint fallback works when Ollama is unreachable | No |

---

## Reflection

Adding AI to a game taught me something important about where AI belongs in a system. The instinct is to let the AI *do everything*, but that leads to unpredictable behavior. The better pattern — learned by building the auto-solver — is to use deterministic logic for decisions and AI for communication. Binary search always finds the answer; the LLM makes the process feel natural and engaging.

The AI hints feature exposed a real design challenge: what happens when the AI fails? A game that crashes because Ollama isn't running is worse than a game with plain hints. Adding a fallback took ten lines of code but made the system robust to real-world conditions. That's the difference between AI that *seems* to work and AI that *proves* it works — which is what the Reliability tab is for.

The biggest surprise was how much the LLM's narration changed the *feel* of the auto-solver. Without it, watching a binary search run is just watching numbers. With it, each step reads like a decision being made. That's the value of generative AI in applications: not replacing logic, but making it legible.
