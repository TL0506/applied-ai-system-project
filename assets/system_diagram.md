# System Architecture Diagram

```mermaid
flowchart TD
    User([👤 Player])

    User -->|interacts with| UI["🖥️ Streamlit UI\n(app.py)"]

    UI --> T1["🎮 Play Tab"]
    UI --> T2["🤖 Auto-Solver Tab"]
    UI --> T3["📊 Analysis Tab"]
    UI --> T4["💡 Strategy Advisor Tab"]
    UI --> T5["🧪 Reliability Tab"]

    T1 -->|parse / validate| LU["logic_utils.py\nparse_guess()\ncheck_guess()\nupdate_score()"]
    T1 -->|AI hints toggle| AI["ai_engine.py\nAIEngine"]
    T2 --> AI
    T3 --> AI

    T4 -->|validate input| GG["game_guardrails.py\nQueryValidator"]
    T4 -->|retrieve + generate| SR["strategy_rag.py\nStrategyRAG"]
    SR -->|TF-IDF index| KB["📚 STRATEGY_KB\n(built-in corpus)"]
    SR -->|LLM call| OL

    AI -->|generate_hint| OL["🦙 Ollama LLM\n(llama3.2, local)"]
    AI -->|analyze_game| OL
    AI -->|auto_solve narration| OL

    T5 -->|5 automated checks| REL["Reliability Tests\nT1–T5 pass/fail report"]
    T5 -->|evaluation suite| SE["strategy_evaluator.py\nStrategyEvalSuite"]
    SE --> SR

    AI --> LOG["📄 app.log\nRotating file logger"]
    SR --> LOG

    subgraph Tests["🧪 Test Suite (33 tests, no Ollama needed)"]
        TG["test_game_logic.py\n11 tests"]
        TA["test_ai_engine.py\n11 tests"]
        TS["test_strategy_rag.py\n11 tests"]
    end
```

## Component Descriptions

| Component | File | Role |
|-----------|------|------|
| Streamlit UI | `app.py` | 5-tab app — Play, Auto-Solver, Analysis, Strategy Advisor, Reliability |
| AI Engine | `ai_engine.py` | `generate_hint`, `auto_solve` (Plan→Act→Check→Reflect), `analyze_game` |
| Game Logic | `logic_utils.py` | `parse_guess`, `check_guess`, `update_score`, `get_range_for_difficulty` |
| Strategy RAG | `strategy_rag.py` | TF-IDF retrieval over built-in KB → Ollama-generated grounded advice |
| Game Guardrails | `game_guardrails.py` | Input validation (query length), hint output direction validation |
| Strategy Evaluator | `strategy_evaluator.py` | 5-case eval suite: keyword + citation checks on RAG advice |
| LLM | Ollama (llama3.2) | Local inference — hints, solver narration, game analysis, strategy advice |
| Logger | `app.log` | Rotating file — all game events, AI calls, errors |

## Agentic Auto-Solver Loop

```
┌─────────────────────────────────────────────┐
│  for each step until Win or max_attempts:   │
│                                             │
│  1. PLAN   → pick midpoint(low, high)       │
│  2. ACT    → submit guess                   │
│  3. CHECK  → compare to secret, narrow range│
│  4. REFLECT→ Ollama narrates the step       │
└─────────────────────────────────────────────┘
```

## RAG Strategy Advisor Flow

```
Player Question
      │
      ▼
QueryValidator.validate_query()   ← guardrail: empty / too short / too long
      │
      ▼
StrategyRAG.retrieve()            ← TF-IDF cosine similarity over STRATEGY_KB
      │  top-3 chunks
      ▼
StrategyRAG.generate_advice()     ← builds [STRATEGY N]-cited prompt → Ollama
      │                              fallback: returns raw chunk text if Ollama down
      ▼
Advice + confidence score + expandable source excerpts
```
