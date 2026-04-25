import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama as _ollama_module
except ImportError:
    _ollama_module = None

ollama = _ollama_module

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 600
CHUNK_OVERLAP_CHARS = 80
TOP_K_CHUNKS = 3
MAX_PROMPT_CHARS = 8_000
DEFAULT_MODEL = "llama3.2"

# Built-in game strategy knowledge base — the RAG corpus.
# Covers 5 topic areas so chunking produces distinct retrievable sections.
STRATEGY_KB = """
BINARY SEARCH AND THE MIDPOINT STRATEGY

The most efficient strategy for a number guessing game is binary search. For a range of 1 to 100, the optimal first guess is always 50 — the exact midpoint. Binary search works by halving the remaining search space with every guess. After learning whether the answer is higher or lower, you pick the midpoint of the surviving range. This guarantees that you find any number between 1 and 100 in at most 7 guesses. Each midpoint guess eliminates half of the remaining possibilities, which is mathematically optimal. No other strategy can consistently beat binary search for this type of problem.

NARROWING THE RANGE AFTER FEEDBACK

Every piece of feedback cuts the search space. When a guess is too low, every number at or below that guess is eliminated — your next target must be strictly higher. When a guess is too high, every number at or above it is gone — go lower. After a "too low" result on guess 50 in a 1–100 game, the new range is 51–100 and the correct next guess is the midpoint: 75. After a "too high" result on 75, the new range is 51–74 and the midpoint is 62. Always compute the midpoint of the current range as (low + high) divided by 2, rounded down. If the secret is between 20 and 80 and your guess of 50 was too low, the next range is 51–80 and you should guess 65 — above 51, not below it. Follow the feedback precisely.

URGENCY TACTICS WITH FEW ATTEMPTS REMAINING

When only one or two attempts remain, every guess must be deliberate. Guessing randomly wastes precious turns. Narrow the remaining range as tightly as possible before committing. If you have one guess left and the range is 20–35, the best guess is the midpoint 27. Avoid guessing extremes (like 20 or 35) unless the range has collapsed to a single value. Stay calm, ignore previous emotional choices, and apply the midpoint rule. Each remaining guess must eliminate the maximum number of possibilities. Do not repeat a previous guess — it provides no new information.

COMMON MISTAKES AND HOW TO AVOID THEM

The most common mistake is guessing randomly without a system. Random guessing provides no guarantee and often repeats similar values. Starting at 1 is the second biggest mistake — it eliminates only one value from a 100-number range, wasting your first guess entirely. A smarter opening guess of 50 eliminates 50 values at once. Another common error is ignoring feedback: after being told "too high," some players still guess a higher number on the next turn. Always honor the feedback. Repeating a previous guess is also wasteful — it provides zero new information. Keep track of your high and low boundaries after each result.

SCORING AND EFFICIENCY

Each wrong guess costs 5 points. Winning with fewer guesses earns a higher score. Binary search minimizes wrong guesses, which directly maximizes your score. In a 1–100 range with 8 attempts, binary search comfortably wins in at most 7 steps. Starting with 50 and following feedback precisely will reliably produce scores well above 50. Every extra guess beyond the minimum costs you 5 points, so efficiency and binary search are directly linked to a good final score.
""".strip()


@dataclass
class StrategyChunk:
    index: int
    text: str
    start_char: int
    end_char: int


@dataclass
class StrategyRetrievalResult:
    chunks: list
    scores: list


@dataclass
class StrategyResponse:
    advice: str
    retrieved_chunks: list
    retrieval_scores: list
    confidence_score: float


class StrategyRAG:

    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.chunks: list = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._is_indexed: bool = False
        logger.info("StrategyRAG initialized: model=%s", model)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_kb(self, text: str) -> list:
        text = text.strip()
        if not text:
            raise ValueError("Cannot chunk empty text.")

        chunks = []
        step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE_CHARS, len(text))

            if end < len(text):
                snap = text.rfind(" ", max(start, end - 50), end)
                if snap > start:
                    end = snap

            chunk_text_str = text[start:end].strip()
            if chunk_text_str:
                chunks.append(
                    StrategyChunk(index=idx, text=chunk_text_str, start_char=start, end_char=end)
                )
                idx += 1

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP_CHARS
            snap = text.find(" ", start, start + 50)
            if snap != -1:
                start = snap + 1

        logger.info("Chunked KB into %d chunks", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self) -> int:
        self.chunks = self._chunk_kb(STRATEGY_KB)
        chunk_texts = [c.text for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5_000,
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        self._is_indexed = True

        logger.info(
            "TF-IDF index built: %d chunks, vocab=%d",
            len(self.chunks),
            len(self.vectorizer.vocabulary_),
        )
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K_CHUNKS) -> StrategyRetrievalResult:
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        query_vec = self.vectorizer.transform([query])

        if query_vec.nnz == 0:
            logger.warning("Zero-norm TF-IDF vector for query: '%s'", query)
            return StrategyRetrievalResult(chunks=[], scores=[])

        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        top_chunks = [self.chunks[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        logger.info(
            "Retrieved top-%d chunks | query='%s...' | top_score=%.3f",
            top_k,
            query[:50],
            top_scores[0] if top_scores else 0.0,
        )
        return StrategyRetrievalResult(chunks=top_chunks, scores=top_scores)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_advice(self, query: str, retrieval: StrategyRetrievalResult) -> StrategyResponse:
        if not retrieval.chunks:
            return StrategyResponse(
                advice=(
                    "I couldn't find relevant strategy tips for that question. "
                    "Try asking about binary search, narrowing the range, or first-guess strategy."
                ),
                retrieved_chunks=[],
                retrieval_scores=[],
                confidence_score=0.0,
            )

        prompt = self._build_strategy_prompt(query, retrieval.chunks)
        confidence = retrieval.scores[0] if retrieval.scores else 0.0

        logger.info(
            "LLM call | model=%s | prompt_chars=%d | confidence=%.3f",
            self.model,
            len(prompt),
            confidence,
        )

        try:
            response = self._call_ollama(prompt, max_tokens=256)
            logger.info("LLM response: %d chars", len(response))
        except Exception as e:
            logger.warning("Ollama unavailable for strategy advice: %s", e)
            response = " ".join(c.text for c in retrieval.chunks[:2])

        return StrategyResponse(
            advice=response,
            retrieved_chunks=retrieval.chunks,
            retrieval_scores=retrieval.scores,
            confidence_score=confidence,
        )

    def advise(self, query: str, top_k: int = TOP_K_CHUNKS) -> StrategyResponse:
        retrieval = self.retrieve(query, top_k=top_k)
        return self.generate_advice(query, retrieval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_strategy_prompt(self, query: str, chunks: list) -> str:
        source_blocks = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            header = f"[STRATEGY {i}]\n"
            body = chunk.text
            block = header + body

            if total_chars + len(block) > MAX_PROMPT_CHARS:
                remaining = MAX_PROMPT_CHARS - total_chars - len(header)
                if remaining > 100:
                    source_blocks.append(header + body[:remaining] + "...")
                break

            source_blocks.append(block)
            total_chars += len(block)

        sources_text = "\n\n".join(source_blocks)
        return (
            "You are an expert strategy coach for a 1–100 number guessing game.\n"
            "Answer the player's question using ONLY the strategy excerpts below.\n"
            "Cite each excerpt you use as [STRATEGY N]. Be concise (2-3 sentences).\n"
            "If the excerpts do not contain enough information, say so clearly.\n\n"
            f"STRATEGY EXCERPTS:\n{sources_text}\n\n"
            f"PLAYER QUESTION: {query}\n\n"
            "ADVICE:"
        )

    def _call_ollama(self, prompt: str, max_tokens: int = 256) -> str:
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
