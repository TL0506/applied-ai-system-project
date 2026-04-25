import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama as _ollama_module
except ImportError:
    _ollama_module = None  # allow import without ollama installed (tests mock it)

# Module-level name so tests can patch 'rag_engine.ollama'
ollama = _ollama_module

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200
TOP_K_CHUNKS = 4
MAX_PROMPT_CHARS = 60_000
DEFAULT_MODEL = "llama3.2"


@dataclass
class Chunk:
    index: int
    text: str
    start_char: int
    end_char: int


@dataclass
class RetrievalResult:
    chunks: list
    scores: list


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: list
    retrieval_scores: list
    prompt_chars: int
    confidence_score: float


class RAGEngine:

    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.chunks: list = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._is_indexed: bool = False
        logger.info("RAGEngine initialized: model=%s host=%s", model, host)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> list:
        text = text.strip()
        if not text:
            raise ValueError("Cannot chunk empty text.")

        chunks = []
        step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE_CHARS, len(text))

            # Snap end to nearest word boundary (scan back up to 50 chars)
            if end < len(text):
                snap = text.rfind(" ", max(start, end - 50), end)
                if snap > start:
                    end = snap

            chunk_text_str = text[start:end].strip()
            if chunk_text_str:
                chunks.append(
                    Chunk(index=idx, text=chunk_text_str, start_char=start, end_char=end)
                )
                idx += 1

            if end >= len(text):
                break

            # Advance with overlap, snapping start to next word boundary
            start = end - CHUNK_OVERLAP_CHARS
            snap = text.find(" ", start, start + 50)
            if snap != -1:
                start = snap + 1

        logger.info("Chunked text into %d chunks", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, text: str) -> int:
        self.chunks = self.chunk_text(text)
        chunk_texts = [c.text for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
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

    def retrieve(self, query: str, top_k: int = TOP_K_CHUNKS) -> RetrievalResult:
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        query_vec = self.vectorizer.transform([query])

        # All-stop-word query produces a zero vector — return empty result
        if query_vec.nnz == 0:
            logger.warning("Zero-norm TF-IDF vector for query: '%s'", query)
            return RetrievalResult(chunks=[], scores=[])

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
        return RetrievalResult(chunks=top_chunks, scores=top_scores)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, retrieval: RetrievalResult) -> RAGResponse:
        if not retrieval.chunks:
            return RAGResponse(
                answer=(
                    "I couldn't find relevant information in the document for your question. "
                    "Try rephrasing, or check that your question relates to the uploaded document."
                ),
                retrieved_chunks=[],
                retrieval_scores=[],
                prompt_chars=0,
                confidence_score=0.0,
            )

        prompt = self._build_qa_prompt(query, retrieval.chunks)
        confidence = retrieval.scores[0] if retrieval.scores else 0.0

        logger.info(
            "LLM call | model=%s | prompt_chars=%d | confidence=%.3f",
            self.model,
            len(prompt),
            confidence,
        )

        response = self._call_ollama(prompt, max_tokens=1024)

        logger.info("LLM response: %d chars", len(response))
        return RAGResponse(
            answer=response,
            retrieved_chunks=retrieval.chunks,
            retrieval_scores=retrieval.scores,
            prompt_chars=len(prompt),
            confidence_score=confidence,
        )

    def query(self, question: str, top_k: int = TOP_K_CHUNKS) -> RAGResponse:
        retrieval = self.retrieve(question, top_k=top_k)
        return self.generate_answer(question, retrieval)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize(self, text: str, style: str = "concise") -> str:
        chunks = self.chunk_text(text)
        logger.info("Summarizing: %d chunks, style=%s", len(chunks), style)

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = (
                "Summarize the following passage in 2-3 sentences. "
                "Be concise and capture only the key points.\n\n"
                f"Passage:\n{chunk.text}"
            )
            summary = self._call_ollama(prompt, max_tokens=256)
            chunk_summaries.append(summary.strip())
            logger.info("Summarized chunk %d/%d", i + 1, len(chunks))

        combined = "\n\n".join(chunk_summaries)
        final_prompt = self._build_summary_prompt(combined, style)
        final_summary = self._call_ollama(final_prompt, max_tokens=1024)

        logger.info("Summarization complete: %d chars", len(final_summary))
        return final_summary.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, max_tokens: int = 1024) -> str:
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
            # Support both attribute-style (object) and dict-style responses
            if isinstance(response, dict):
                return response["message"]["content"]
            return response.message.content
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

    def _build_qa_prompt(self, query: str, chunks: list) -> str:
        source_blocks = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            header = f"[SOURCE {i}]\n"
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
            "You are a helpful document assistant. Answer the question using ONLY the "
            "information in the sources below. Cite sources inline as [SOURCE N]. "
            "If the sources do not contain enough information, say so clearly.\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

    def _build_summary_prompt(self, combined_chunk_summaries: str, style: str) -> str:
        style_instructions = {
            "concise": "Write a concise summary in 3-5 sentences capturing the main points.",
            "detailed": "Write a detailed summary covering all key topics and important details.",
            "bullet_points": "Write a bulleted list where each bullet is one key point from the document.",
        }
        instruction = style_instructions.get(style, style_instructions["concise"])

        return (
            "Below are summaries of sections from a document. "
            "Synthesize them into a single coherent summary.\n\n"
            f"{instruction}\n\n"
            f"Section summaries:\n{combined_chunk_summaries}\n\n"
            "Final summary:"
        )
