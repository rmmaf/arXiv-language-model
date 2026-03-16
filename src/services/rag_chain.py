"""RAG orchestration: query -> embed -> hybrid search -> PDF extraction -> re-rank -> LLM answer."""

import asyncio
import logging
import time
from typing import Any

import numpy as np
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.elastic import ElasticClient
from src.core.llm import LLMManager
from src.services.pdf_reader import AsyncPDFReader

logger = logging.getLogger(__name__)

RAG_PROMPT = PromptTemplate.from_template(
    "You are a research assistant. Use ONLY the context below to answer the question.\n"
    "If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class RAGService:
    """End-to-end hybrid RAG pipeline over arXiv papers."""

    def __init__(
        self,
        elastic: ElasticClient,
        llm_manager: LLMManager,
    ) -> None:
        self._elastic = elastic
        self._llm = llm_manager
        self._encoder = SentenceTransformer(
            settings.embedding_model_name, device=settings.embedding_device
        )
        self._pdf_reader = AsyncPDFReader()

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(texts, show_progress_bar=False)

    def _rerank_chunks(
        self,
        query_vector: np.ndarray,
        chunks: list[str],
        top_n: int = 5,
    ) -> list[str]:
        """Re-rank chunks by cosine similarity to the query vector."""
        if not chunks:
            return []

        chunk_vectors = self._embed(chunks)

        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        c_norms = chunk_vectors / (
            np.linalg.norm(chunk_vectors, axis=1, keepdims=True) + 1e-10
        )
        similarities = c_norms @ q_norm

        ranked_indices = np.argsort(similarities)[::-1][:top_n]
        return [chunks[i] for i in ranked_indices]

    async def ask(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run the full RAG pipeline and return the answer with sources."""
        start = time.perf_counter()
        k = top_k or settings.top_k_results

        logger.info("RAG query: %s", question[:120])

        query_vector = self._embed([question])[0]

        search_results = await self._elastic.hybrid_search(
            query_text=question,
            query_vector=query_vector.tolist(),
            top_k=k,
        )

        if not search_results:
            return {
                "answer": "No relevant papers found for your question.",
                "sources": [],
                "processing_time_seconds": time.perf_counter() - start,
            }

        paper_ids = [r["paper_id"] for r in search_results]
        logger.info("Downloading PDFs for papers: %s", paper_ids)

        chunks_by_paper = await self._pdf_reader.process_papers(paper_ids)

        all_chunks: list[str] = []
        for pid in paper_ids:
            all_chunks.extend(chunks_by_paper.get(pid, []))

        if not all_chunks:
            context_lines = []
            for r in search_results:
                context_lines.append(f"Title: {r['title']}\nAbstract: {r['abstract']}")
            context = "\n\n---\n\n".join(context_lines)
            logger.warning("No PDF text extracted; falling back to abstracts")
        else:
            top_chunks = self._rerank_chunks(query_vector, all_chunks, top_n=5)
            context = "\n\n---\n\n".join(top_chunks)

        prompt_text = RAG_PROMPT.format(context=context, question=question)
        logger.debug("Prompt length: %d characters", len(prompt_text))

        try:
            answer = await asyncio.wait_for(
                self._llm.pipeline.ainvoke(prompt_text),
                timeout=settings.llm_timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            logger.error("LLM inference timed out after %.1fs", elapsed)
            return {
                "answer": (
                    "The model took too long to generate a response. "
                    "Try a more specific question or reduce top_k."
                ),
                "sources": [
                    {"paper_id": r["paper_id"], "title": r["title"], "score": r["score"]}
                    for r in search_results
                ],
                "processing_time_seconds": round(elapsed, 3),
            }

        sources = [
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "score": r["score"],
            }
            for r in search_results
        ]

        elapsed = time.perf_counter() - start
        logger.info("RAG pipeline completed in %.2f seconds", elapsed)

        return {
            "answer": answer.strip() if isinstance(answer, str) else str(answer).strip(),
            "sources": sources,
            "processing_time_seconds": round(elapsed, 3),
        }
