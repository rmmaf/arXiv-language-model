"""RAG orchestration: query -> embed -> hybrid search -> PDF extraction -> re-rank -> LLM answer."""

import asyncio
import logging
import math
import time
from typing import Any

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

    @staticmethod
    def _adaptive_chunk_size(active_tenants: int) -> int:
        """Reduce chunk size as more tenants are active to lower LLM pressure."""
        base = settings.base_chunk_size
        minimum = settings.min_chunk_size
        if active_tenants <= 1:
            return base
        factor = 1.0 / math.log2(active_tenants + 1)
        return max(minimum, int(base * factor))

    def _build_pdf_reader(self, active_tenants: int) -> AsyncPDFReader:
        chunk_size = self._adaptive_chunk_size(active_tenants)
        return AsyncPDFReader(
            splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

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
        tenant_id: str,
        top_k: int | None = None,
        active_tenant_count: int = 1,
    ) -> dict[str, Any]:
        """Run the full RAG pipeline scoped to *tenant_id*."""
        start = time.perf_counter()
        k = top_k or settings.top_k_results

        logger.info("RAG query [tenant=%s]: %s", tenant_id, question[:120])

        # #region agent log
        import json as _json, time as _time
        with open("data/debug-36bb97.log", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"sessionId":"36bb97","hypothesisId":"H2","location":"rag_chain.py:ask","message":"RAG ask called","data":{"tenant_id":tenant_id,"question_prefix":question[:80],"top_k":k},"timestamp":int(_time.time()*1000)}) + "\n")
        # #endregion

        query_vector = self._embed([question])[0]

        search_results = await self._elastic.hybrid_search(
            query_text=question,
            query_vector=query_vector.tolist(),
            tenant_id=tenant_id,
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

        pdf_reader = self._build_pdf_reader(active_tenant_count)
        chunks_by_paper = await pdf_reader.process_papers(paper_ids)

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
