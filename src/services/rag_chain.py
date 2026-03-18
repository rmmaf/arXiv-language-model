"""RAG orchestration.

query -> embed -> hybrid search -> PDF extraction ->
re-rank -> LLM answer.
"""

import asyncio
import logging
import math
import re
import threading
import time
from functools import partial
from typing import Any

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.conversation import ConversationStore
from src.core.elastic import ElasticClient
from src.core.llm import LLMManager
from src.services.pdf_reader import AsyncPDFReader

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4

_CUSTOM_DOC_PATTERN = re.compile(
    r"(pdf|documento|arquivo|anexo|enviado|uploaded|attached"
    r"|file|deste|desse|do\s+que\s+enviei|em\s+anexo)",
    re.IGNORECASE,
)

RAG_PROMPT = PromptTemplate.from_template(
    "You are a research assistant specialized in academic papers. "
    "Answer the question using the context below.\n\n"
    "Guidelines:\n"
    "- Synthesize information across multiple context passages when needed.\n"
    "- For comparison or analytical questions, identify similarities and "
    "differences between papers/findings even if the context does not "
    "state them explicitly.\n"
    "- Ground every claim in the provided context. Cite paper titles or "
    "authors when available.\n"
    "- If the context only partially addresses the question, answer what "
    "you can and clearly state what information is missing.\n"
    "- Do NOT fabricate facts not supported by the context.\n"
    "- NEVER refer to 'the context' or 'the provided context' in your "
    "answer. Present information naturally as if you know it, citing "
    "specific paper titles or authors instead.\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{chat_history}\n\n"
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
        """Reduce chunk size as tenant count grows."""
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

    @staticmethod
    def _detect_custom_intent(question: str) -> bool:
        """Return True when the question explicitly references an
        attached / uploaded document."""
        return bool(_CUSTOM_DOC_PATTERN.search(question))

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(texts, show_progress_bar=False)

    def _rerank_chunks(
        self,
        query_vector: np.ndarray,
        chunks: list[str],
        top_n: int = 5,
        chunk_is_custom: list[bool] | None = None,
        boost_factor: float = 1.0,
        reserved_custom_slots: int = 0,
    ) -> list[str]:
        """Re-rank *chunks* by cosine similarity to *query_vector*.

        When *chunk_is_custom* is provided the scores of custom chunks
        are multiplied by *boost_factor*.  If *reserved_custom_slots*
        is greater than zero, that many top-N positions are guaranteed
        for the highest-scoring custom chunks (when available).
        """
        if not chunks:
            return []

        chunk_vectors = self._embed(chunks)

        q_norm = query_vector / (
            np.linalg.norm(query_vector) + 1e-10
        )
        c_norms = chunk_vectors / (
            np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
            + 1e-10
        )
        similarities = c_norms @ q_norm

        if chunk_is_custom and boost_factor > 1.0:
            mask = np.array(chunk_is_custom, dtype=np.float64)
            similarities = similarities * (
                1.0 + mask * (boost_factor - 1.0)
            )

        if reserved_custom_slots > 0 and chunk_is_custom:
            custom_idx = [
                i for i, c in enumerate(chunk_is_custom) if c
            ]
            custom_ranked = sorted(
                custom_idx,
                key=lambda i: similarities[i],
                reverse=True,
            )
            reserved = custom_ranked[:reserved_custom_slots]
            reserved_set = set(reserved)

            remaining = sorted(
                (
                    i for i in range(len(chunks))
                    if i not in reserved_set
                ),
                key=lambda i: similarities[i],
                reverse=True,
            )
            fill = remaining[: top_n - len(reserved)]

            final = reserved + fill
            final.sort(
                key=lambda i: similarities[i], reverse=True,
            )
            return [chunks[i] for i in final]

        ranked_indices = np.argsort(similarities)[::-1][:top_n]
        return [chunks[i] for i in ranked_indices]

    def _build_document_search_vector(
        self,
        question_vector: np.ndarray,
        doc_chunks: list[dict[str, Any]],
    ) -> np.ndarray:
        """Blend question embedding with mean document embedding.

        The document content is weighted by ``custom_content_weight``
        so that the arXiv vector search focuses on the document's topic
        rather than the literal words in the user's question.
        """
        contents = [c["content"] for c in doc_chunks]
        if not contents:
            return question_vector

        chunk_vectors = self._embed(contents)
        doc_vector = np.mean(chunk_vectors, axis=0)

        w = settings.custom_content_weight
        blended = w * doc_vector + (1.0 - w) * question_vector
        blended = blended / (np.linalg.norm(blended) + 1e-10)

        logger.info(
            "Built document search vector: content_weight=%.2f, "
            "doc_chunks=%d",
            w, len(contents),
        )
        return blended

    @staticmethod
    def _format_chat_history(history: list[dict[str, str]]) -> str:
        """Format recent chat history turns into a readable string."""
        if not history:
            return "(none)"
        recent = history[-(MAX_HISTORY_TURNS * 2):]
        lines: list[str] = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    async def _fetch_custom_chunks(
        self,
        query_vector: np.ndarray,
        tenant_id: str,
        document_ids: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Retrieve chunks and sources from custom documents."""
        results = await self._elastic.search_custom_documents(
            query_vector=query_vector.tolist(),
            tenant_id=tenant_id,
            document_ids=document_ids,
            top_k=10,
        )
        chunks = [r["content"] for r in results]
        seen: dict[str, dict[str, Any]] = {}
        for r in results:
            did = r["document_id"]
            if did not in seen:
                seen[did] = {
                    "paper_id": did,
                    "title": r["filename"],
                    "score": r["score"],
                    "source_type": "custom_upload",
                }
        return chunks, list(seen.values())

    async def ask(
        self,
        question: str,
        tenant_id: str,
        conversation_store: ConversationStore,
        top_k: int | None = None,
        active_tenant_count: int = 1,
        conversation_id: str | None = None,
        fetch_new_papers: bool = True,
        custom_document_ids: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run the RAG pipeline scoped to *tenant_id*.

        When *conversation_id* is provided and *fetch_new_papers* is
        False the expensive search + PDF download steps are skipped and
        the stored context is reused for a follow-up question.

        When *custom_document_ids* is provided, chunks from those
        uploaded documents are merged with arXiv results before
        re-ranking.

        When *cancel_event* is provided, it is forwarded to the LLM
        ``generate()`` call so that GPU inference can be interrupted
        at the token level.

        Supports asyncio cancellation -- CancelledError is caught at
        key checkpoints so partial state can be persisted.
        """
        start = time.perf_counter()
        k = top_k or settings.top_k_results

        logger.info("RAG query [tenant=%s]: %s", tenant_id, question[:120])

        existing_conv = (
            await conversation_store.get(conversation_id)
            if conversation_id else None
        )

        chat_history = (
            await conversation_store.get_chat_history(conversation_id)
            if existing_conv
            else []
        )
        chat_history_str = self._format_chat_history(chat_history)

        if existing_conv and not fetch_new_papers and not custom_document_ids:
            context = existing_conv["context"]
            sources = existing_conv["sources"]
            logger.info(
                "Reusing stored context for conversation %s",
                conversation_id,
            )
        else:
            query_vector = self._embed([question])[0]

            arxiv_chunks: list[str] = []
            sources: list[dict[str, Any]] = []
            search_results: list[dict[str, Any]] = []
            custom_chunks: list[str] = []

            search_vector = query_vector

            if custom_document_ids:
                raw_doc_chunks = (
                    await self._elastic.get_custom_document_chunks(
                        tenant_id=tenant_id,
                        document_ids=custom_document_ids,
                    )
                )
                if raw_doc_chunks:
                    search_vector = (
                        self._build_document_search_vector(
                            query_vector, raw_doc_chunks,
                        )
                    )
                    custom_chunks = [
                        f"[Source: {c['filename']}]\n{c['content']}"
                        for c in raw_doc_chunks
                    ]
                    seen: dict[str, dict[str, Any]] = {}
                    for c in raw_doc_chunks:
                        did = c["document_id"]
                        if did not in seen:
                            seen[did] = {
                                "paper_id": did,
                                "title": c["filename"],
                                "score": 0.0,
                                "source_type": "custom_upload",
                            }
                    sources.extend(list(seen.values()))

            if fetch_new_papers or not existing_conv:
                await asyncio.sleep(0)  # cancellation checkpoint

                search_results = await self._elastic.hybrid_search(
                    query_text=question,
                    query_vector=search_vector.tolist(),
                    tenant_id=tenant_id,
                    top_k=k,
                    use_text_search=not bool(custom_chunks),
                )

                if search_results:
                    paper_ids = [r["paper_id"] for r in search_results]
                    logger.info("Downloading PDFs for papers: %s", paper_ids)

                    await asyncio.sleep(0)  # cancellation checkpoint

                    pdf_reader = self._build_pdf_reader(active_tenant_count)
                    chunks_by_paper = await pdf_reader.process_papers(
                        paper_ids,
                    )

                    title_by_pid = {
                        r["paper_id"]: r["title"]
                        for r in search_results
                    }
                    for pid in paper_ids:
                        title = title_by_pid.get(pid, pid)
                        for chunk in chunks_by_paper.get(pid, []):
                            arxiv_chunks.append(
                                f"[Paper: {title}]\n{chunk}"
                            )

                    if not arxiv_chunks:
                        context_lines = []
                        for r in search_results:
                            context_lines.append(
                                f"Title: {r['title']}\n"
                                f"Abstract: {r['abstract']}"
                            )
                        arxiv_chunks = context_lines
                        logger.warning(
                            "No PDF text extracted; "
                            "falling back to abstracts",
                        )

                    sources.extend([
                        {
                            "paper_id": r["paper_id"],
                            "title": r["title"],
                            "score": r["score"],
                            "source_type": "arxiv",
                        }
                        for r in search_results
                    ])

            all_chunks = arxiv_chunks + custom_chunks

            if not all_chunks and not search_results:
                conv_id = (
                    conversation_id
                    or await conversation_store.create(
                        tenant_id=tenant_id,
                    )
                )
                no_results_answer = (
                    "No relevant papers or documents "
                    "found for your question."
                )
                await conversation_store.add_message(
                    conv_id, "assistant", no_results_answer,
                )
                return {
                    "answer": no_results_answer,
                    "sources": [],
                    "processing_time_seconds": (
                        time.perf_counter() - start
                    ),
                    "conversation_id": conv_id,
                }

            if all_chunks:
                chunk_is_custom = (
                    [False] * len(arxiv_chunks)
                    + [True] * len(custom_chunks)
                )
                has_custom = bool(
                    custom_document_ids and custom_chunks
                )
                explicit_intent = (
                    self._detect_custom_intent(question)
                )

                if has_custom and explicit_intent:
                    boost = settings.custom_boost_factor
                    reserved = settings.custom_reserved_slots
                elif has_custom:
                    boost = settings.custom_mild_boost_factor
                    reserved = 0
                else:
                    boost = 1.0
                    reserved = 0

                top_chunks = self._rerank_chunks(
                    search_vector,
                    all_chunks,
                    top_n=5,
                    chunk_is_custom=chunk_is_custom,
                    boost_factor=boost,
                    reserved_custom_slots=reserved,
                )
                context = "\n\n---\n\n".join(top_chunks)
            else:
                context = ""

            if existing_conv:
                await conversation_store.update_context(
                    conversation_id, context, sources,
                )

        prompt_text = RAG_PROMPT.format(
            context=context, chat_history=chat_history_str, question=question
        )
        logger.debug("Prompt length: %d characters", len(prompt_text))

        await asyncio.sleep(0)  # cancellation checkpoint before LLM

        event = cancel_event or threading.Event()
        loop = asyncio.get_running_loop()

        try:
            answer = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(
                        self._llm.generate, prompt_text, event,
                    ),
                ),
                timeout=settings.llm_timeout,
            )
        except asyncio.CancelledError:
            event.set()
            logger.info(
                "RAG pipeline cancelled during LLM inference",
            )
            if conversation_id:
                await conversation_store.add_message(
                    conversation_id, "assistant",
                    "(Response cancelled by user)",
                )
            raise
        except asyncio.TimeoutError:
            event.set()
            elapsed = time.perf_counter() - start
            logger.error(
                "LLM inference timed out after %.1fs", elapsed,
            )
            timeout_answer = (
                "The model took too long to generate a response. "
                "Try a more specific question or reduce top_k."
            )
            conv_id = (
                conversation_id
                or await conversation_store.create(
                    tenant_id=tenant_id,
                    context=context,
                    sources=sources,
                )
            )
            await conversation_store.add_message(
                conv_id, "assistant", timeout_answer,
            )
            return {
                "answer": timeout_answer,
                "sources": sources,
                "processing_time_seconds": round(elapsed, 3),
                "conversation_id": conv_id,
            }

        if event.is_set():
            logger.info(
                "LLM generation was interrupted by cancel event",
            )
            if conversation_id:
                await conversation_store.add_message(
                    conversation_id, "assistant",
                    "(Response cancelled by user)",
                )
            raise asyncio.CancelledError

        answer_text = (
            answer.strip() if isinstance(answer, str)
            else str(answer).strip()
        )

        if existing_conv:
            conv_id = conversation_id
        else:
            conv_id = conversation_id or await conversation_store.create(
                tenant_id=tenant_id, context=context, sources=sources,
            )

        await conversation_store.add_message(conv_id, "assistant", answer_text)

        msg_count = await conversation_store.get_message_count(conv_id)
        if msg_count <= 2:
            auto_title = question[:60] + ("..." if len(question) > 60 else "")
            await conversation_store.update_title(conv_id, auto_title)

        elapsed = time.perf_counter() - start
        logger.info("RAG pipeline completed in %.2f seconds", elapsed)

        return {
            "answer": answer_text,
            "sources": sources,
            "processing_time_seconds": round(elapsed, 3),
            "conversation_id": conv_id,
        }
