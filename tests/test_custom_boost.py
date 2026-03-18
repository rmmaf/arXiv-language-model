"""Tests for custom-document boost in the RAG re-ranking pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.services.rag_chain import RAGService, _CUSTOM_DOC_PATTERN


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_service() -> RAGService:
    """Build a RAGService with mocked dependencies."""
    elastic = MagicMock()
    llm = MagicMock()
    with patch(
        "src.services.rag_chain.SentenceTransformer",
    ):
        service = RAGService(elastic=elastic, llm_manager=llm)
    return service


def _deterministic_embed(texts: list[str]) -> np.ndarray:
    """Return a reproducible embedding per chunk text.

    Arxiv chunks get vectors pointing in the query direction
    (high similarity); custom chunks get orthogonal vectors
    (low similarity) unless boosted.
    """
    vecs = []
    for t in texts:
        if t.startswith("arxiv"):
            vecs.append([0.9, 0.1, 0.0])
        elif t.startswith("custom"):
            vecs.append([0.1, 0.9, 0.0])
        else:
            vecs.append([0.5, 0.5, 0.0])
    return np.array(vecs, dtype=np.float64)


QUERY_VECTOR = np.array([1.0, 0.0, 0.0], dtype=np.float64)


# ------------------------------------------------------------------ #
#  1. Intent detection
# ------------------------------------------------------------------ #

class TestDetectCustomIntent:
    """Verify _detect_custom_intent matches the expected phrases."""

    @pytest.mark.parametrize(
        "question",
        [
            "me passe os estudos do pdf em anexo",
            "sobre o documento enviado",
            "resuma o arquivo que enviei",
            "from the attached file",
            "summarize the uploaded pdf",
            "o que diz deste documento?",
            "compare desse arquivo com transformers",
        ],
    )
    def test_positive_detection(self, question: str) -> None:
        assert RAGService._detect_custom_intent(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "what is the attention mechanism?",
            "explique redes neurais convolucionais",
            "list recent papers on reinforcement learning",
        ],
    )
    def test_negative_detection(self, question: str) -> None:
        assert RAGService._detect_custom_intent(question) is False

    def test_pattern_is_case_insensitive(self) -> None:
        assert _CUSTOM_DOC_PATTERN.flags & 2  # re.IGNORECASE


# ------------------------------------------------------------------ #
#  2. Boost with explicit intent (reserved slots)
# ------------------------------------------------------------------ #

class TestBoostWithExplicitIntent:
    """With boost + reserved slots, custom chunks must appear."""

    def test_reserved_slots_include_custom_chunks(self) -> None:
        service = _make_service()
        service._embed = _deterministic_embed

        arxiv = [f"arxiv_chunk_{i}" for i in range(6)]
        custom = [f"custom_chunk_{i}" for i in range(4)]
        chunks = arxiv + custom
        is_custom = [False] * len(arxiv) + [True] * len(custom)

        result = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=is_custom,
            boost_factor=1.5,
            reserved_custom_slots=2,
        )

        custom_in_result = [c for c in result if c.startswith("custom")]
        assert len(custom_in_result) >= 2

    def test_reserved_slots_still_include_best_arxiv(self) -> None:
        """Non-reserved slots should be filled with top arxiv."""
        service = _make_service()
        service._embed = _deterministic_embed

        arxiv = [f"arxiv_chunk_{i}" for i in range(6)]
        custom = [f"custom_chunk_{i}" for i in range(4)]
        chunks = arxiv + custom
        is_custom = [False] * len(arxiv) + [True] * len(custom)

        result = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=is_custom,
            boost_factor=1.5,
            reserved_custom_slots=2,
        )

        arxiv_in_result = [c for c in result if c.startswith("arxiv")]
        assert len(arxiv_in_result) >= 1


# ------------------------------------------------------------------ #
#  3. Mild boost without explicit intent (no reserved slots)
# ------------------------------------------------------------------ #

class TestMildBoostWithoutIntent:
    """Mild multiplier shifts scores but doesn't reserve slots."""

    def test_mild_boost_changes_order(self) -> None:
        service = _make_service()
        service._embed = _deterministic_embed

        arxiv = [f"arxiv_chunk_{i}" for i in range(5)]
        custom = [f"custom_chunk_{i}" for i in range(5)]
        chunks = arxiv + custom
        is_custom = [False] * len(arxiv) + [True] * len(custom)

        result_no_boost = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=is_custom,
            boost_factor=1.0,
            reserved_custom_slots=0,
        )

        result_mild = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=is_custom,
            boost_factor=1.2,
            reserved_custom_slots=0,
        )

        custom_no = sum(1 for c in result_no_boost if c.startswith("custom"))
        custom_mild = sum(1 for c in result_mild if c.startswith("custom"))
        assert custom_mild >= custom_no


# ------------------------------------------------------------------ #
#  4. No custom_document_ids  (regression — current behaviour)
# ------------------------------------------------------------------ #

class TestNoCustomDocuments:
    """Without custom chunks the output must match the old behaviour."""

    def test_no_custom_same_as_baseline(self) -> None:
        service = _make_service()
        service._embed = _deterministic_embed

        chunks = [f"arxiv_chunk_{i}" for i in range(8)]

        baseline = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
        )

        boosted = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=None,
            boost_factor=1.0,
            reserved_custom_slots=0,
        )

        assert baseline == boosted

    def test_all_false_is_custom_same_as_baseline(self) -> None:
        service = _make_service()
        service._embed = _deterministic_embed

        chunks = [f"arxiv_chunk_{i}" for i in range(8)]

        baseline = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
        )

        result = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=[False] * len(chunks),
            boost_factor=1.5,
            reserved_custom_slots=0,
        )

        assert baseline == result


# ------------------------------------------------------------------ #
#  5. Edge: PDF returns fewer chunks than reserved slots
# ------------------------------------------------------------------ #

class TestInsufficientCustomChunks:
    """Graceful degradation when custom chunks < reserved slots."""

    def test_zero_custom_chunks(self) -> None:
        service = _make_service()
        service._embed = _deterministic_embed

        chunks = [f"arxiv_chunk_{i}" for i in range(8)]
        is_custom = [False] * len(chunks)

        result = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=is_custom,
            boost_factor=1.5,
            reserved_custom_slots=2,
        )

        assert len(result) == 5

    def test_one_custom_chunk_two_reserved(self) -> None:
        service = _make_service()
        service._embed = _deterministic_embed

        arxiv = [f"arxiv_chunk_{i}" for i in range(7)]
        custom = ["custom_chunk_0"]
        chunks = arxiv + custom
        is_custom = [False] * len(arxiv) + [True] * len(custom)

        result = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=chunks,
            top_n=5,
            chunk_is_custom=is_custom,
            boost_factor=1.5,
            reserved_custom_slots=2,
        )

        assert len(result) == 5
        custom_in_result = [
            c for c in result if c.startswith("custom")
        ]
        assert len(custom_in_result) == 1

    def test_empty_chunks_returns_empty(self) -> None:
        service = _make_service()
        result = service._rerank_chunks(
            query_vector=QUERY_VECTOR,
            chunks=[],
            top_n=5,
        )
        assert result == []
