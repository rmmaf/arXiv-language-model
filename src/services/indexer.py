"""Standalone script to index arXiv metadata into Elasticsearch.

Reads the JSONL dataset line-by-line via a generator to keep RAM usage low,
encodes embeddings on GPU (with CPU fallback), and bulk-inserts into
Elasticsearch in batches.  Encoding and indexing are pipelined so the GPU
stays busy while the previous batch is sent to Elasticsearch.

Usage:
    python -m src.services.indexer
"""

import asyncio
import json
import logging
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.elastic import ElasticClient

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def iter_metadata(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield one parsed JSON object per line from the JSONL dataset."""
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)


def build_text(doc: dict[str, Any]) -> str:
    """Combine title and abstract into a single string for embedding."""
    title = doc.get("title", "").strip()
    abstract = doc.get("abstract", "").strip()
    return f"{title} {abstract}"


def _encode_batch(
    encoder: SentenceTransformer, texts: list[str]
) -> np.ndarray:
    """Blocking call — meant to run inside a thread-pool executor."""
    return encoder.encode(
        texts,
        batch_size=settings.encoder_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def _collect_batch(
    doc_iter: Generator[dict[str, Any], None, None],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Accumulate up to *indexer_batch_size* records from the iterator."""
    batch: list[dict[str, Any]] = []
    texts: list[str] = []
    for doc in doc_iter:
        paper_id = doc.get("id")
        if not paper_id:
            continue
        text = build_text(doc)
        if not text.strip():
            continue

        batch.append(
            {
                "paper_id": paper_id,
                "title": doc.get("title", "").strip(),
                "abstract": doc.get("abstract", "").strip(),
                "categories": doc.get("categories", ""),
                "authors": doc.get("authors", ""),
            }
        )
        texts.append(text)
        if len(batch) >= settings.indexer_batch_size:
            break
    return batch, texts


async def run_indexing() -> None:
    if settings.embedding_device == "cpu":
        logger.warning(
            "⚠ Embedding running on CPU — indexing will be extremely slow. "
            "Set EMBEDDING_DEVICE=cuda or use a machine with GPU for 10-50x speedup."
        )

    encoder = SentenceTransformer(
        settings.embedding_model_name, device=settings.embedding_device
    )
    logger.info(
        "Loaded embedding model '%s' on %s  |  indexer_batch=%d  encoder_batch=%d",
        settings.embedding_model_name,
        settings.embedding_device,
        settings.indexer_batch_size,
        settings.encoder_batch_size,
    )

    elastic = ElasticClient()
    await elastic.connect()
    await elastic.create_index()

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)

    total_indexed = 0
    start_time = time.perf_counter()
    doc_iter = iter_metadata(settings.data_file)

    # -- Kickstart: collect + encode the first batch synchronously -----------
    batch, texts = _collect_batch(doc_iter)
    if not batch:
        logger.info("No documents to index.")
        await elastic.close()
        return

    embeddings = await loop.run_in_executor(
        pool, _encode_batch, encoder, texts
    )

    while batch:
        # Attach embeddings to current batch records
        for rec, emb in zip(batch, embeddings):
            rec["embedding"] = emb.tolist()

        # Collect NEXT batch of raw docs (CPU-only, fast)
        next_batch, next_texts = _collect_batch(doc_iter)

        # Launch encoding of next batch on GPU (in background thread) …
        encode_future: asyncio.Future | None = None
        if next_batch:
            encode_future = loop.run_in_executor(
                pool, _encode_batch, encoder, next_texts
            )

        # … while simultaneously bulk-indexing the current batch via network.
        try:
            indexed = await elastic.bulk_index(batch)
            total_indexed += indexed
        except Exception:
            logger.exception("Failed to index batch")

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Indexed %d documents total (%.1f docs/sec)",
            total_indexed,
            total_indexed / elapsed if elapsed > 0 else 0,
        )

        # Wait for next encoding to finish (it likely already has)
        if encode_future is not None:
            embeddings = await encode_future
        batch, texts = next_batch, next_texts

    pool.shutdown(wait=False)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Indexing complete: %d documents in %.1f seconds (%.1f docs/sec)",
        total_indexed,
        elapsed,
        total_indexed / elapsed if elapsed > 0 else 0,
    )

    await elastic.close()


if __name__ == "__main__":
    asyncio.run(run_indexing())
