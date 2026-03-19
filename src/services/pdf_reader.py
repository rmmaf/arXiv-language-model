"""Async PDF downloader and text extractor for arXiv papers.

Downloads PDFs entirely in memory using httpx, extracts text
via LangChain's PyPDFLoader, and produces chunks via a
RecursiveCharacterTextSplitter.
"""

import asyncio
import logging
import os
import tempfile

import httpx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


class AsyncPDFReader:
    """Fetches arXiv PDFs in memory, extracts text, and splits into chunks."""

    def __init__(
        self,
        splitter: RecursiveCharacterTextSplitter | None = None,
    ) -> None:
        self._splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def fetch_pdf(self, paper_id: str) -> bytes:
        """Download a PDF from arXiv by paper ID, returning raw bytes."""
        url = f"{settings.pdf_base_url}/{paper_id}"

        async with httpx.AsyncClient(
            timeout=settings.pdf_download_timeout, follow_redirects=True
        ) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    logger.debug(
                        "Downloaded PDF for %s (%d bytes)",
                        paper_id, len(response.content),
                    )
                    return response.content

                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        wait = RETRY_BACKOFF_BASE**attempt
                        logger.warning(
                            "Rate-limited on %s, retrying in "
                            "%.1fs (attempt %d/%d)",
                            paper_id, wait, attempt, MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error(
                        "HTTP %d fetching PDF %s: %s",
                        exc.response.status_code,
                        paper_id, exc,
                    )
                    raise

                except httpx.RequestError as exc:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF_BASE**attempt
                        logger.warning(
                            "Network error fetching %s, retrying "
                            "in %.1fs (attempt %d/%d): %s",
                            paper_id, wait, attempt, MAX_RETRIES, exc,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error(
                        "Failed to fetch PDF %s after "
                        "%d attempts: %s",
                        paper_id, MAX_RETRIES, exc,
                    )
                    raise

        raise RuntimeError(f"Exhausted retries for PDF {paper_id}")

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """Extract plain text from in-memory PDF bytes.

        Writes bytes to a temporary file so that PyPDFLoader
        (which requires a filesystem path) can read them, then
        concatenates the page contents.
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False,
        )
        try:
            tmp.write(pdf_bytes)
            tmp.close()
            docs = PyPDFLoader(tmp.name).load()
            return "\n\n".join(doc.page_content for doc in docs)
        finally:
            os.unlink(tmp.name)

    def chunk_text(self, text: str) -> list[str]:
        """Split extracted text into overlapping chunks."""
        docs = self._splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    async def _process_single(self, paper_id: str) -> list[str]:
        """Download, extract, and chunk a single paper."""
        pdf_bytes = await self.fetch_pdf(paper_id)
        raw_text = self.extract_text(pdf_bytes)
        if not raw_text.strip():
            logger.warning("No text extracted from PDF %s", paper_id)
            return []
        chunks = self.chunk_text(raw_text)
        logger.info("Paper %s: extracted %d chunks", paper_id, len(chunks))
        return chunks

    async def process_papers(
        self, paper_ids: list[str],
    ) -> dict[str, list[str]]:
        """Process multiple papers concurrently."""
        tasks = [self._process_single(pid) for pid in paper_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, list[str]] = {}
        for pid, result in zip(paper_ids, results):
            if isinstance(result, BaseException):
                logger.error("Failed to process PDF %s: %s", pid, result)
                output[pid] = []
            else:
                output[pid] = result

        return output
