"""Processes custom PDF uploads: extraction, chunking,
embedding, and indexing."""

import logging
import shutil
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.documents import DocumentManager, DocumentMeta
from src.core.elastic import ElasticClient
from src.services.pdf_reader import AsyncPDFReader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles the full lifecycle of a custom PDF upload."""

    def __init__(
        self,
        elastic: ElasticClient,
        document_manager: DocumentManager,
        encoder: SentenceTransformer,
    ) -> None:
        self._elastic = elastic
        self._doc_manager = document_manager
        self._encoder = encoder

    async def process_upload(
        self,
        pdf_bytes: bytes,
        filename: str,
        tenant_id: str,
    ) -> DocumentMeta:
        raw_text = AsyncPDFReader.extract_text(pdf_bytes)
        if not raw_text.strip():
            raise ValueError(f"Could not extract text from '{filename}'")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_text(raw_text)
        if not chunks:
            raise ValueError(f"No chunks produced from '{filename}'")

        embeddings = self._encoder.encode(
            chunks, show_progress_bar=False,
        ).tolist()

        doc_meta = await self._doc_manager.save_document(
            tenant_id=tenant_id,
            filename=filename,
            total_chunks=len(chunks),
        )

        await self._elastic.index_custom_chunks(
            document_id=doc_meta.id,
            tenant_id=tenant_id,
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
        )

        upload_dir = Path(settings.upload_dir) / tenant_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / f"{doc_meta.id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        logger.info("Saved PDF to %s (%d bytes)", pdf_path, len(pdf_bytes))

        logger.info(
            "Processed document '%s' for tenant %s: %d chunks",
            filename, tenant_id, len(chunks),
        )
        return doc_meta

    async def delete_document(self, document_id: str, tenant_id: str) -> bool:
        deleted_meta = await self._doc_manager.delete_document(
            document_id, tenant_id,
        )
        if not deleted_meta:
            return False

        await self._elastic.delete_custom_document(document_id, tenant_id)

        pdf_path = Path(settings.upload_dir) / tenant_id / f"{document_id}.pdf"
        pdf_path.unlink(missing_ok=True)

        logger.info(
            "Deleted document %s for tenant %s",
            document_id, tenant_id,
        )
        return True

    async def delete_all_by_tenant(self, tenant_id: str) -> int:
        """Remove all custom docs for a tenant."""
        count = await self._doc_manager.delete_all_by_tenant(tenant_id)
        await self._elastic.delete_custom_documents_by_tenant(tenant_id)

        tenant_dir = Path(settings.upload_dir) / tenant_id
        if tenant_dir.exists():
            shutil.rmtree(tenant_dir)
            logger.info("Removed upload directory %s", tenant_dir)

        logger.info("Cleaned up %d documents for tenant %s", count, tenant_id)
        return count
