"""Document metadata registry backed by SQLite (async via aiosqlite).

Tracks custom PDF uploads per tenant with CRUD operations.
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from src.core.config import settings


@dataclass
class DocumentMeta:
    id: str
    tenant_id: str
    filename: str
    total_chunks: int
    uploaded_at: str


class DocumentManager:
    """Manages custom document metadata in the same SQLite database as tenants."""

    def __init__(self) -> None:
        self._db_path = settings.tenant_db_path

    async def init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_documents (
                    id          TEXT PRIMARY KEY,
                    tenant_id   TEXT NOT NULL,
                    filename    TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    uploaded_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_custom_docs_tenant "
                "ON custom_documents (tenant_id)"
            )
            await db.commit()

    async def save_document(
        self,
        tenant_id: str,
        filename: str,
        total_chunks: int,
    ) -> DocumentMeta:
        doc = DocumentMeta(
            id=uuid.uuid4().hex[:16],
            tenant_id=tenant_id,
            filename=filename,
            total_chunks=total_chunks,
            uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO custom_documents (id, tenant_id, filename, total_chunks, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc.id, doc.tenant_id, doc.filename, doc.total_chunks, doc.uploaded_at),
            )
            await db.commit()
        return doc

    async def list_documents(self, tenant_id: str) -> list[DocumentMeta]:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id, tenant_id, filename, total_chunks, uploaded_at "
                "FROM custom_documents WHERE tenant_id = ? ORDER BY uploaded_at DESC",
                (tenant_id,),
            )
            rows = await cursor.fetchall()
        return [
            DocumentMeta(id=r[0], tenant_id=r[1], filename=r[2], total_chunks=r[3], uploaded_at=r[4])
            for r in rows
        ]

    async def get_document(self, document_id: str, tenant_id: str) -> DocumentMeta | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id, tenant_id, filename, total_chunks, uploaded_at "
                "FROM custom_documents WHERE id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return DocumentMeta(id=row[0], tenant_id=row[1], filename=row[2], total_chunks=row[3], uploaded_at=row[4])

    async def delete_document(self, document_id: str, tenant_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM custom_documents WHERE id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_all_by_tenant(self, tenant_id: str) -> int:
        """Remove all documents belonging to a tenant. Returns count deleted."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM custom_documents WHERE tenant_id = ?",
                (tenant_id,),
            )
            await db.commit()
            return cursor.rowcount
