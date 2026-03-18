"""Persistent conversation store backed by SQLite (async via aiosqlite).

Supports multiple conversations per tenant with full message history,
auto-generated titles, and JSON-serialised sources/context.
"""

import json
import time
import uuid
from typing import Any

import aiosqlite

from src.core.config import settings


class ConversationStore:
    """SQLite-backed store for multi-tenant conversation sessions."""

    def __init__(self) -> None:
        self._db_path = settings.tenant_db_path

    async def init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id             TEXT PRIMARY KEY,
                    tenant_id      TEXT NOT NULL,
                    title          TEXT DEFAULT 'New conversation',
                    context        TEXT DEFAULT '',
                    sources        TEXT DEFAULT '[]',
                    created_at     REAL NOT NULL,
                    last_accessed  REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_tenant
                ON conversations(tenant_id, last_accessed DESC)
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id  TEXT NOT NULL
                                     REFERENCES conversations(id) ON DELETE CASCADE,
                    role             TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content          TEXT NOT NULL,
                    created_at       REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at)
                """
            )
            await db.commit()

    # ------------------------------------------------------------------ #
    #  Create / get / delete conversations
    # ------------------------------------------------------------------ #

    async def create(
        self,
        tenant_id: str,
        context: str = "",
        sources: list[dict[str, Any]] | None = None,
        title: str = "New conversation",
    ) -> str:
        conv_id = uuid.uuid4().hex[:12]
        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                "INSERT INTO conversations (id, tenant_id, title, context, sources, created_at, last_accessed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conv_id, tenant_id, title, context, json.dumps(sources or []), now, now),
            )
            await db.commit()
        return conv_id

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, tenant_id, title, context, sources, created_at, last_accessed "
                "FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "tenant_id": row["tenant_id"],
            "title": row["title"],
            "context": row["context"],
            "sources": json.loads(row["sources"]),
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
        }

    async def get_with_messages(self, conversation_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, tenant_id, title, context, sources, created_at, last_accessed "
                "FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            conv_row = await cursor.fetchone()
            if conv_row is None:
                return None

            cursor = await db.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            )
            msg_rows = await cursor.fetchall()

        return {
            "conversation_id": conv_row["id"],
            "tenant_id": conv_row["tenant_id"],
            "title": conv_row["title"],
            "context": conv_row["context"],
            "sources": json.loads(conv_row["sources"]),
            "created_at": conv_row["created_at"],
            "last_accessed": conv_row["last_accessed"],
            "messages": [
                {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                for r in msg_rows
            ],
        }

    async def delete(self, conversation_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            cursor = await db.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    #  List conversations for a tenant
    # ------------------------------------------------------------------ #

    async def list_by_tenant(self, tenant_id: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT c.id, c.title, c.last_accessed, c.created_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.tenant_id = ?
                GROUP BY c.id
                ORDER BY c.last_accessed DESC
                """,
                (tenant_id,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "last_accessed": r["last_accessed"],
                "created_at": r["created_at"],
                "message_count": r["message_count"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    #  Update helpers
    # ------------------------------------------------------------------ #

    async def touch(self, conversation_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET last_accessed = ? WHERE id = ?",
                (time.time(), conversation_id),
            )
            await db.commit()

    async def update_title(self, conversation_id: str, title: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )
            await db.commit()

    async def update_context(
        self, conversation_id: str, context: str, sources: list[dict[str, Any]]
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET context = ?, sources = ?, last_accessed = ? WHERE id = ?",
                (context, json.dumps(sources), time.time(), conversation_id),
            )
            await db.commit()

    # ------------------------------------------------------------------ #
    #  Message helpers
    # ------------------------------------------------------------------ #

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, time.time()),
            )
            await db.execute(
                "UPDATE conversations SET last_accessed = ? WHERE id = ?",
                (time.time(), conversation_id),
            )
            await db.commit()

    async def append_turn(self, conversation_id: str, question: str, answer: str) -> None:
        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, 'user', ?, ?)",
                (conversation_id, question, now),
            )
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, 'assistant', ?, ?)",
                (conversation_id, answer, now),
            )
            await db.execute(
                "UPDATE conversations SET last_accessed = ? WHERE id = ?",
                (now, conversation_id),
            )
            await db.commit()

    async def get_chat_history(self, conversation_id: str) -> list[dict[str, str]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            )
            rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def get_message_count(self, conversation_id: str) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
        return row[0] if row else 0
