"""Tenant registry backed by SQLite (async via aiosqlite).

Provides CRUD operations and an in-memory cache with TTL to avoid
repeated database round-trips on every request.
"""

import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from src.core.config import settings

_CACHE_TTL_SECONDS = 60.0


@dataclass
class Tenant:
    id: str
    name: str
    api_key: str
    rate_limit: int
    is_active: bool
    created_at: str


class TenantManager:
    """Manages tenant lifecycle and caches lookups by API key."""

    def __init__(self) -> None:
        self._db_path = settings.tenant_db_path
        self._cache: dict[str, tuple[Tenant, float]] = {}

    async def init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    id         TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    api_key    TEXT NOT NULL UNIQUE,
                    rate_limit INTEGER NOT NULL,
                    is_active  INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.commit()

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    def _from_row(self, row: aiosqlite.Row) -> Tenant:
        return Tenant(
            id=row[0],
            name=row[1],
            api_key=row[2],
            rate_limit=row[3],
            is_active=bool(row[4]),
            created_at=row[5],
        )

    async def create_tenant(
        self, name: str, rate_limit: int | None = None
    ) -> Tenant:
        tenant = Tenant(
            id=str(uuid.uuid4()),
            name=name,
            api_key=secrets.token_urlsafe(32),
            rate_limit=rate_limit or settings.default_rate_limit,
            is_active=True,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO tenants (id, name, api_key, rate_limit, is_active, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tenant.id,
                    tenant.name,
                    tenant.api_key,
                    tenant.rate_limit,
                    int(tenant.is_active),
                    tenant.created_at,
                ),
            )
            await db.commit()
        self._invalidate_cache()
        return tenant

    async def get_by_api_key(self, api_key: str) -> Tenant | None:
        cached = self._cache.get(api_key)
        if cached is not None:
            tenant, ts = cached
            if time.monotonic() - ts < _CACHE_TTL_SECONDS:
                return tenant

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id, name, api_key, rate_limit, is_active, created_at "
                "FROM tenants WHERE api_key = ?",
                (api_key,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        tenant = self._from_row(row)
        self._cache[api_key] = (tenant, time.monotonic())
        return tenant

    async def list_tenants(self) -> list[Tenant]:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT id, name, api_key, rate_limit, is_active, created_at "
                "FROM tenants ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
        return [self._from_row(r) for r in rows]

    async def count_active(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tenants WHERE is_active = 1"
            )
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def deactivate(self, tenant_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "UPDATE tenants SET is_active = 0 WHERE id = ?",
                (tenant_id,),
            )
            await db.commit()
            changed = cursor.rowcount > 0
        if changed:
            self._invalidate_cache()
        return changed
