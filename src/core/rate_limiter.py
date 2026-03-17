"""In-memory sliding-window rate limiter keyed by tenant ID."""

import time

from fastapi import HTTPException, Request

from src.core.tenants import Tenant


class RateLimiter:
    """Track request timestamps per tenant using a 60-second sliding window."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = {}

    def check(self, tenant_id: str, limit: int) -> bool:
        """Return True if the request is allowed, False if rate limit exceeded."""
        now = time.monotonic()
        cutoff = now - 60.0

        timestamps = self._windows.get(tenant_id, [])
        timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= limit:
            self._windows[tenant_id] = timestamps
            return False

        timestamps.append(now)
        self._windows[tenant_id] = timestamps
        return True


async def check_rate_limit(request: Request, tenant: Tenant) -> None:
    """FastAPI-compatible callable that raises 429 when the tenant exceeds its limit."""
    limiter: RateLimiter = request.app.state.rate_limiter
    if not limiter.check(tenant.id, tenant.rate_limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({tenant.rate_limit} requests/minute)",
        )
