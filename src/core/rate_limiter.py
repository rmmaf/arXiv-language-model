"""In-memory sliding-window rate limiter keyed by tenant ID."""

import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException, Request

from src.core.tenants import Tenant

_MAX_HISTORY = 200


@dataclass
class RequestRecord:
    """Single request log entry."""

    timestamp: str
    tenant_id: str
    tenant_name: str
    question: str
    status: str
    processing_time: float | None = None


class RequestHistory:
    """Ring-buffer of recent request records for the monitoring dashboard."""

    def __init__(self, maxlen: int = _MAX_HISTORY) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=maxlen)

    def log(
        self,
        tenant_id: str,
        tenant_name: str,
        question: str,
        status: str,
        processing_time: float | None = None,
    ) -> None:
        self._records.appendleft(
            RequestRecord(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                question=(
                    question if len(question) <= 120
                    else question[:117] + "..."
                ),
                status=status,
                processing_time=processing_time,
            )
        )

    def recent(self, limit: int = 50) -> list[RequestRecord]:
        return list(self._records)[:limit]


class RateLimiter:
    """Track request timestamps per tenant using a 60-second sliding window."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = {}

    def check(self, tenant_id: str, limit: int) -> bool:
        """Return True if allowed, False if rate limit exceeded."""
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

    def get_metrics(self) -> dict:
        """Return metrics about requests in the last minute."""
        now = time.monotonic()
        cutoff = now - 60.0

        tenant_requests = {}
        total_requests = 0

        for tenant_id, timestamps in list(self._windows.items()):
            valid_timestamps = [t for t in timestamps if t > cutoff]
            if valid_timestamps:
                self._windows[tenant_id] = valid_timestamps
                count = len(valid_timestamps)
                tenant_requests[tenant_id] = count
                total_requests += count
            else:
                self._windows.pop(tenant_id, None)

        return {
            "requests_last_minute": total_requests,
            "tenant_requests": tenant_requests
        }


async def check_rate_limit(request: Request, tenant: Tenant) -> None:
    """Raise 429 when the tenant exceeds its rate limit."""
    limiter: RateLimiter = request.app.state.rate_limiter
    if not limiter.check(tenant.id, tenant.rate_limit):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded "
                f"({tenant.rate_limit} requests/minute)"
            ),
        )
