"""In-memory async task manager for background RAG pipeline execution.

Tracks running asyncio tasks so the frontend can poll for status,
retrieve results after completion, and cancel work in progress.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    task_id: str
    conversation_id: str
    tenant_id: str
    status: str = "processing"  # processing | completed | cancelled | error
    result: dict[str, Any] | None = None
    error_message: str | None = None
    asyncio_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class TaskManager:
    """Manages background RAG tasks with polling, cancellation and TTL cleanup."""

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self._ttl = ttl_seconds or settings.task_ttl_seconds
        self._tasks: dict[str, TaskState] = {}

    def submit(
        self,
        coro: Any,
        tenant_id: str,
        conversation_id: str,
    ) -> str:
        self._cleanup_expired()
        task_id = uuid.uuid4().hex[:12]
        state = TaskState(
            task_id=task_id,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
        )
        loop = asyncio.get_running_loop()
        asyncio_task = loop.create_task(self._run(state, coro))
        state.asyncio_task = asyncio_task
        self._tasks[task_id] = state
        logger.info("Task %s submitted for tenant %s / conv %s", task_id, tenant_id, conversation_id)
        return task_id

    async def _run(self, state: TaskState, coro: Any) -> None:
        try:
            result = await coro
            state.result = result
            state.status = "completed"
            state.finished_at = time.time()
            logger.info("Task %s completed", state.task_id)
        except asyncio.CancelledError:
            state.status = "cancelled"
            state.finished_at = time.time()
            logger.info("Task %s cancelled", state.task_id)
        except Exception as exc:
            state.status = "error"
            state.error_message = str(exc)
            state.finished_at = time.time()
            logger.exception("Task %s failed: %s", state.task_id, exc)

    def get_status(self, task_id: str) -> TaskState | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        state = self._tasks.get(task_id)
        if state is None:
            return False
        if state.status != "processing":
            return False
        if state.asyncio_task and not state.asyncio_task.done():
            state.asyncio_task.cancel()
        state.status = "cancelled"
        state.finished_at = time.time()
        logger.info("Task %s cancel requested", task_id)
        return True

    def get_active_task_for_conversation(self, conversation_id: str) -> TaskState | None:
        for state in self._tasks.values():
            if state.conversation_id == conversation_id and state.status == "processing":
                return state
        return None

    def get_active_tasks_for_tenant(self, tenant_id: str) -> dict[str, str]:
        """Return {conversation_id: task_id} for all processing tasks of a tenant."""
        result: dict[str, str] = {}
        for state in self._tasks.values():
            if state.tenant_id == tenant_id and state.status == "processing":
                result[state.conversation_id] = state.task_id
        return result

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            tid
            for tid, s in self._tasks.items()
            if s.finished_at is not None and now - s.finished_at > self._ttl
        ]
        for tid in expired:
            del self._tasks[tid]
