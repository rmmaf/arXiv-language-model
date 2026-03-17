"""In-memory conversation store with TTL-based expiration."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """State for a single conversation session."""

    context: str
    sources: list[dict[str, Any]]
    chat_history: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class ConversationStore:
    """Thread-safe in-memory store for conversation sessions.

    Each conversation holds the assembled RAG context (re-ranked chunks),
    the source papers, and the chat history so follow-up questions can
    reuse the same context without re-fetching PDFs.
    """

    def __init__(self, ttl_seconds: float = 3600) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, ConversationState] = {}

    def create(self, context: str, sources: list[dict[str, Any]]) -> str:
        self.cleanup_expired()
        conv_id = uuid.uuid4().hex[:12]
        self._store[conv_id] = ConversationState(context=context, sources=sources)
        return conv_id

    def get(self, conversation_id: str) -> ConversationState | None:
        state = self._store.get(conversation_id)
        if state is None:
            return None
        if time.time() - state.last_accessed > self._ttl:
            self._store.pop(conversation_id, None)
            return None
        state.last_accessed = time.time()
        return state

    def append_turn(self, conversation_id: str, question: str, answer: str) -> None:
        state = self.get(conversation_id)
        if state is None:
            return
        state.chat_history.append({"role": "user", "content": question})
        state.chat_history.append({"role": "assistant", "content": answer})

    def update_context(
        self, conversation_id: str, context: str, sources: list[dict[str, Any]]
    ) -> None:
        state = self.get(conversation_id)
        if state is None:
            return
        state.context = context
        state.sources = sources

    def delete(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)

    def cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            cid
            for cid, s in self._store.items()
            if now - s.last_accessed > self._ttl
        ]
        for cid in expired:
            del self._store[cid]
