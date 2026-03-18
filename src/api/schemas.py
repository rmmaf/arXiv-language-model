from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for the /ask endpoint."""

    question: str = Field(..., min_length=10, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID to continue. Omit to start a new conversation.",
    )
    fetch_new_papers: bool = Field(
        default=True,
        description="When False, reuses the context from the existing conversation instead of searching for new papers.",
    )
    custom_document_ids: list[str] | None = Field(
        default=None,
        description="IDs of custom uploaded documents to include as context.",
    )


class SourceDocument(BaseModel):
    """Reference to a source document (arXiv paper or custom upload)."""

    paper_id: str
    title: str
    score: float
    source_type: str = "arxiv"


class AskSubmittedResponse(BaseModel):
    """Immediate response when a question is submitted (async task created)."""

    task_id: str
    conversation_id: str


class AskResult(BaseModel):
    """Final result from the RAG pipeline (returned inside TaskStatusResponse)."""

    answer: str
    sources: list[SourceDocument]
    processing_time_seconds: float
    conversation_id: str


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    elasticsearch: bool
    llm_loaded: bool


# --------------- Tenant admin schemas ---------------

class TenantCreate(BaseModel):
    """Payload for creating a new tenant."""

    name: str = Field(..., min_length=1, max_length=200)
    rate_limit: int | None = Field(default=None, ge=1, le=1000)


class TenantResponse(BaseModel):
    """Tenant data returned by admin endpoints."""

    id: str
    name: str
    api_key: str
    rate_limit: int
    is_active: bool
    created_at: str


class TenantListItem(BaseModel):
    """Tenant summary (API key masked) for list endpoints."""

    id: str
    name: str
    rate_limit: int
    is_active: bool
    created_at: str


class ServerMetrics(BaseModel):
    """Server monitoring metrics."""

    active_tenants: int
    requests_last_minute: int
    current_chunk_size: int
    tenant_requests: dict[str, int]


class RequestLogEntry(BaseModel):
    """Single entry in the request history log."""

    timestamp: str
    tenant_id: str
    tenant_name: str
    question: str
    status: str
    processing_time: float | None = None


# --------------- Task schemas ---------------

class TaskStatusResponse(BaseModel):
    """Status of an async RAG task."""

    task_id: str
    status: str  # processing | completed | cancelled | error
    result: AskResult | None = None
    error_message: str | None = None


# --------------- Conversation schemas ---------------

class MessageItem(BaseModel):
    """Single message in a conversation."""

    role: str
    content: str
    created_at: float


class ConversationListItem(BaseModel):
    """Conversation summary for sidebar listing."""

    id: str
    title: str
    last_accessed: float
    created_at: float
    message_count: int
    pending_task_id: str | None = None


class ConversationDetail(BaseModel):
    """Full conversation with messages, used when loading a conversation."""

    conversation_id: str
    title: str
    messages: list[MessageItem]
    sources: list[SourceDocument]
    pending_task_id: str | None = None


class ConversationCreateResponse(BaseModel):
    """Response after creating a new empty conversation."""

    id: str
    title: str


# --------------- Custom document schemas ---------------

class DocumentUploadResponse(BaseModel):
    """Response after a successful custom PDF upload."""

    id: str
    filename: str
    total_chunks: int
    uploaded_at: str


class DocumentListItem(BaseModel):
    """Summary of a custom uploaded document."""

    id: str
    filename: str
    total_chunks: int
    uploaded_at: str
