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


class SourceDocument(BaseModel):
    """Reference to an arXiv paper returned as a source."""

    paper_id: str
    title: str
    score: float


class AskResponse(BaseModel):
    """Structured response from the RAG pipeline."""

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
