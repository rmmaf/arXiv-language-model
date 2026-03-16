from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for the /ask endpoint."""

    question: str = Field(..., min_length=10, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


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


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    elasticsearch: bool
    llm_loaded: bool
