import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import AskRequest, AskResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a research question using the hybrid RAG pipeline."""
    rag_service = request.app.state.rag_service

    try:
        result = await rag_service.ask(question=body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(**result)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Check service health: Elasticsearch connectivity and LLM status."""
    es_ok = False
    try:
        await request.app.state.elastic.client.ping()
        es_ok = True
    except Exception:
        logger.warning("Elasticsearch health check failed")

    llm_ok = request.app.state.llm_manager.is_loaded

    status = "healthy" if (es_ok and llm_ok) else "degraded"
    return HealthResponse(status=status, elasticsearch=es_ok, llm_loaded=llm_ok)
