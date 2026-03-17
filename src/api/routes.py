import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.core.auth import get_current_tenant
from src.core.config import settings
from src.core.rate_limiter import check_rate_limit
from src.core.tenants import Tenant

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: Request,
    body: AskRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> AskResponse:
    """Answer a research question using the hybrid RAG pipeline."""
    await check_rate_limit(request, tenant)

    rag_service = request.app.state.rag_service
    tenant_manager = request.app.state.tenant_manager
    active_count = await tenant_manager.count_active()

    try:
        result = await asyncio.wait_for(
            rag_service.ask(
                question=body.question,
                tenant_id=tenant.id,
                top_k=body.top_k,
                active_tenant_count=active_count,
            ),
            timeout=settings.api_request_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("Request timed out after %.0fs", settings.api_request_timeout)
        raise HTTPException(status_code=504, detail="Request timed out")
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
