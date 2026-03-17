"""Admin endpoints for tenant management (protected by ADMIN_API_KEY)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import RequestLogEntry, ServerMetrics, TenantCreate, TenantResponse
from src.core.auth import require_admin
from src.core.rate_limiter import RequestHistory
from src.core.tenants import TenantManager

logger = logging.getLogger(__name__)

admin_router = APIRouter(
    prefix="/api/v1/admin",
    dependencies=[Depends(require_admin)],
)


def _get_manager(request: Request) -> TenantManager:
    return request.app.state.tenant_manager


@admin_router.post("/tenants", response_model=TenantResponse, status_code=201)
async def create_tenant(
    body: TenantCreate,
    request: Request,
) -> TenantResponse:
    manager = _get_manager(request)
    tenant = await manager.create_tenant(
        name=body.name,
        rate_limit=body.rate_limit,
    )
    logger.info("Created tenant %s (%s)", tenant.id, tenant.name)
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        api_key=tenant.api_key,
        rate_limit=tenant.rate_limit,
        is_active=tenant.is_active,
        created_at=tenant.created_at,
    )


@admin_router.get("/tenants", response_model=list[TenantResponse])
async def list_tenants(request: Request) -> list[TenantResponse]:
    manager = _get_manager(request)
    tenants = await manager.list_tenants()
    return [
        TenantResponse(
            id=t.id,
            name=t.name,
            api_key=t.api_key,
            rate_limit=t.rate_limit,
            is_active=t.is_active,
            created_at=t.created_at,
        )
        for t in tenants
    ]


@admin_router.delete("/tenants/{tenant_id}", status_code=204)
async def deactivate_tenant(tenant_id: str, request: Request) -> None:
    manager = _get_manager(request)
    changed = await manager.deactivate(tenant_id)
    if not changed:
        raise HTTPException(status_code=404, detail="Tenant not found")
    logger.info("Deactivated tenant %s", tenant_id)


@admin_router.get("/request-history", response_model=list[RequestLogEntry])
async def get_request_history(request: Request, limit: int = 50) -> list[RequestLogEntry]:
    history: RequestHistory = request.app.state.request_history
    return [
        RequestLogEntry(
            timestamp=r.timestamp,
            tenant_id=r.tenant_id,
            tenant_name=r.tenant_name,
            question=r.question,
            status=r.status,
            processing_time=r.processing_time,
        )
        for r in history.recent(limit=min(limit, 200))
    ]


@admin_router.get("/metrics", response_model=ServerMetrics)
async def get_metrics(request: Request) -> ServerMetrics:
    manager = _get_manager(request)
    rate_limiter = request.app.state.rate_limiter
    rag_service = request.app.state.rag_service
    
    active_count = await manager.count_active()
    chunk_size = rag_service._adaptive_chunk_size(active_count)
    
    metrics = rate_limiter.get_metrics()
    
    return ServerMetrics(
        active_tenants=active_count,
        requests_last_minute=metrics["requests_last_minute"],
        current_chunk_size=chunk_size,
        tenant_requests=metrics["tenant_requests"],
    )
