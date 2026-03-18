"""FastAPI dependencies for tenant authentication and admin access."""

from fastapi import Header, HTTPException, Request

from src.core.config import settings
from src.core.tenants import Tenant, TenantManager


async def get_current_tenant(
    request: Request,
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> Tenant:
    """Validate the API key and return the associated tenant."""
    tenant_manager: TenantManager = request.app.state.tenant_manager
    tenant = await tenant_manager.get_by_api_key(x_api_key)
    if not tenant or not tenant.is_active:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
        )
    return tenant


async def require_admin(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> None:
    """Verify the request carries a valid admin key."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
