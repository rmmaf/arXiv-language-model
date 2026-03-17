"""Endpoints for custom document upload and management (tenant-scoped)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from src.api.schemas import DocumentListItem, DocumentUploadResponse
from src.core.auth import get_current_tenant
from src.core.config import settings
from src.core.tenants import Tenant

logger = logging.getLogger(__name__)

document_router = APIRouter(prefix="/api/v1/documents")


@document_router.post("/", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile,
    tenant: Tenant = Depends(get_current_tenant),
) -> DocumentUploadResponse:
    """Upload a custom PDF to use as RAG context."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    pdf_bytes = await file.read()
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb}MB limit.",
        )
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    processor = request.app.state.document_processor

    try:
        doc_meta = await processor.process_upload(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            tenant_id=tenant.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "Tenant %s uploaded '%s' (%d chunks)",
        tenant.id, doc_meta.filename, doc_meta.total_chunks,
    )
    return DocumentUploadResponse(
        id=doc_meta.id,
        filename=doc_meta.filename,
        total_chunks=doc_meta.total_chunks,
        uploaded_at=doc_meta.uploaded_at,
    )


@document_router.get("/", response_model=list[DocumentListItem])
async def list_documents(
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
) -> list[DocumentListItem]:
    """List all custom documents uploaded by the current tenant."""
    doc_manager = request.app.state.document_manager
    docs = await doc_manager.list_documents(tenant.id)
    return [
        DocumentListItem(
            id=d.id,
            filename=d.filename,
            total_chunks=d.total_chunks,
            uploaded_at=d.uploaded_at,
        )
        for d in docs
    ]


@document_router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
) -> None:
    """Delete a custom document (ES chunks + file + metadata)."""
    processor = request.app.state.document_processor
    deleted = await processor.delete_document(document_id, tenant.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    logger.info("Tenant %s deleted document %s", tenant.id, document_id)
