import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    AskRequest,
    AskResult,
    AskSubmittedResponse,
    ConversationCreateResponse,
    ConversationDetail,
    ConversationListItem,
    HealthResponse,
    MessageItem,
    SourceDocument,
    TaskStatusResponse,
)
from src.core.auth import get_current_tenant
from src.core.config import settings
from src.core.rate_limiter import RequestHistory, check_rate_limit
from src.core.tasks import TaskManager
from src.core.tenants import Tenant

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


# ------------------------------------------------------------------ #
#  Ask (async task-based)
# ------------------------------------------------------------------ #

@router.post("/ask", response_model=AskSubmittedResponse)
async def ask(
    request: Request,
    body: AskRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> AskSubmittedResponse:
    """Submit a question to the RAG pipeline as a background task."""
    await check_rate_limit(request, tenant)

    rag_service = request.app.state.rag_service
    tenant_manager = request.app.state.tenant_manager
    history: RequestHistory = request.app.state.request_history
    conversation_store = request.app.state.conversation_store
    task_manager: TaskManager = request.app.state.task_manager
    active_count = await tenant_manager.count_active()

    conv_id = body.conversation_id
    title = body.question[:60] + ("..." if len(body.question) > 60 else "")
    if not conv_id:
        conv_id = await conversation_store.create(
            tenant_id=tenant.id, title=title,
        )

    conv = await conversation_store.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv["tenant_id"] != tenant.id:
        raise HTTPException(
            status_code=403,
            detail="Conversation belongs to another tenant",
        )

    if conv["title"] == "New conversation":
        await conversation_store.update_title(conv_id, title)

    await conversation_store.add_message(
        conv_id, "user", body.question,
    )

    cancel_event = threading.Event()

    coro = rag_service.ask(
        question=body.question,
        tenant_id=tenant.id,
        conversation_store=conversation_store,
        top_k=body.top_k,
        active_tenant_count=active_count,
        conversation_id=conv_id,
        fetch_new_papers=body.fetch_new_papers,
        custom_document_ids=body.custom_document_ids,
        cancel_event=cancel_event,
    )

    task_id = task_manager.submit(
        coro,
        tenant_id=tenant.id,
        conversation_id=conv_id,
        cancel_event=cancel_event,
    )

    history.log(
        tenant.id, tenant.name, body.question,
        "submitted", task_id=task_id,
    )

    return AskSubmittedResponse(task_id=task_id, conversation_id=conv_id)


# ------------------------------------------------------------------ #
#  Task polling / cancellation
# ------------------------------------------------------------------ #

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    request: Request,
    task_id: str,
    tenant: Tenant = Depends(get_current_tenant),
) -> TaskStatusResponse:
    """Poll the status of a background RAG task."""
    task_manager: TaskManager = request.app.state.task_manager
    state = task_manager.get_status(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if state.tenant_id != tenant.id:
        raise HTTPException(
            status_code=403,
            detail="Task belongs to another tenant",
        )

    result = None
    processing_time: float | None = None
    if state.status == "completed" and state.result:
        processing_time = state.result.get(
            "processing_time_seconds", 0,
        )
        result = AskResult(
            answer=state.result["answer"],
            sources=[
                SourceDocument(**s)
                for s in state.result.get("sources", [])
            ],
            processing_time_seconds=processing_time,
            conversation_id=state.result.get(
                "conversation_id",
                state.conversation_id,
            ),
        )

    if state.status in ("completed", "error"):
        history: RequestHistory = (
            request.app.state.request_history
        )
        history.update_status(
            task_id, state.status, processing_time,
        )

    return TaskStatusResponse(
        task_id=state.task_id,
        status=state.status,
        result=result,
        error_message=state.error_message,
    )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    request: Request,
    task_id: str,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict:
    """Cancel a running background RAG task."""
    task_manager: TaskManager = request.app.state.task_manager
    state = task_manager.get_status(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if state.tenant_id != tenant.id:
        raise HTTPException(
            status_code=403,
            detail="Task belongs to another tenant",
        )

    cancelled = task_manager.cancel(task_id)

    if cancelled:
        history: RequestHistory = request.app.state.request_history
        history.update_status(task_id, "cancelled")

    return {"cancelled": cancelled}


# ------------------------------------------------------------------ #
#  Conversations CRUD
# ------------------------------------------------------------------ #

@router.get("/conversations", response_model=list[ConversationListItem])
async def list_conversations(
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
) -> list[ConversationListItem]:
    """List all conversations for the authenticated tenant."""
    conversation_store = request.app.state.conversation_store
    task_manager: TaskManager = request.app.state.task_manager

    convs = await conversation_store.list_by_tenant(tenant.id)
    active_tasks = task_manager.get_active_tasks_for_tenant(tenant.id)

    return [
        ConversationListItem(
            id=c["id"],
            title=c["title"],
            last_accessed=c["last_accessed"],
            created_at=c["created_at"],
            message_count=c["message_count"],
            pending_task_id=active_tasks.get(c["id"]),
        )
        for c in convs
    ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
)
async def get_conversation(
    request: Request,
    conversation_id: str,
    tenant: Tenant = Depends(get_current_tenant),
) -> ConversationDetail:
    """Load a conversation with all messages."""
    conversation_store = request.app.state.conversation_store
    task_manager: TaskManager = request.app.state.task_manager

    conv = await conversation_store.get_with_messages(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv["tenant_id"] != tenant.id:
        raise HTTPException(
            status_code=403,
            detail="Conversation belongs to another tenant",
        )

    active_task = task_manager.get_active_task_for_conversation(
        conversation_id,
    )

    return ConversationDetail(
        conversation_id=conv["conversation_id"],
        title=conv["title"],
        messages=[
            MessageItem(
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"],
            )
            for m in conv["messages"]
        ],
        sources=[
            SourceDocument(**s) for s in conv["sources"]
        ],
        pending_task_id=(
            active_task.task_id if active_task else None
        ),
    )


@router.post(
    "/conversations",
    response_model=ConversationCreateResponse,
    status_code=201,
)
async def create_conversation(
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
) -> ConversationCreateResponse:
    """Create a new empty conversation."""
    conversation_store = request.app.state.conversation_store
    title = "New conversation"
    conv_id = await conversation_store.create(
        tenant_id=tenant.id, title=title,
    )
    return ConversationCreateResponse(
        id=conv_id, title=title,
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    request: Request,
    conversation_id: str,
    tenant: Tenant = Depends(get_current_tenant),
) -> None:
    """Delete a conversation and all its messages."""
    conversation_store = request.app.state.conversation_store

    conv = await conversation_store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv["tenant_id"] != tenant.id:
        raise HTTPException(
            status_code=403,
            detail="Conversation belongs to another tenant",
        )

    task_manager: TaskManager = request.app.state.task_manager
    active_task = task_manager.get_active_task_for_conversation(
        conversation_id,
    )
    if active_task:
        task_manager.cancel(active_task.task_id)
        history: RequestHistory = (
            request.app.state.request_history
        )
        history.update_status(
            active_task.task_id, "cancelled",
        )

    await conversation_store.delete(conversation_id)


# ------------------------------------------------------------------ #
#  Health
# ------------------------------------------------------------------ #

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
    return HealthResponse(
        status=status, elasticsearch=es_ok, llm_loaded=llm_ok,
    )
