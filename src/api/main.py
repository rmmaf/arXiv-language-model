"""FastAPI application entry-point with lifespan management."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.admin_routes import admin_router
from src.api.routes import router
from src.core.config import settings
from src.core.elastic import ElasticClient
from src.core.llm import LLMManager
from src.core.rate_limiter import RateLimiter, RequestHistory
from src.core.tenants import TenantManager
from src.services.rag_chain import RAGService

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: connect to ES, load LLM, init tenants, build RAG service.  Shutdown: release."""
    logger.info("Starting up ...")

    tenant_manager = TenantManager()
    await tenant_manager.init_db()
    app.state.tenant_manager = tenant_manager
    logger.info("Tenant database initialised")

    app.state.rate_limiter = RateLimiter()
    app.state.request_history = RequestHistory()

    elastic = ElasticClient()
    await elastic.connect()
    await elastic.create_index()
    app.state.elastic = elastic

    llm_manager = LLMManager()
    llm_manager.load()
    app.state.llm_manager = llm_manager

    app.state.rag_service = RAGService(elastic=elastic, llm_manager=llm_manager)

    logger.info("Application ready")
    yield

    logger.info("Shutting down ...")
    await elastic.close()


app = FastAPI(
    title="ArXiv Hybrid RAG API",
    version="1.0.0",
    description="Multi-tenant hybrid semantic + lexical search over arXiv papers with local LLM.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(admin_router)
