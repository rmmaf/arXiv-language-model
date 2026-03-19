# arXiv Research Assistant — Multi-Tenant Hybrid RAG Pipeline

A multi-tenant Retrieval-Augmented Generation (RAG) application that answers research questions by searching over **arXiv** papers and **custom uploaded PDFs** using a hybrid semantic + lexical search strategy, then generating answers with a locally hosted **Phi-3.5 Mini Instruct** LLM. Supports persistent multi-conversation sessions, asynchronous task processing, custom document uploads with intelligent boosting, per-tenant rate limiting, and an admin dashboard for tenant management and monitoring.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Data & Model Setup](#data--model-setup)
- [Running with Docker (Recommended)](#running-with-docker-recommended)
- [Running Locally (Without Docker)](#running-locally-without-docker)
- [Indexing the Dataset](#indexing-the-dataset)
- [Using the Application](#using-the-application)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)

---

## Overview

When a user asks a question, the system:

1. **Authenticates** the request via an `X-API-Key` header tied to a registered tenant.
2. **Checks rate limits** using a per-tenant sliding-window limiter (requests/minute).
3. **Creates an async task** — the question is submitted as a background task, returning a `task_id` for polling.
4. **Embeds** the question using the `all-MiniLM-L6-v2` sentence-transformer model.
5. **Searches** Elasticsearch with a hybrid query that combines BM25 (lexical) and kNN (semantic) scoring, scoped to the tenant's arXiv documents. If custom documents are attached, searches the `custom_documents` index as well.
6. **Downloads** the full PDFs of the top-matching papers from arXiv.
7. **Extracts and chunks** text from the PDFs using LangChain's PyPDFLoader and RecursiveCharacterTextSplitter, with an adaptive chunk size that decreases as more tenants are active.
8. **Re-ranks** the chunks by cosine similarity to the original question, with configurable boosting for custom document chunks (reserved slots + score multiplier).
9. **Generates** an answer by feeding the top chunks plus conversation history as context into the Phi-3.5 Mini Instruct LLM (running locally in 4-bit quantization).
10. **Stores the conversation** persistently in SQLite so follow-up questions can reuse the same context without re-fetching PDFs.

If PDF extraction fails, the system gracefully falls back to using paper abstracts as context.

## Architecture

```
┌──────────────┐        ┌────────────────────┐        ┌────────────────────────┐
│  Streamlit   │──HTTP──▶   FastAPI API       │──async─▶  Elasticsearch         │
│  Frontend    │        │  (Multi-Tenant RAG) │        │  arxiv_papers index    │
│  + Admin UI  │        └──────┬──────────────┘        │  custom_documents index │
└──────────────┘               │                       └────────────────────────┘
                  ┌────────────┼──────────────────────┐
                  ▼            ▼                      ▼
          ┌────────────┐ ┌───────────┐  ┌──────────────────┐
          │  Sentence   │ │  arXiv    │  │  Phi-3.5 Mini    │
          │ Transformer │ │  PDF DL   │  │ (4-bit, local)   │
          │  Encoder    │ │+ PyPDFLoad│  │ via HuggingFace  │
          └────────────┘ └───────────┘  └──────────────────┘
                  │
     ┌────────────┼────────────────────────────┐
     ▼            ▼                ▼           ▼
┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ SQLite   │ │ Rate Limiter │ │ Task     │ │  Document    │
│ Tenants  │ │ (in-memory)  │ │ Manager  │ │  Processor   │
│ Convos   │ └──────────────┘ │ (async)  │ │ (uploads)    │
│ Doc Meta │                  └──────────┘ └──────────────┘
└──────────┘
```

## Project Structure

```
arXiv-language-model/
├── data/                              # arXiv metadata JSONL, SQLite DBs, uploads
│   └── uploads/                       # Custom PDF uploads (per-tenant subdirectories)
├── models/                            # Phi-3.5 Mini Instruct model files (user-provided)
├── src/
│   ├── api/
│   │   ├── main.py                    # FastAPI app entry-point with lifespan management
│   │   ├── routes.py                  # Tenant endpoints (/ask, /tasks, /conversations, /health)
│   │   ├── admin_routes.py            # Admin endpoints for tenant CRUD and monitoring
│   │   ├── document_routes.py         # Custom document upload and management endpoints
│   │   └── schemas.py                 # Pydantic request/response models
│   ├── core/
│   │   ├── auth.py                    # FastAPI dependencies: tenant auth + admin auth
│   │   ├── config.py                  # Centralized settings (env vars / .env)
│   │   ├── conversation.py            # Persistent conversation store (SQLite)
│   │   ├── documents.py               # Custom document metadata manager (SQLite)
│   │   ├── elastic.py                 # Async Elasticsearch client with tenant-scoped hybrid search
│   │   ├── llm.py                     # LLM manager: extraction, 4-bit loading, pipeline
│   │   ├── rate_limiter.py            # Sliding-window rate limiter + request history log
│   │   ├── tasks.py                   # Async background task manager
│   │   └── tenants.py                 # Tenant registry backed by SQLite (via aiosqlite)
│   ├── services/
│   │   ├── document_processor.py      # Custom PDF processing, chunking, and indexing
│   │   ├── indexer.py                 # CLI script to index arXiv metadata per tenant
│   │   ├── pdf_reader.py             # Async PDF downloader + text extractor + chunker
│   │   └── rag_chain.py              # RAG orchestration: search → PDF → re-rank → LLM
│   └── ui/
│       ├── app.py                     # Streamlit chat frontend (multi-conversation)
│       └── pages/
│           └── 1_Admin.py             # Admin dashboard: monitoring, tenant CRUD, per-tenant RAG
├── tests/
│   └── test_custom_boost.py           # Tests for custom document boost logic
├── docker-compose.yml                 # Multi-service orchestration (ES + API + UI)
├── Dockerfile                         # CUDA-enabled container for the API
├── Dockerfile.ui                      # Lightweight container for the Streamlit frontend
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md
```

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM       | 16 GB   | 32 GB       |
| GPU VRAM  | 6 GB    | 8+ GB       |
| Disk      | 30 GB   | 50+ GB      |

> **Note:** The LLM runs in 4-bit quantization (`bitsandbytes`), so a mid-range NVIDIA GPU (e.g. RTX 3060 6 GB) is sufficient. CPU-only mode is supported but will be extremely slow for both embedding and inference.

### Software

- **Python** 3.11+
- **Docker** & **Docker Compose** (for containerized deployment)
- **NVIDIA GPU drivers** + **NVIDIA Container Toolkit** (for GPU support in Docker)
- **CUDA 12.1+** (if running locally without Docker)

## Data & Model Setup

### 1. arXiv Metadata Dataset

Download the arXiv metadata dataset from Kaggle:

**Source:** https://www.kaggle.com/datasets/Cornell-University/arxiv

Place the file at:

```
data/arxiv-metadata-oai-snapshot.json
```

This is a JSONL file (one JSON object per line) containing metadata for ~2.5M+ arXiv papers.

### 2. Phi-3.5 Mini Instruct Model

Download the model archive from Kaggle:

**Source:** https://www.kaggle.com/models/Microsoft/phi-3

Place the `.tar.gz` archive at:

```
models/phi-3-pytorch-phi-3.5-mini-instruct-v2.tar.gz
```

The application will automatically extract it to `models/phi-3.5-mini-instruct/` on first startup. Alternatively, if you extract it manually, ensure the model files are under that directory.

## Running with Docker (Recommended)

### 1. Configure Environment

Create a `.env` file in the project root:

```env
ADMIN_API_KEY=your-secure-admin-key
TENANT_NAME=My Research Lab
API_KEY_DEFAULT=
```

See the [Configuration](#configuration) section for all available variables.

### 2. Build and Start

```bash
docker compose up --build -d
```

This starts three services:
- **elasticsearch** — Single-node Elasticsearch 8.14 on port `9200`
- **api** — The FastAPI backend on port `8000` (with GPU passthrough)
- **streamlit** — The Streamlit web UI on port `8501`

### 3. Create a Tenant and Index the Dataset

After the services are running, create your first tenant via the admin API:

```bash
curl -X POST http://localhost:8000/api/v1/admin/tenants \
  -H "X-Admin-Key: your-secure-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Research Lab", "rate_limit": 30}'
```

Copy the `id` and `api_key` from the response. Then index the arXiv metadata for that tenant:

```bash
docker compose exec api python -m src.services.indexer --tenant-id <tenant-uuid>
```

Optionally, set the `API_KEY_DEFAULT` in `.env` (or in `docker-compose.yml`) to the tenant's API key so the Streamlit UI auto-fills it:

```env
API_KEY_DEFAULT=<tenant-api-key>
```

After all services are up and indexing is complete, open the UI at **http://localhost:8501**.

### Stopping

```bash
docker compose down
```

Add `-v` to also remove the Elasticsearch data volume:

```bash
docker compose down -v
```

## Running Without Full Docker Compose

If you prefer to start only the API and the frontend via Docker (without orchestrating everything with `docker compose up`), you still need an Elasticsearch instance running.

### 1. Start Elasticsearch

Run Elasticsearch 8.14 locally with security disabled:

```bash
docker run -d --name elasticsearch \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms2g -Xmx2g" \
  -p 9200:9200 \
  elasticsearch:8.14.0
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (or export the variables directly):

```env
ELASTICSEARCH_URL=http://elasticsearch:9200
EMBEDDING_DEVICE=cuda
ADMIN_API_KEY=your-secure-admin-key
LOG_LEVEL=INFO
```

See the [Configuration](#configuration) section for all available variables.

### 3. Build and Start the API + Frontend

```bash
docker compose up -d --build streamlit api
```

This builds and starts both the **api** (FastAPI backend with GPU passthrough) and the **streamlit** (frontend) containers. The Elasticsearch service defined in `docker-compose.yml` will also start automatically since the `api` service depends on it.

On startup the API server will:
1. Initialize the tenant database (SQLite) and in-memory rate limiter.
2. Connect to Elasticsearch and create the index (if it doesn't exist).
3. Extract the model archive (if not already extracted).
4. Load the Phi-3.5 Mini model in 4-bit quantization.
5. Initialize the RAG pipeline and conversation store.

### 4. Create a Tenant and Index the Dataset

```bash
# Create a tenant
curl -X POST http://localhost:8000/api/v1/admin/tenants \
  -H "X-Admin-Key: your-secure-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Tenant"}'

# Index arXiv metadata (use the tenant id from the response above)
docker compose exec api python -m src.services.indexer --tenant-id <tenant-uuid>
```

Open the UI at **http://localhost:8501**.

## Indexing the Dataset

The indexer (`src/services/indexer.py`) requires a `--tenant-id` argument so every indexed document is associated with a specific tenant. This enables tenant-scoped search results.

```bash
python -m src.services.indexer --tenant-id <tenant-uuid>
```

The indexer is designed for large-scale ingestion:

- **Streaming reads** — Parses the JSONL file line-by-line via a generator, keeping RAM usage low regardless of dataset size.
- **GPU-accelerated encoding** — Encodes title + abstract text into 384-dimensional embeddings using `all-MiniLM-L6-v2` on the GPU.
- **Pipelined execution** — While one batch is being sent to Elasticsearch, the next batch is being encoded on the GPU, maximizing throughput.
- **Configurable batch sizes** — `INDEXER_BATCH_SIZE` (default 2000) controls the Elasticsearch bulk size; `ENCODER_BATCH_SIZE` (default 64) controls the sentence-transformer batch size.

The Elasticsearch index uses a hybrid mapping with:
- `paper_id` and `tenant_id` as keyword fields
- `title` and `abstract` as full-text searchable fields (BM25)
- `embedding` as a `dense_vector` field with HNSW cosine similarity index (kNN)

## Using the Application

### Streamlit UI

Open your browser at **http://localhost:8501**. Enter your tenant API key in the sidebar, type a research question, and press Enter.

The chat interface supports:
- **Persistent multi-conversation sessions** — create, switch between, and delete conversations from the sidebar; state is stored in SQLite and survives page reloads
- **Background task processing** — questions are processed asynchronously with live polling and a stop button to cancel in-flight tasks
- **Custom document uploads** — upload your own PDFs to use as additional RAG context alongside arXiv papers
- **Search new papers** — a sidebar checkbox forces a new arXiv search + PDF download for the next question
- **top_k control** — adjust how many papers to retrieve (1–10)

The UI displays:
- The generated answer from the LLM
- Expandable source papers with links to their arXiv pages, relevance scores, and source type (`arxiv` or `custom_upload`)
- Processing time per response

### Admin Dashboard

Navigate to the **Admin** page via the Streamlit sidebar. Enter your `ADMIN_API_KEY` to authenticate. The dashboard provides:

- **Monitoring tab** — real-time metrics (active tenants, requests/minute, adaptive chunk size) and a request history table with status indicators
- **One tab per active tenant** — each tab is a fully functional RAG interface that sends requests using that tenant's API key, with a deactivation button
- **Tenant creation tab** ("+ New Tenant") — create a new tenant with a custom rate limit; the generated API key is displayed for copying

To pre-fill the admin key field automatically, set the `ADMIN_API_KEY` environment variable in your `.env` file or pass it to the `streamlit` service in `docker-compose.yml`.

### cURL Examples

Submit a question (returns a task ID for async polling):

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tenant-api-key>" \
  -d '{"question": "What are the latest advances in vision transformers?", "top_k": 3}'
```

Poll the task status until completion:

```bash
curl http://localhost:8000/api/v1/tasks/<task-id> \
  -H "X-API-Key: <tenant-api-key>"
```

Follow-up question reusing existing context:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tenant-api-key>" \
  -d '{"question": "How do they compare to CNNs?", "conversation_id": "<id-from-previous-response>", "fetch_new_papers": false}'
```

Upload a custom PDF document:

```bash
curl -X POST http://localhost:8000/api/v1/documents/ \
  -H "X-API-Key: <tenant-api-key>" \
  -F "file=@my-paper.pdf"
```

Ask a question using custom documents as context:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tenant-api-key>" \
  -d '{"question": "Summarize the key findings", "custom_document_ids": ["<document-id>"]}'
```

## API Reference

### Authentication

All tenant endpoints (`/api/v1/ask`, `/api/v1/tasks/*`, `/api/v1/conversations/*`, `/api/v1/documents/*`) require an `X-API-Key` header with a valid tenant API key. All `/api/v1/admin/*` requests require an `X-Admin-Key` header with the configured admin key.

---

### Tenant Endpoints

#### `POST /api/v1/ask`

Submit a research question to the RAG pipeline as a background task.

**Headers:**

| Header      | Required | Description          |
|-------------|----------|----------------------|
| `X-API-Key` | Yes      | Tenant API key       |

**Request Body:**

| Field                | Type     | Required | Default | Description                                                            |
|----------------------|----------|----------|---------|------------------------------------------------------------------------|
| `question`           | string   | Yes      | —       | Research question (10–1000 chars)                                      |
| `top_k`              | int      | No       | 3       | Number of papers to retrieve (1–10)                                    |
| `conversation_id`    | string   | No       | null    | Existing conversation ID to continue; omit to start a new conversation |
| `fetch_new_papers`   | boolean  | No       | true    | When `false`, reuses stored context instead of searching for new papers |
| `custom_document_ids`| string[] | No       | null    | IDs of custom uploaded documents to include as context                 |

**Response:**

```json
{
  "task_id": "abc123",
  "conversation_id": "a1b2c3d4e5f6"
}
```

The question is processed asynchronously. Use the `task_id` to poll for results via `GET /api/v1/tasks/{task_id}`.

**Error Responses:**

| Status | Description                             |
|--------|-----------------------------------------|
| 401    | Invalid or inactive API key             |
| 403    | Conversation belongs to another tenant  |
| 404    | Conversation not found                  |
| 429    | Tenant rate limit exceeded              |

---

#### `GET /api/v1/tasks/{task_id}`

Poll the status of a background RAG task.

**Response:**

```json
{
  "task_id": "abc123",
  "status": "completed",
  "result": {
    "answer": "Vision transformers have seen significant advances in ...",
    "sources": [
      {
        "paper_id": "2103.14030",
        "title": "Swin Transformer: Hierarchical Vision Transformer ...",
        "score": 15.432,
        "source_type": "arxiv"
      }
    ],
    "processing_time_seconds": 12.345,
    "conversation_id": "a1b2c3d4e5f6"
  },
  "error_message": null
}
```

`status` is one of: `processing`, `completed`, `cancelled`, `error`. The `result` field is only present when `status` is `completed`. The `source_type` field is `"arxiv"` for arXiv papers or `"custom_upload"` for uploaded documents.

---

#### `POST /api/v1/tasks/{task_id}/cancel`

Cancel a running background task. Returns `{"cancelled": true}` on success.

---

#### `GET /api/v1/conversations`

List all conversations for the authenticated tenant.

**Response:**

```json
[
  {
    "id": "conv-uuid",
    "title": "What are vision transformers?...",
    "last_accessed": 1700000000.0,
    "created_at": 1700000000.0,
    "message_count": 4,
    "pending_task_id": null
  }
]
```

---

#### `GET /api/v1/conversations/{conversation_id}`

Load a conversation with all messages and sources.

**Response:**

```json
{
  "conversation_id": "conv-uuid",
  "title": "What are vision transformers?...",
  "messages": [
    {"role": "user", "content": "What are vision transformers?", "created_at": 1700000000.0},
    {"role": "assistant", "content": "Vision transformers are ...", "created_at": 1700000001.0}
  ],
  "sources": [{"paper_id": "2103.14030", "title": "Swin Transformer...", "score": 15.432, "source_type": "arxiv"}],
  "pending_task_id": null
}
```

---

#### `POST /api/v1/conversations`

Create a new empty conversation. Returns `201`.

**Response:**

```json
{"id": "conv-uuid", "title": "New conversation"}
```

---

#### `DELETE /api/v1/conversations/{conversation_id}`

Delete a conversation and all its messages. Cancels any active task. Returns `204`.

---

#### `POST /api/v1/documents/`

Upload a custom PDF to use as RAG context. Accepts `multipart/form-data` with a `file` field (PDF only, max 50 MB by default).

**Response (201):**

```json
{
  "id": "doc-uuid",
  "filename": "my-paper.pdf",
  "total_chunks": 42,
  "uploaded_at": "2025-01-01T00:00:00Z"
}
```

| Status | Description                    |
|--------|--------------------------------|
| 400    | Not a PDF or empty file        |
| 413    | File exceeds size limit        |
| 422    | Could not process the PDF      |

---

#### `GET /api/v1/documents/`

List all custom documents uploaded by the current tenant.

**Response:** Array of `{id, filename, total_chunks, uploaded_at}` objects.

---

#### `DELETE /api/v1/documents/{document_id}`

Delete a custom document (ES chunks + file + metadata). Returns `204` on success, `404` if not found.

---

#### `GET /api/v1/health`

Check the health status of the service (no authentication required).

**Response:**

```json
{
  "status": "healthy",
  "elasticsearch": true,
  "llm_loaded": true
}
```

`status` is `"healthy"` when both Elasticsearch and the LLM are operational, and `"degraded"` otherwise.

---

### Admin Endpoints

#### `POST /api/v1/admin/tenants`

Create a new tenant.

**Headers:**

| Header        | Required | Description    |
|---------------|----------|----------------|
| `X-Admin-Key` | Yes      | Admin API key  |

**Request Body:**

| Field        | Type   | Required | Default | Description                          |
|--------------|--------|----------|---------|--------------------------------------|
| `name`       | string | Yes      | —       | Tenant name (1–200 chars)            |
| `rate_limit` | int    | No       | 30      | Max requests per minute (1–1000)     |

**Response (201):**

```json
{
  "id": "uuid",
  "name": "My Research Lab",
  "api_key": "generated-api-key",
  "rate_limit": 30,
  "is_active": true,
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

#### `GET /api/v1/admin/tenants`

List all tenants.

**Headers:** `X-Admin-Key` (required)

**Response:** Array of tenant objects (same schema as creation response).

---

#### `DELETE /api/v1/admin/tenants/{tenant_id}`

Deactivate a tenant and clean up all its custom documents. Returns `204` on success, `404` if not found.

**Headers:** `X-Admin-Key` (required)

---

#### `GET /api/v1/admin/metrics`

Server monitoring metrics.

**Headers:** `X-Admin-Key` (required)

**Response:**

```json
{
  "active_tenants": 3,
  "requests_last_minute": 12,
  "current_chunk_size": 800,
  "tenant_requests": {
    "tenant-uuid-1": 5,
    "tenant-uuid-2": 7
  }
}
```

---

#### `GET /api/v1/admin/request-history`

Recent request history log.

**Headers:** `X-Admin-Key` (required)

**Query Parameters:**

| Parameter | Type | Default | Description                |
|-----------|------|---------|----------------------------|
| `limit`   | int  | 50      | Number of entries (max 200) |

**Response:**

```json
[
  {
    "timestamp": "2025-01-01T12:00:00Z",
    "tenant_id": "uuid",
    "tenant_name": "My Lab",
    "question": "What are vision transformers?",
    "status": "completed",
    "processing_time": 12.345
  }
]
```

## Configuration

All settings are loaded from environment variables or a `.env` file. Below is the full list with defaults:

**Core Settings**

| Variable                | Default                                                     | Description                                       |
|-------------------------|-------------------------------------------------------------|---------------------------------------------------|
| `ELASTICSEARCH_URL`     | `http://elasticsearch:9200`                                 | Elasticsearch connection URL                      |
| `INDEX_NAME`            | `arxiv_papers`                                              | Elasticsearch index for arXiv papers              |
| `DATA_PATH`             | `data/arxiv-metadata-oai-snapshot.json`                     | Path to the arXiv JSONL metadata file             |
| `MODEL_ARCHIVE_PATH`    | `models/phi-3-pytorch-phi-3.5-mini-instruct-v2.tar.gz`     | Path to the model `.tar.gz` archive               |
| `MODEL_EXTRACTED_PATH`  | `models/phi-3.5-mini-instruct`                              | Path where the model is extracted to              |
| `EMBEDDING_MODEL_NAME`  | `all-MiniLM-L6-v2`                                         | Sentence-transformer model for embeddings         |
| `EMBEDDING_DEVICE`      | `cuda` (if available, else `cpu`)                           | Device for embedding computation                  |
| `EMBEDDING_DIM`         | `384`                                                       | Embedding vector dimensionality                   |
| `LLM_MAX_NEW_TOKENS`    | `512`                                                       | Max tokens the LLM can generate per answer        |
| `LLM_TEMPERATURE`       | `0.3`                                                       | Sampling temperature for LLM generation           |
| `LLM_TIMEOUT`           | `900.0`                                                     | Timeout (seconds) for LLM inference               |
| `API_REQUEST_TIMEOUT`   | `1200.0`                                                    | Timeout (seconds) for the full API request        |
| `CHUNK_SIZE`            | `1000`                                                      | Character size of text chunks from PDFs           |
| `CHUNK_OVERLAP`         | `200`                                                       | Overlap between consecutive chunks                |
| `TOP_K_RESULTS`         | `3`                                                         | Default number of search results                  |
| `PDF_DOWNLOAD_TIMEOUT`  | `30.0`                                                      | Timeout (seconds) for arXiv PDF downloads         |
| `PDF_BASE_URL`          | `https://arxiv.org/pdf`                                     | Base URL for arXiv PDF downloads                  |
| `INDEXER_BATCH_SIZE`    | `2000`                                                      | Documents per Elasticsearch bulk request          |
| `ENCODER_BATCH_SIZE`    | `64`                                                        | Sentences per encoding batch                      |
| `LOG_LEVEL`             | `INFO`                                                      | Logging verbosity                                 |

**Multi-Tenancy**

| Variable                | Default                                                     | Description                                       |
|-------------------------|-------------------------------------------------------------|---------------------------------------------------|
| `TENANT_DB_PATH`        | `data/tenants.db`                                           | Path to the SQLite database (tenants, conversations, document metadata) |
| `ADMIN_API_KEY`         | `admin`                                                     | Admin key for tenant management endpoints         |
| `DEFAULT_RATE_LIMIT`    | `30`                                                        | Default requests/minute for new tenants           |
| `BASE_CHUNK_SIZE`       | `1000`                                                      | Base chunk size (adapted by active tenant count)  |
| `MIN_CHUNK_SIZE`        | `400`                                                       | Minimum chunk size under high tenant load         |

**Custom Document Uploads**

| Variable                    | Default              | Description                                           |
|-----------------------------|----------------------|-------------------------------------------------------|
| `CUSTOM_DOCUMENTS_INDEX`    | `custom_documents`   | Elasticsearch index for custom document chunks        |
| `UPLOAD_DIR`                | `data/uploads`       | Directory for storing uploaded PDF files              |
| `MAX_UPLOAD_SIZE_MB`        | `50`                 | Maximum upload file size in megabytes                 |
| `CUSTOM_BOOST_FACTOR`       | `1.5`                | Score multiplier for custom document chunks when user explicitly references them |
| `CUSTOM_MILD_BOOST_FACTOR`  | `1.2`                | Score multiplier for custom chunks without explicit intent |
| `CUSTOM_RESERVED_SLOTS`     | `2`                  | Number of top-k slots reserved for custom documents   |
| `CUSTOM_CONTENT_WEIGHT`     | `0.7`                | Weight of content similarity vs. score in re-ranking  |

**Async Task Manager**

| Variable                | Default | Description                                       |
|-------------------------|---------|---------------------------------------------------|
| `TASK_TTL_SECONDS`      | `3600`  | Time-to-live for completed/failed tasks           |
| `TASK_POLL_INTERVAL`    | `2.0`   | Default polling interval (seconds) for UI clients |

**Streamlit Frontend**

| Variable              | Default                   | Description                                  |
|-----------------------|---------------------------|----------------------------------------------|
| `API_URL`             | `http://localhost:8000`   | Backend API URL                              |
| `API_REQUEST_TIMEOUT` | `600`                     | Timeout (seconds) for API requests from UI   |
| `POLL_INTERVAL`       | `2`                       | Polling interval (seconds) for task status   |
| `TENANT_NAME`         | *(empty)*                 | Display name shown in the UI header          |
| `API_KEY_DEFAULT`     | *(empty)*                 | Pre-filled API key for the chat interface    |
| `ADMIN_API_KEY`       | *(empty)*                 | Pre-filled admin key for the admin dashboard |

## Tech Stack

| Component             | Technology                                                                  |
|-----------------------|-----------------------------------------------------------------------------|
| **LLM**              | [Phi-3.5 Mini Instruct](https://www.kaggle.com/models/Microsoft/phi-3) (4-bit quantized via `bitsandbytes`) |
| **Embeddings**        | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim) |
| **Search Engine**     | [Elasticsearch 8.14](https://www.elastic.co/) — hybrid BM25 + HNSW kNN    |
| **API Framework**     | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **LLM Orchestration** | [LangChain](https://www.langchain.com/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers/) |
| **PDF Processing**    | [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/) (LangChain) + [pypdf](https://pypi.org/project/pypdf/) |
| **Frontend**          | [Streamlit](https://streamlit.io/)                                          |
| **Persistence**       | [SQLite](https://www.sqlite.org/) via [aiosqlite](https://github.com/omnilib/aiosqlite) (tenants, conversations, document metadata) |
| **Containerization**  | Docker + Docker Compose (NVIDIA CUDA 12.1 base image)                      |
| **Data Source**       | [arXiv Dataset (Kaggle)](https://www.kaggle.com/datasets/Cornell-University/arxiv) |
