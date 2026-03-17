# arXiv Research Assistant — Hybrid RAG Pipeline

A Retrieval-Augmented Generation (RAG) application that answers research questions by searching over **arXiv** papers using a hybrid semantic + lexical search strategy, then generating answers with a locally hosted **Phi-3.5 Mini Instruct** LLM.

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

1. **Embeds** the question using the `all-MiniLM-L6-v2` sentence-transformer model.
2. **Searches** Elasticsearch with a hybrid query that combines BM25 (lexical) and kNN (semantic) scoring.
3. **Downloads** the full PDFs of the top-matching papers from arXiv.
4. **Extracts and chunks** text from the PDFs using PyMuPDF and LangChain text splitters.
5. **Re-ranks** the chunks by cosine similarity to the original question.
6. **Generates** an answer by feeding the top chunks as context into the Phi-3.5 Mini Instruct LLM (running locally in 4-bit quantization).

If PDF extraction fails, the system gracefully falls back to using paper abstracts as context.

## Architecture

```
┌──────────────┐        ┌────────────────┐        ┌────────────────────┐
│  Streamlit   │──HTTP──▶   FastAPI API   │──async─▶  Elasticsearch    │
│  Frontend    │        │  (RAG Engine)   │        │  (Hybrid Index)    │
└──────────────┘        └───────┬────────┘        └────────────────────┘
                                │
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────────────┐
             │ Sentence  │ │  arXiv   │ │  Phi-3.5 Mini    │
             │Transformer│ │ PDF DL   │ │ (4-bit, local)   │
             │ Encoder   │ │ + PyMuPDF│ │ via HuggingFace  │
             └──────────┘ └──────────┘ └──────────────────┘
```

## Project Structure

```
arXiv-language-model/
├── data/                          # arXiv metadata JSONL dataset (user-provided)
├── models/                        # Phi-3.5 Mini Instruct model files (user-provided)
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI app entry-point with lifespan management
│   │   ├── routes.py              # API route definitions (/ask, /health)
│   │   └── schemas.py             # Pydantic request/response models
│   ├── core/
│   │   ├── config.py              # Centralized settings (env vars / .env)
│   │   ├── elastic.py             # Async Elasticsearch client with hybrid search
│   │   └── llm.py                 # LLM manager: extraction, 4-bit loading, pipeline
│   ├── services/
│   │   ├── indexer.py             # Standalone script to index arXiv metadata into ES
│   │   ├── pdf_reader.py          # Async PDF downloader + text extractor + chunker
│   │   └── rag_chain.py           # RAG orchestration: search → PDF → re-rank → LLM
│   └── ui/
│       └── app.py                 # Streamlit web frontend
├── docker-compose.yml             # Multi-service orchestration (ES + API + UI)
├── Dockerfile                     # CUDA-enabled container for the API
├── Dockerfile.ui                  # Lightweight container for the Streamlit frontend
├── requirements.txt               # Python dependencies
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

### 1. Build and Start

```bash
docker compose up --build -d
```

This starts three services:
- **elasticsearch** — Single-node Elasticsearch 8.14 on port `9200`
- **api** — The FastAPI backend on port `8000` (with GPU passthrough)
- **streamlit** — The Streamlit web UI on port `8501`

After all services are up, open the UI at **http://localhost:8501**.

### 2. Index the Dataset

Once the services are running and Elasticsearch is healthy, index the arXiv metadata:

```bash
docker compose exec api python -m src.services.indexer
```

This reads the JSONL dataset line-by-line (low memory footprint), encodes embeddings in batches on the GPU, and bulk-indexes documents into Elasticsearch. Progress is logged to stdout.

### Stopping

```bash
docker compose down
```

Add `-v` to also remove the Elasticsearch data volume:

```bash
docker compose down -v
```

## Running Locally (Without Docker)

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

### 2. Create a Virtual Environment and Install Dependencies

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** For GPU support, ensure you have a CUDA-compatible version of PyTorch installed. See https://pytorch.org/get-started/locally/ for install instructions specific to your CUDA version.

### 3. Configure Environment Variables

Create a `.env` file in the project root (or export the variables directly):

```env
ELASTICSEARCH_URL=http://localhost:9200
EMBEDDING_DEVICE=cuda
LOG_LEVEL=INFO
```

See the [Configuration](#configuration) section for all available variables.

### 4. Index the Dataset

```bash
python -m src.services.indexer
```

### 5. Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

On startup the server will:
1. Connect to Elasticsearch and create the index (if it doesn't exist).
2. Extract the model archive (if not already extracted).
3. Load the Phi-3.5 Mini model in 4-bit quantization.
4. Initialize the RAG pipeline.

### 6. Start the Streamlit Frontend

```bash
streamlit run src/ui/app.py
```

## Indexing the Dataset

The indexer (`src/services/indexer.py`) is designed for large-scale ingestion:

- **Streaming reads** — Parses the JSONL file line-by-line via a generator, keeping RAM usage low regardless of dataset size.
- **GPU-accelerated encoding** — Encodes title + abstract text into 384-dimensional embeddings using `all-MiniLM-L6-v2` on the GPU.
- **Pipelined execution** — While one batch is being sent to Elasticsearch, the next batch is being encoded on the GPU, maximizing throughput.
- **Configurable batch sizes** — `INDEXER_BATCH_SIZE` (default 2000) controls the Elasticsearch bulk size; `ENCODER_BATCH_SIZE` (default 64) controls the sentence-transformer batch size.

The Elasticsearch index uses a hybrid mapping with:
- `title` and `abstract` as full-text searchable fields (BM25)
- `embedding` as a `dense_vector` field with HNSW cosine similarity index (kNN)

## Using the Application

### Streamlit UI

The Streamlit UI starts automatically with `docker compose up`. Open your browser at **http://localhost:8501**. Type a research question, select the number of results (`top_k`), and click **Ask**.

The UI displays:
- The generated answer from the LLM
- A list of source papers with links to their arXiv pages
- Total processing time

### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest advances in vision transformers?", "top_k": 3}'
```

## API Reference

### `POST /api/v1/ask`

Ask a research question using the RAG pipeline.

**Request Body:**

| Field      | Type   | Required | Default | Description                          |
|------------|--------|----------|---------|--------------------------------------|
| `question` | string | Yes      | —       | Research question (10–1000 chars)    |
| `top_k`    | int    | No       | 3       | Number of papers to retrieve (1–10)  |

**Response:**

```json
{
  "answer": "Vision transformers have seen significant advances in ...",
  "sources": [
    {
      "paper_id": "2103.14030",
      "title": "Swin Transformer: Hierarchical Vision Transformer ...",
      "score": 15.432
    }
  ],
  "processing_time_seconds": 12.345
}
```

### `GET /api/v1/health`

Check the health status of the service.

**Response:**

```json
{
  "status": "healthy",
  "elasticsearch": true,
  "llm_loaded": true
}
```

`status` is `"healthy"` when both Elasticsearch and the LLM are operational, and `"degraded"` otherwise.

## Configuration

All settings are loaded from environment variables or a `.env` file. Below is the full list with defaults:

| Variable                | Default                                                     | Description                                |
|-------------------------|-------------------------------------------------------------|--------------------------------------------|
| `ELASTICSEARCH_URL`     | `http://elasticsearch:9200`                                 | Elasticsearch connection URL               |
| `INDEX_NAME`            | `arxiv_papers`                                              | Elasticsearch index name                   |
| `DATA_PATH`             | `data/arxiv-metadata-oai-snapshot.json`                     | Path to the arXiv JSONL metadata file      |
| `MODEL_ARCHIVE_PATH`    | `models/phi-3-pytorch-phi-3.5-mini-instruct-v2.tar.gz`     | Path to the model `.tar.gz` archive        |
| `MODEL_EXTRACTED_PATH`  | `models/phi-3.5-mini-instruct`                              | Path where the model is extracted to       |
| `EMBEDDING_MODEL_NAME`  | `all-MiniLM-L6-v2`                                         | Sentence-transformer model for embeddings  |
| `EMBEDDING_DEVICE`      | `cuda` (if available, else `cpu`)                           | Device for embedding computation           |
| `EMBEDDING_DIM`         | `384`                                                       | Embedding vector dimensionality            |
| `LLM_MAX_NEW_TOKENS`    | `512`                                                       | Max tokens the LLM can generate per answer |
| `LLM_TEMPERATURE`       | `0.3`                                                       | Sampling temperature for LLM generation    |
| `CHUNK_SIZE`            | `1000`                                                      | Character size of text chunks from PDFs    |
| `CHUNK_OVERLAP`         | `200`                                                       | Overlap between consecutive chunks         |
| `TOP_K_RESULTS`         | `3`                                                         | Default number of search results           |
| `PDF_DOWNLOAD_TIMEOUT`  | `30.0`                                                      | Timeout (seconds) for arXiv PDF downloads  |
| `PDF_BASE_URL`          | `https://arxiv.org/pdf`                                     | Base URL for arXiv PDF downloads           |
| `INDEXER_BATCH_SIZE`    | `2000`                                                      | Documents per Elasticsearch bulk request   |
| `ENCODER_BATCH_SIZE`    | `64`                                                        | Sentences per encoding batch               |
| `LOG_LEVEL`             | `INFO`                                                      | Logging verbosity                          |

## Tech Stack

| Component             | Technology                                                                  |
|-----------------------|-----------------------------------------------------------------------------|
| **LLM**              | [Phi-3.5 Mini Instruct](https://www.kaggle.com/models/Microsoft/phi-3) (4-bit quantized via `bitsandbytes`) |
| **Embeddings**        | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim) |
| **Search Engine**     | [Elasticsearch 8.14](https://www.elastic.co/) — hybrid BM25 + HNSW kNN    |
| **API Framework**     | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **LLM Orchestration** | [LangChain](https://www.langchain.com/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers/) |
| **PDF Processing**    | [PyMuPDF](https://pymupdf.readthedocs.io/) (in-memory extraction)          |
| **Frontend**          | [Streamlit](https://streamlit.io/)                                          |
| **Containerization**  | Docker + Docker Compose (NVIDIA CUDA 12.1 base image)                      |
| **Data Source**        | [arXiv Dataset (Kaggle)](https://www.kaggle.com/datasets/Cornell-University/arxiv) |
