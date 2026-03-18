from pathlib import Path

import torch
from pydantic_settings import BaseSettings


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment / .env file."""

    elasticsearch_url: str = "http://elasticsearch:9200"
    index_name: str = "arxiv_papers"

    data_path: str = "data/arxiv-metadata-oai-snapshot.json"
    model_archive_path: str = (
        "models/phi-3-pytorch-phi-3.5-mini-instruct-v2.tar.gz"
    )
    model_extracted_path: str = "models/phi-3.5-mini-instruct"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = _default_device()
    embedding_dim: int = 384

    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.3

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 3

    pdf_download_timeout: float = 30.0
    pdf_base_url: str = "https://arxiv.org/pdf"

    llm_timeout: float = 900.0
    api_request_timeout: float = 1200.0

    indexer_batch_size: int = 2000
    encoder_batch_size: int = 64
    log_level: str = "INFO"

    # Multi-tenancy
    tenant_db_path: str = "data/tenants.db"
    admin_api_key: str = "admin"
    default_rate_limit: int = 30
    base_chunk_size: int = 1000
    min_chunk_size: int = 400

    # Custom document uploads
    custom_documents_index: str = "custom_documents"
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50

    # Async task manager
    task_ttl_seconds: int = 3600
    task_poll_interval: float = 2.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def data_file(self) -> Path:
        return Path(self.data_path)

    @property
    def model_archive(self) -> Path:
        return Path(self.model_archive_path)

    @property
    def model_dir(self) -> Path:
        return Path(self.model_extracted_path)


settings = Settings()
