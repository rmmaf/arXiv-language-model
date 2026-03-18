import logging
from typing import Any

from elasticsearch import AsyncElasticsearch, helpers

from src.core.config import settings

logger = logging.getLogger(__name__)


class ElasticClient:
    """Async Elasticsearch wrapper with hybrid search support."""

    def __init__(self) -> None:
        self._client: AsyncElasticsearch | None = None

    async def connect(self) -> None:
        self._client = AsyncElasticsearch(hosts=[settings.elasticsearch_url])
        info = await self._client.info()
        logger.info(
            "Connected to Elasticsearch cluster: %s",
            info["cluster_name"],
        )

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Elasticsearch connection closed")

    @property
    def client(self) -> AsyncElasticsearch:
        if self._client is None:
            raise RuntimeError(
                "ElasticClient is not connected. "
                "Call connect() first."
            )
        return self._client

    async def create_index(self) -> None:
        """Create the arXiv papers index with hybrid mapping."""
        if await self.client.indices.exists(index=settings.index_name):
            logger.info(
                "Index '%s' already exists, skipping creation",
                settings.index_name,
            )
            return

        mapping: dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "paper_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "abstract": {"type": "text", "analyzer": "standard"},
                    "categories": {"type": "keyword"},
                    "authors": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": settings.embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }

        await self.client.indices.create(
            index=settings.index_name, body=mapping,
        )
        logger.info(
            "Index '%s' created with hybrid mapping",
            settings.index_name,
        )

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        tenant_id: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a hybrid BM25 + kNN search scoped to *tenant_id*.

        Uses Elasticsearch 8.x native kNN for the vector part (HNSW index)
        instead of brute-force script_score, which is critical
        for large indices.
        """
        k = top_k or settings.top_k_results
        knn_candidates = max(k * 10, 100)
        tenant_filter: dict[str, Any] = {
            "bool": {
                "should": [
                    {"term": {"tenant_id.keyword": tenant_id}},
                    {"bool": {"must_not": {"exists": {"field": "tenant_id"}}}},
                ],
                "minimum_should_match": 1,
            }
        }

        response = await self.client.search(
            index=settings.index_name,
            size=k,
            query={
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title^2", "abstract"],
                            "type": "best_fields",
                        }
                    },
                    "filter": tenant_filter,
                }
            },
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": knn_candidates,
                "filter": tenant_filter,
            },
            source=["paper_id", "title", "abstract", "categories", "authors"],
        )

        results: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["score"] = hit["_score"]
            results.append(doc)

        logger.info(
            "Hybrid search returned %d results for query: %.80s...",
            len(results), query_text,
        )
        return results

    # ---- Custom documents index ----

    async def create_custom_documents_index(self) -> None:
        """Create the index for custom uploaded PDF chunks."""
        idx = settings.custom_documents_index
        if await self.client.indices.exists(index=idx):
            logger.info("Index '%s' already exists, skipping creation", idx)
            return

        mapping: dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "filename": {"type": "text"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "chunk_index": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": settings.embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }

        await self.client.indices.create(index=idx, body=mapping)
        logger.info("Index '%s' created for custom documents", idx)

    async def index_custom_chunks(
        self,
        document_id: str,
        tenant_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Bulk-index chunks from a custom uploaded document."""
        idx = settings.custom_documents_index
        actions = [
            {
                "_index": idx,
                "_id": f"{document_id}_{i}",
                "_source": {
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "filename": filename,
                    "content": chunk,
                    "chunk_index": i,
                    "embedding": embeddings[i],
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        success_count, errors = await helpers.async_bulk(
            self.client, actions, raise_on_error=False
        )
        if errors:
            logger.warning("Custom doc indexing had %d errors", len(errors))
        logger.info(
            "Indexed %d chunks for document %s",
            success_count, document_id,
        )
        return success_count

    async def search_custom_documents(
        self,
        query_vector: list[float],
        tenant_id: str,
        document_ids: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """kNN search over custom document chunks."""
        idx = settings.custom_documents_index

        if not await self.client.indices.exists(index=idx):
            return []

        doc_filter: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"tenant_id": tenant_id}},
                    {"terms": {"document_id": document_ids}},
                ],
            }
        }

        knn_candidates = max(top_k * 10, 100)

        response = await self.client.search(
            index=idx,
            size=top_k,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": knn_candidates,
                "filter": doc_filter,
            },
            source=["document_id", "filename", "content", "chunk_index"],
        )

        results: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["score"] = hit["_score"]
            results.append(doc)

        logger.info(
            "Custom document search returned %d chunks for tenant %s",
            len(results), tenant_id,
        )
        return results

    async def delete_custom_document(
        self, document_id: str, tenant_id: str,
    ) -> int:
        """Delete all chunks for a single custom document."""
        idx = settings.custom_documents_index
        if not await self.client.indices.exists(index=idx):
            return 0
        resp = await self.client.delete_by_query(
            index=idx,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"document_id": document_id}},
                            {"term": {"tenant_id": tenant_id}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        deleted = resp.get("deleted", 0)
        logger.info("Deleted %d chunks for document %s", deleted, document_id)
        return deleted

    async def delete_custom_documents_by_tenant(self, tenant_id: str) -> int:
        """Delete all custom document chunks belonging to a tenant."""
        idx = settings.custom_documents_index
        if not await self.client.indices.exists(index=idx):
            return 0
        resp = await self.client.delete_by_query(
            index=idx,
            body={"query": {"term": {"tenant_id": tenant_id}}},
            refresh=True,
        )
        deleted = resp.get("deleted", 0)
        logger.info(
            "Deleted %d custom chunks for tenant %s",
            deleted, tenant_id,
        )
        return deleted

    # ---- arXiv bulk indexing ----

    async def bulk_index(self, documents: list[dict[str, Any]]) -> int:
        """Bulk-index a list of documents.

        Returns the number of successfully indexed docs.
        """
        actions = [
            {
                "_index": settings.index_name,
                "_id": doc["paper_id"],
                "_source": doc,
            }
            for doc in documents
        ]

        success_count, errors = await helpers.async_bulk(
            self.client, actions, raise_on_error=False
        )

        if errors:
            logger.warning("Bulk indexing encountered %d errors", len(errors))

        return success_count
