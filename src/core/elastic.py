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
        logger.info("Connected to Elasticsearch cluster: %s", info["cluster_name"])

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Elasticsearch connection closed")

    @property
    def client(self) -> AsyncElasticsearch:
        if self._client is None:
            raise RuntimeError("ElasticClient is not connected. Call connect() first.")
        return self._client

    async def create_index(self) -> None:
        """Create the arXiv papers index with hybrid (BM25 + dense_vector) mapping."""
        if await self.client.indices.exists(index=settings.index_name):
            logger.info("Index '%s' already exists, skipping creation", settings.index_name)
            return

        mapping: dict[str, Any] = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "paper_id": {"type": "keyword"},
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

        await self.client.indices.create(index=settings.index_name, body=mapping)
        logger.info("Index '%s' created with hybrid mapping", settings.index_name)

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a hybrid BM25 + dense vector search and return top_k results."""
        k = top_k or settings.top_k_results

        query_body: dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["title^2", "abstract"],
                                "type": "best_fields",
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_vector},
                                },
                            }
                        },
                    ]
                }
            },
            "_source": ["paper_id", "title", "abstract", "categories", "authors"],
        }

        response = await self.client.search(index=settings.index_name, body=query_body)

        results: list[dict[str, Any]] = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["score"] = hit["_score"]
            results.append(doc)

        logger.info("Hybrid search returned %d results for query: %.80s...", len(results), query_text)
        return results

    async def bulk_index(self, documents: list[dict[str, Any]]) -> int:
        """Bulk-index a list of documents. Returns the number of successfully indexed docs."""
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
