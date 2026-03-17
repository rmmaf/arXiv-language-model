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

        await self.client.indices.create(index=settings.index_name, body=mapping)
        logger.info("Index '%s' created with hybrid mapping", settings.index_name)

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        tenant_id: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a hybrid BM25 + kNN search scoped to *tenant_id*.

        Uses Elasticsearch 8.x native kNN for the vector part (HNSW index)
        instead of brute-force script_score, which is critical for large indices.
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

        # #region agent log
        import json as _json, time as _time
        try:
            _idx_exists = await self.client.indices.exists(index=settings.index_name)
            _doc_count_resp = await self.client.count(index=settings.index_name) if _idx_exists else None
            _doc_count = _doc_count_resp["count"] if _doc_count_resp else 0
            _tenant_count_resp = await self.client.count(index=settings.index_name, body={"query": tenant_filter}) if _idx_exists else None
            _tenant_count = _tenant_count_resp["count"] if _tenant_count_resp else 0
            with open("data/debug-36bb97.log", "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({"sessionId":"36bb97","hypothesisId":"H6","location":"elastic.py:hybrid_search","message":"pre-search with fallback filter","data":{"index_name":settings.index_name,"total_doc_count":_doc_count,"tenant_id_queried":tenant_id,"docs_matching_filter":_tenant_count},"timestamp":int(_time.time()*1000)}) + "\n")
        except Exception as _e:
            with open("data/debug-36bb97.log", "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({"sessionId":"36bb97","hypothesisId":"H6","location":"elastic.py:hybrid_search","message":"pre-search diagnostics FAILED","data":{"error":str(_e)},"timestamp":int(_time.time()*1000)}) + "\n")
        # #endregion

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

        # #region agent log
        import json as _json2, time as _time2
        try:
            with open("data/debug-36bb97.log", "a", encoding="utf-8") as _f:
                _f.write(_json2.dumps({"sessionId":"36bb97","hypothesisId":"H1-H2-H5","location":"elastic.py:hybrid_search:post","message":"search results","data":{"total_hits":response["hits"]["total"]["value"],"returned_count":len(results),"tenant_id":tenant_id,"query_text_prefix":query_text[:80]},"timestamp":int(_time2.time()*1000)}) + "\n")
        except Exception:
            pass
        # #endregion

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
