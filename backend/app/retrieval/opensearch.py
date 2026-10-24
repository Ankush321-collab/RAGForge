import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth

from backend.app.core.config import get_settings
from backend.app.ingestion.embedding import get_model
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.vector import create_vector_store

logger = logging.getLogger(__name__)


class OpenSearchVectorStore:
    """Manages indexing and hybrid search (Vector + BM25) on AWS OpenSearch with local fallback."""

    def __init__(
        self,
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_local_fallback: bool = True,
    ) -> None:
        settings = get_settings()
        self.host = host or settings.opensearch_host
        self.username = username or settings.opensearch_username
        self.password = password or settings.opensearch_password
        self.index_name = index_name or settings.opensearch_index_name
        self.embedding_model_name = embedding_model_name
        self.use_local_fallback = use_local_fallback
        self.client: Optional[OpenSearch] = None

        # Local fallback store
        self._local_chunks: List[Dict[str, Any]] = []
        self._local_chroma = None
        self._local_bm25 = None

        if self.host:
            parsed = urlparse(self.host if self.host.startswith("http") else f"https://{self.host}")
            use_ssl = parsed.scheme == "https"
            port = parsed.port or (443 if use_ssl else 9200)
            hostname = parsed.hostname or self.host

            is_aoss = ".aoss." in hostname

            if is_aoss:
                access_key = settings.s3_access_key_id
                secret_key = settings.s3_secret_access_key
                region = settings.s3_region or "ap-south-2"
                auth = AWS4Auth(access_key, secret_key, region, "aoss")
            else:
                auth = (self.username, self.password) if self.username and self.password else None

            try:
                self.client = OpenSearch(
                    hosts=[{"host": hostname, "port": port}],
                    http_auth=auth,
                    use_ssl=use_ssl,
                    verify_certs=use_ssl,
                    ssl_show_warn=False,
                    connection_class=RequestsHttpConnection,
                    timeout=5,
                    max_retries=1,
                    retry_on_timeout=False,
                )
                # Verify live connectivity
                self.client.info()
            except Exception as e:
                logger.warning("Failed to initialize OpenSearch client (%s). Using local fallback.", e)
                self.client = None

    def ensure_index(self, dimension: int = 384) -> bool:
        """Ensure that the vector index with knn mapping exists in OpenSearch."""
        if not self.client:
            return False
        try:
            if not self.client.indices.exists(index=self.index_name):
                index_body = {
                    "settings": {
                        "index": {
                            "knn": True,
                        }
                    },
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "vector": {
                                "type": "knn_vector",
                                "dimension": dimension,
                                "space_type": "cosinesimil",
                            },
                            "chunk_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "parent_id": {"type": "keyword"},
                            "page_number": {"type": "integer"},
                            "strategy": {"type": "keyword"},
                        }
                    },
                }
                self.client.indices.create(index=self.index_name, body=index_body)
                logger.info("Created OpenSearch index '%s' with knn mapping.", self.index_name)
            return True
        except Exception as e:
            logger.warning("ensure_index failed on OpenSearch (%s).", e)
            return False

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Bulk index chunk records into OpenSearch with fallback to local store."""
        if not chunks:
            return 0

        # Ensure embeddings are attached if missing
        texts_to_embed = [c["text"] for c in chunks if "embedding" not in c]
        if texts_to_embed:
            model = get_model(self.embedding_model_name)
            embeddings = model.encode(texts_to_embed)
            embed_idx = 0
            for chunk in chunks:
                if "embedding" not in chunk:
                    chunk["embedding"] = embeddings[embed_idx].tolist()
                    embed_idx += 1

        if self.client:
            try:
                dim = len(chunks[0].get("embedding", [])) or 384
                self.ensure_index(dimension=dim)
                actions = []
                for idx, chunk in enumerate(chunks, start=1):
                    doc_id = f"{chunk.get('document_id', 'doc')}_{chunk.get('chunk_id', idx)}"
                    doc_body = {
                        "text": chunk["text"],
                        "vector": chunk["embedding"].tolist() if hasattr(chunk["embedding"], "tolist") else chunk["embedding"],
                        "chunk_id": str(chunk.get("chunk_id", "")),
                        "document_id": chunk.get("document_id"),
                        "parent_id": chunk.get("parent_id"),
                        "page_number": chunk.get("page_number", 1),
                        "strategy": chunk.get("strategy", "unknown"),
                    }
                    actions.append({
                        "_index": self.index_name,
                        "_id": doc_id,
                        "_source": doc_body,
                    })

                success_count, _ = helpers.bulk(self.client, actions)
                return success_count
            except Exception as e:
                logger.warning("OpenSearch bulk indexing failed (%s). Falling back to local store.", e)

        # Fallback to local store
        self._local_chunks = list(chunks)
        try:
            self._local_chroma = create_vector_store(
                self._local_chunks,
                collection_name=f"local_{self.index_name}",
                persist_directory=None,
                model_name=self.embedding_model_name,
            )
        except Exception:
            pass
        self._local_bm25 = BM25Retriever(self._local_chunks)
        return len(chunks)

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense vector similarity search using 384-dim query embedding."""
        if self.client:
            try:
                query_embedding = get_model(self.embedding_model_name).encode(query).tolist()
                query_body = {
                    "size": top_k,
                    "query": {
                        "knn": {
                            "vector": {
                                "vector": query_embedding,
                                "k": top_k,
                            }
                        }
                    },
                }
                response = self.client.search(index=self.index_name, body=query_body)
                return [
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        **hit["_source"],
                    }
                    for hit in response["hits"]["hits"]
                ]
            except Exception as e:
                logger.warning("OpenSearch vector search failed (%s); using local fallback.", e)

        # Local fallback search
        if self._local_chroma:
            hits = self._local_chroma.similarity_search(query, k=top_k)
            return [
                {
                    "id": h.metadata.get("chunk_id") or "chunk",
                    "score": 1.0,
                    "text": h.page_content,
                    **h.metadata,
                }
                for h in hits
            ]

        # In-memory cosine fallback
        if self._local_chunks:
            import numpy as np
            q_emb = np.array(get_model(self.embedding_model_name).encode(query), dtype=np.float32)
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm

            scored = []
            for c in self._local_chunks:
                emb = np.array(c.get("embedding", []), dtype=np.float32)
                norm = np.linalg.norm(emb)
                score = float(np.dot(emb / (norm if norm > 0 else 1e-9), q_emb))
                scored.append({"id": c.get("chunk_id"), "score": score, **c})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        return []

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Sparse BM25 keyword search."""
        if self.client:
            try:
                query_body = {
                    "size": top_k,
                    "query": {
                        "match": {
                            "text": query
                        }
                    },
                }
                response = self.client.search(index=self.index_name, body=query_body)
                return [
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        **hit["_source"],
                    }
                    for hit in response["hits"]["hits"]
                ]
            except Exception as e:
                logger.warning("OpenSearch BM25 search failed (%s); using local fallback.", e)

        if self._local_bm25:
            return self._local_bm25.search(query, top_k=top_k)
        return []
