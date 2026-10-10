from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth

from backend.app.core.config import get_settings
from backend.app.ingestion.embedding import get_model


class OpenSearchVectorStore:
    """Manages indexing and hybrid search (Vector + BM25) on AWS OpenSearch."""

    def __init__(
        self,
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        settings = get_settings()
        self.host = host or settings.opensearch_host
        self.username = username or settings.opensearch_username
        self.password = password or settings.opensearch_password
        self.index_name = index_name or settings.opensearch_index_name
        self.embedding_model_name = embedding_model_name

        if not self.host:
            raise ValueError("OpenSearch host URL is required. Set OPENSEARCH_HOST in .env")

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

        self.client = OpenSearch(
            hosts=[{"host": hostname, "port": port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Bulk index chunk records into OpenSearch using bulk operations."""
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

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense vector similarity search using 384-dim query embedding."""
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

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Sparse BM25 keyword search."""
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
