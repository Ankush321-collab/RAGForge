from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document

from .bm25 import BM25Retriever
from .vector import similarity_search


def get_doc_id(doc: Union[Dict[str, Any], Document], fallback_id: Optional[str] = None) -> str:
    """Extract a consistent identifier from a dictionary or Document."""
    if isinstance(doc, Document):
        metadata = doc.metadata or {}
        return str(
            metadata.get("chunk_id")
            or metadata.get("id")
            or metadata.get("doc_id")
            or fallback_id
            or hash(doc.page_content)
        )
    elif isinstance(doc, dict):
        return str(
            doc.get("chunk_id")
            or doc.get("id")
            or doc.get("doc_id")
            or (doc.get("metadata", {}).get("chunk_id") if isinstance(doc.get("metadata"), dict) else None)
            or fallback_id
            or hash(doc.get("text", "") or doc.get("page_content", ""))
        )
    return str(fallback_id or id(doc))


def doc_to_dict(doc: Union[Dict[str, Any], Document]) -> Dict[str, Any]:
    """Convert a Document or Dict into a normalized dictionary."""
    if isinstance(doc, Document):
        item = dict(doc.metadata) if doc.metadata else {}
        item["text"] = doc.page_content
        return item
    return doc.copy()


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Union[Dict[str, Any], Document]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    id_fn: Callable[[Union[Dict[str, Any], Document]], str] = get_doc_id,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked retrieval lists using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_score(d) = SUM_m [ weight_m / (k + rank_m(d)) ]
    
    where rank_m is 1-indexed.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Dict[str, Any]] = {}
    ranks_per_system: Dict[str, Dict[int, int]] = {}

    for system_idx, (ranked_list, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = id_fn(item)
            if doc_id not in doc_registry:
                doc_registry[doc_id] = doc_to_dict(item)
                ranks_per_system[doc_id] = {}

            ranks_per_system[doc_id][system_idx] = rank
            score_contribution = weight / (k + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score_contribution

    # Sort documents by fused score descending
    sorted_doc_ids = sorted(fused_scores.keys(), key=lambda did: fused_scores[did], reverse=True)

    results: List[Dict[str, Any]] = []
    for rank, doc_id in enumerate(sorted_doc_ids, start=1):
        doc = doc_registry[doc_id].copy()
        doc["rrf_score"] = fused_scores[doc_id]
        doc["rrf_rank"] = rank
        doc["source_ranks"] = ranks_per_system[doc_id]
        results.append(doc)

    return results


class HybridRetriever:
    """Combines BM25 sparse retrieval and dense vector retrieval using RRF."""

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        k_rrf: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.k_rrf = k_rrf
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        mode: str = "hybrid",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents using 'hybrid', 'vector_only', or 'bm25_only' with metadata filtering."""
        mode = mode.lower()

        if mode == "bm25_only":
            if not self.bm25_retriever:
                return []
            return self.bm25_retriever.search(query, top_k=top_k, filter=filter)

        if mode == "vector_only":
            if not self.vector_store:
                return []
            try:
                vector_docs = similarity_search(self.vector_store, query, k=top_k, filter=filter)
            except Exception:
                vector_docs = similarity_search(self.vector_store, query, k=top_k)
            hits = [doc_to_dict(d) for d in vector_docs]
            if filter:
                from .bm25 import matches_metadata_filter
                hits = [h for h in hits if matches_metadata_filter(h, filter)]
            return hits[:top_k]

        if mode != "hybrid":
            raise ValueError(f"Unknown retrieval mode: '{mode}'. Choose 'hybrid', 'vector_only', or 'bm25_only'.")

        # Hybrid: retrieve candidates from both systems
        bm25_hits: List[Dict[str, Any]] = []
        if self.bm25_retriever:
            bm25_hits = self.bm25_retriever.search(query, top_k=candidate_k, filter=filter)

        vector_hits: List[Dict[str, Any]] = []
        if self.vector_store:
            try:
                vector_docs = similarity_search(self.vector_store, query, k=candidate_k, filter=filter)
            except Exception:
                vector_docs = similarity_search(self.vector_store, query, k=candidate_k)
            vector_hits = [doc_to_dict(d) for d in vector_docs]
            if filter:
                from .bm25 import matches_metadata_filter
                vector_hits = [h for h in vector_hits if matches_metadata_filter(h, filter)]

        fused = reciprocal_rank_fusion(
            ranked_lists=[bm25_hits, vector_hits],
            k=self.k_rrf,
            weights=[self.bm25_weight, self.vector_weight],
        )

        return fused[:top_k]

    def multi_query_search(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        candidate_k: int = 20,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """Phase 8: Retrieve for multiple queries and fuse results using RRF."""
        ranked_lists = []
        for q in queries:
            hits = self.search(q, top_k=candidate_k, candidate_k=candidate_k, mode=mode)
            ranked_lists.append(hits)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k=self.k_rrf,
        )
        return fused[:top_k]

    def search_as_documents(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        mode: str = "hybrid",
    ) -> List[Document]:
        """Retrieve results wrapped as LangChain Document objects."""
        hits = self.search(query, top_k=top_k, candidate_k=candidate_k, mode=mode)
        docs = []
        for hit in hits:
            content = hit.get("text", "")
            metadata = {k: v for k, v in hit.items() if k != "text"}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

