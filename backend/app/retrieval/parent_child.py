from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document

from backend.app.retrieval.hybrid import doc_to_dict


class ParentChildRetriever:
    """Phase 10: Manages parent chunk storage and maps retrieved child chunks back to parent contexts."""

    def __init__(self) -> None:
        self._docstore: Dict[str, Dict[str, Any]] = {}

    def add_parent_chunks(self, parent_chunks: Sequence[Union[Dict[str, Any], Document]]) -> None:
        """Store parent chunks in docstore keyed by parent_id."""
        for chunk in parent_chunks:
            normalized = doc_to_dict(chunk)
            parent_id = str(normalized.get("parent_id", "") or normalized.get("id", ""))
            if parent_id:
                self._docstore[parent_id] = normalized

    def get_parent_chunk(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve parent chunk by ID."""
        return self._docstore.get(parent_id)

    def resolve_parents(
        self,
        child_hits: Sequence[Union[Dict[str, Any], Document]],
    ) -> List[Dict[str, Any]]:
        """Maps child hits to their parent chunks, removing duplicates while retaining hit order.
        
        If a child hit doesn't specify parent_id or the parent is not in docstore,
        the child hit itself is retained.
        """
        resolved = []
        seen_parents = set()

        for hit in child_hits:
            hit_dict = doc_to_dict(hit)
            parent_id = str(hit_dict.get("parent_id", "") or hit_dict.get("metadata", {}).get("parent_id", ""))
            if parent_id and parent_id in self._docstore:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_doc = self._docstore[parent_id].copy()
                    score_val = hit_dict.get("score") if hit_dict.get("score") is not None else hit_dict.get("rrf_score")
                    parent_doc["score"] = score_val
                    parent_doc["child_score"] = score_val
                    parent_doc["matched_child_id"] = hit_dict.get("child_id") or hit_dict.get("id")
                    parent_doc["retrieval_strategy"] = "parent_child"
                    resolved.append(parent_doc)

            else:
                resolved.append(hit_dict)

        return resolved

    def search_and_resolve(
        self,
        child_retriever: Any,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search child chunk index and resolve returned hits to parent context."""
        if hasattr(child_retriever, "search"):
            child_hits = child_retriever.search(query, top_k=candidate_k)
        elif hasattr(child_retriever, "similarity_search"):
            child_hits = child_retriever.similarity_search(query, k=candidate_k)
        elif callable(child_retriever):
            child_hits = child_retriever(query, candidate_k)
        else:
            raise AttributeError("Child retriever must have search(), similarity_search(), or be callable.")

        resolved_parents = self.resolve_parents(child_hits)
        return resolved_parents[:top_k]

    def search_resolve_and_rerank(
        self,
        child_retriever: Any,
        reranker: Any,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Execute full Phase 10 + Phase 6 pipeline:
        Child Retrieval -> Parent Context Resolution -> Cross-Encoder Reranking.
        """
        resolved_parents = self.search_and_resolve(
            child_retriever=child_retriever,
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )
        if hasattr(reranker, "rerank"):
            return reranker.rerank(query=query, documents=resolved_parents, top_k=top_k)
        return resolved_parents[:top_k]

    def resolve_parents_as_documents(
        self,
        child_hits: Sequence[Union[Dict[str, Any], Document]],
    ) -> List[Document]:
        """Maps child hits to parent chunks and returns as LangChain Document objects."""
        parents = self.resolve_parents(child_hits)
        docs = []
        for p in parents:
            content = p.get("text", "") or p.get("page_content", "")
            metadata = {k: v for k, v in p.items() if k not in ("text", "page_content")}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def clear(self) -> None:
        """Clear docstore."""
        self._docstore.clear()
