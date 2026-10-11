from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document


class CrossEncoderReranker:
    """Reranks candidate retrieval results using a transformer Cross-Encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _doc_to_dict(self, doc: Union[Dict[str, Any], Document]) -> Dict[str, Any]:
        if isinstance(doc, Document):
            data = dict(doc.metadata) if doc.metadata else {}
            data["text"] = doc.page_content
            return data
        return doc.copy()

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[Dict[str, Any], Document]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Compute cross-encoder relevance scores for (query, doc) pairs and sort descending."""
        if not documents:
            return []

        normalized_docs = [self._doc_to_dict(d) for d in documents]
        texts = [doc.get("text", "") for doc in normalized_docs]
        pairs = [[query, text] for text in texts]

        model = self._get_model()
        scores = model.predict(pairs)

        # Handle scalar or list score output
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        scored_docs = []
        for doc, score in zip(normalized_docs, scores):
            d = doc.copy()
            d["rerank_score"] = float(score)
            scored_docs.append(d)

        # Sort descending by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, item in enumerate(scored_docs, start=1):
            item["rerank_rank"] = rank

        return scored_docs[:top_k]

    def rerank_as_documents(
        self,
        query: str,
        documents: Sequence[Union[Dict[str, Any], Document]],
        top_k: int = 5,
    ) -> List[Document]:
        """Return reranked results formatted as LangChain Document objects."""
        hits = self.rerank(query, documents, top_k=top_k)
        docs = []
        for hit in hits:
            content = hit.get("text", "")
            metadata = {k: v for k, v in hit.items() if k != "text"}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs


def hybrid_search_and_rerank(
    hybrid_retriever: Any,
    reranker: CrossEncoderReranker,
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> List[Dict[str, Any]]:
    """Execute Phase 6 pipeline: Hybrid Retrieval (Top candidate_k) -> Cross-Encoder Rerank (Top top_k)."""
    candidates = hybrid_retriever.search(query=query, top_k=candidate_k, candidate_k=candidate_k)
    return reranker.rerank(query=query, documents=candidates, top_k=top_k)
