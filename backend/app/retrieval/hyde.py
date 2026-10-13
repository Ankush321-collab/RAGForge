import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.app.retrieval.hybrid import doc_to_dict, reciprocal_rank_fusion
from backend.app.retrieval.query_transform import _resolve_llm
from backend.app.retrieval.vector import similarity_search

load_dotenv()

DEFAULT_HYDE_SYSTEM_PROMPT = (
    "You are a knowledgeable technical expert writing an authoritative reference document.\n"
    "Given a user query, write a concise, plausible, and factual passage or textbook excerpt that directly answers the question.\n"
    "Include key domain concepts, technical terminology, and explanatory details that would appear in a top-tier technical manual or textbook.\n"
    "Do NOT write conversational greetings, disclaimers, or intros like 'Here is an explanation...'.\n"
    "Provide ONLY the technical passage text."
)


_UNSET = object()


class HyDEGenerator:
    """Phase 9: Generates hypothetical document excerpts to serve as dense vector retrieval queries."""

    def __init__(
        self,
        llm: Any = _UNSET,
        system_prompt: str = DEFAULT_HYDE_SYSTEM_PROMPT,
        model_name: Optional[str] = None,
    ) -> None:
        self.llm = _resolve_llm(model_name=model_name) if llm is _UNSET else llm
        self.system_prompt = system_prompt

    def generate_hypothetical_document(self, query: str) -> Dict[str, str]:
        """Generate a hypothetical document answering the query.
        
        Important: The hypothetical document is NOT the final answer; it is purely a retrieval aid.
        """
        clean_query = query.strip()
        if not clean_query:
            return {"original_query": query, "hypothetical_document": ""}

        if self.llm is None:
            # Fallback when LLM is unavailable: use original query as the retrieval text
            return {
                "original_query": query,
                "hypothetical_document": clean_query,
            }

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Question: {clean_query}"),
        ]

        try:
            response = self.llm.invoke(messages)
            passage = response.content if hasattr(response, "content") else str(response)
            passage = passage.strip().strip('"\'')
            if not passage:
                passage = clean_query
        except Exception:
            passage = clean_query

        return {
            "original_query": query,
            "hypothetical_document": passage,
        }


class HyDERetriever:
    """Phase 9: Retrieves real documents by embedding an LLM-generated hypothetical document."""

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        generator: Optional[HyDEGenerator] = None,
        llm: Any = _UNSET,
    ) -> None:
        self.vector_store = vector_store
        self.generator = generator or HyDEGenerator(llm=llm)


    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run HyDE pipeline: Query -> Hypothetical Answer -> Vector Search -> Real Documents."""
        if not query.strip():
            return []

        hyde_result = self.generator.generate_hypothetical_document(query)
        hypothetical_doc = hyde_result.get("hypothetical_document", query)

        if not self.vector_store:
            return []

        # Vector search using the hypothetical document text
        if hasattr(self.vector_store, "vector_search"):
            # OpenSearchVectorStore
            hits = self.vector_store.vector_search(hypothetical_doc, top_k=top_k)
        elif hasattr(self.vector_store, "similarity_search"):
            # Chroma or standard LangChain vector store
            raw_docs = similarity_search(self.vector_store, hypothetical_doc, k=top_k)
            hits = [doc_to_dict(d) for d in raw_docs]
        elif callable(self.vector_store):
            hits = self.vector_store(hypothetical_doc, top_k)
        else:
            raise AttributeError("Vector store must have vector_search or similarity_search method.")

        normalized_hits = []
        for rank, h in enumerate(hits, start=1):
            doc = doc_to_dict(h)
            doc["retrieval_strategy"] = "hyde"
            doc["hyde_rank"] = rank
            doc["hypothetical_document"] = hypothetical_doc
            doc["original_query"] = query
            normalized_hits.append(doc)

        return normalized_hits

    def search_as_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Return HyDE retrieved hits as LangChain Document objects."""
        hits = self.search(query, top_k=top_k)
        docs = []
        for hit in hits:
            content = hit.get("text", "")
            metadata = {k: v for k, v in hit.items() if k != "text"}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs


def hybrid_hyde_search(
    hybrid_retriever: Any,
    hyde_retriever: HyDERetriever,
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    k_rrf: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Phase 9 Hybrid + HyDE pipeline:
    
    Combines BM25 keyword retrieval (on raw query) with HyDE dense vector retrieval
    (on generated hypothetical document) and fuses them with Reciprocal Rank Fusion.
    """
    if not query.strip():
        return []

    # 1. Sparse keyword retrieval using raw query
    bm25_hits: List[Dict[str, Any]] = []
    if hasattr(hybrid_retriever, "bm25_retriever") and hybrid_retriever.bm25_retriever:
        bm25_hits = hybrid_retriever.bm25_retriever.search(query, top_k=candidate_k)
    elif hasattr(hybrid_retriever, "search"):
        try:
            bm25_hits = hybrid_retriever.search(query, top_k=candidate_k, mode="bm25_only")
        except Exception:
            bm25_hits = []

    # 2. HyDE dense vector retrieval using hypothetical document
    hyde_hits = hyde_retriever.search(query, top_k=candidate_k)

    # 3. Fuse with RRF
    fused = reciprocal_rank_fusion(
        ranked_lists=[bm25_hits, hyde_hits],
        k=k_rrf,
        weights=weights or [1.0, 1.0],
    )

    for doc in fused:
        doc["retrieval_strategy"] = "hybrid_hyde"

    return fused[:top_k]
