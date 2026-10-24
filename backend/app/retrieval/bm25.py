import math
import re
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def default_tokenizer(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25.
    
    Preserves numbers, acronyms, and technical tokens (e.g., 'gpt-4', 'v1.2').
    """
    if not text:
        return []
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b", text.lower())
    return tokens if tokens else text.lower().split()


def matches_metadata_filter(doc: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
    """Evaluate whether document metadata satisfies key-value and operator filter constraints."""
    if not metadata_filter:
        return True

    for key, condition in metadata_filter.items():
        doc_val = doc.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                if op == "$eq" and doc_val != target:
                    return False
                elif op == "$ne" and doc_val == target:
                    return False
                elif op == "$gt" and (doc_val is None or doc_val <= target):
                    return False
                elif op == "$gte" and (doc_val is None or doc_val < target):
                    return False
                elif op == "$lt" and (doc_val is None or doc_val >= target):
                    return False
                elif op == "$lte" and (doc_val is None or doc_val > target):
                    return False
                elif op == "$in" and doc_val not in target:
                    return False
                elif op == "$nin" and doc_val in target:
                    return False
        else:
            if doc_val != condition:
                return False
    return True


class BM25Retriever:
    """In-memory BM25 sparse retriever using BM25Okapi with Lucene-smoothed IDF and metadata filtering."""

    def __init__(
        self,
        corpus: Optional[Sequence[Union[Dict[str, Any], Document]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer=default_tokenizer,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []

        if corpus:
            self.index(corpus)

    def _normalize_doc(self, doc: Union[Dict[str, Any], Document]) -> Dict[str, Any]:
        """Normalize a dictionary or LangChain Document into a standard dictionary format."""
        if isinstance(doc, Document):
            data = dict(doc.metadata) if doc.metadata else {}
            data["text"] = doc.page_content
            return data
        elif isinstance(doc, dict):
            data = doc.copy()
            if "text" not in data and "page_content" in data:
                data["text"] = data["page_content"]
            return data
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    def index(self, corpus: Sequence[Union[Dict[str, Any], Document]]) -> None:
        """Index a corpus of documents or chunks."""
        self.documents = [self._normalize_doc(d) for d in corpus]
        tokenized_corpus = [
            self.tokenizer(doc.get("text", "")) for doc in self.documents
        ]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            num_docs = len(tokenized_corpus)
            nd: Dict[str, int] = {}
            for doc_tokens in tokenized_corpus:
                for term in set(doc_tokens):
                    nd[term] = nd.get(term, 0) + 1

            for term, freq in nd.items():
                self.bm25.idf[term] = math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
        else:
            self.bm25 = None

    def search(
        self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k documents with BM25 score with optional metadata filtering."""
        if not self.bm25 or not self.documents:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        scored_indices = [
            (idx, score)
            for idx, score in enumerate(scores)
            if score > 0 and (filter is None or matches_metadata_filter(self.documents[idx], filter))
        ]

        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices[:top_k]:
            doc_copy = self.documents[idx].copy()
            doc_copy["score"] = float(score)
            results.append(doc_copy)

        return results

    def search_as_documents(
        self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve top_k matches converted into LangChain Document objects."""
        hits = self.search(query, top_k=top_k, filter=filter)
        docs = []
        for hit in hits:
            content = hit.get("text", "")
            metadata = {k: v for k, v in hit.items() if k != "text"}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def clear(self) -> None:
        """Clear the indexed documents and BM25 model."""
        self.documents = []
        self.bm25 = None
