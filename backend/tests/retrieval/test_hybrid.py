import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.hybrid import (
    HybridRetriever,
    get_doc_id,
    reciprocal_rank_fusion,
)


def test_get_doc_id():
    d1 = {"chunk_id": "c1", "text": "hello"}
    assert get_doc_id(d1) == "c1"

    d2 = Document(page_content="world", metadata={"id": "doc2"})
    assert get_doc_id(d2) == "doc2"


def test_reciprocal_rank_fusion_math():
    list_a = [
        {"chunk_id": "doc_both", "text": "Common hit"},
        {"chunk_id": "doc_a_only", "text": "A only"},
    ]
    list_b = [
        {"chunk_id": "doc_both", "text": "Common hit"},
        {"chunk_id": "doc_b_only", "text": "B only"},
    ]

    # k=60
    # doc_both rank=1 in both: 1/(60+1) + 1/(60+1) = 2/61 ~ 0.032787
    # doc_a_only rank=2 in A: 1/(60+2) = 1/62 ~ 0.016129
    # doc_b_only rank=2 in B: 1/(60+2) = 1/62 ~ 0.016129
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "doc_both"
    expected_top_score = (1.0 / 61) + (1.0 / 61)
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-4) == expected_top_score
    assert fused[0]["rrf_rank"] == 1


def test_hybrid_retriever_modes():
    bm25 = BM25Retriever(
        [
            {"chunk_id": "c1", "text": "Quantum computing qubit algorithm"},
            {"chunk_id": "c2", "text": "Deep learning neural network gradient"},
        ]
    )

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = [
        Document(page_content="Deep learning neural network gradient", metadata={"chunk_id": "c2"}),
        Document(page_content="Support vector machines kernel", metadata={"chunk_id": "c3"}),
    ]

    retriever = HybridRetriever(vector_store=mock_vector_store, bm25_retriever=bm25)

    # 1. BM25 only
    bm25_results = retriever.search("qubit", mode="bm25_only")
    assert len(bm25_results) == 1
    assert bm25_results[0]["chunk_id"] == "c1"

    # 2. Vector only
    vector_results = retriever.search("neural network", mode="vector_only")
    assert len(vector_results) == 2
    assert vector_results[0]["chunk_id"] == "c2"

    # 3. Hybrid
    hybrid_results = retriever.search("neural network", mode="hybrid")
    assert len(hybrid_results) > 0
    # c2 was returned by both or top rank, so it should rank first in hybrid
    assert hybrid_results[0]["chunk_id"] == "c2"
    assert "rrf_score" in hybrid_results[0]
    assert "source_ranks" in hybrid_results[0]


def test_hybrid_retriever_search_as_documents():
    bm25 = BM25Retriever(
        [
            {"chunk_id": "doc_1", "text": "BM25 keyword retriever"},
        ]
    )
    retriever = HybridRetriever(bm25_retriever=bm25)
    docs = retriever.search_as_documents("BM25", mode="hybrid")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "BM25 keyword retriever"
    assert docs[0].metadata["chunk_id"] == "doc_1"
    assert "rrf_score" in docs[0].metadata
