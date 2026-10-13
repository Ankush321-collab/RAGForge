import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from backend.app.retrieval.parent_child import ParentChildRetriever


def test_parent_child_retriever():
    retriever = ParentChildRetriever()

    parent_chunks = [
        {"parent_id": "p_1", "text": "This is full parent context number one.", "page_number": 1},
        {"parent_id": "p_2", "text": "This is full parent context number two.", "page_number": 2},
    ]

    retriever.add_parent_chunks(parent_chunks)

    assert retriever.get_parent_chunk("p_1")["text"] == "This is full parent context number one."
    assert retriever.get_parent_chunk("non_existent") is None

    child_hits = [
        {"child_id": "c_1", "parent_id": "p_1", "score": 0.92, "text": "context number one"},
        {"child_id": "c_2", "parent_id": "p_1", "score": 0.85, "text": "parent context"},  # duplicate parent
        {"child_id": "c_3", "parent_id": "p_2", "score": 0.78, "text": "context number two"},
    ]

    resolved_parents = retriever.resolve_parents(child_hits)

    assert len(resolved_parents) == 2
    assert resolved_parents[0]["parent_id"] == "p_1"
    assert resolved_parents[0]["child_score"] == 0.92
    assert resolved_parents[0]["retrieval_strategy"] == "parent_child"
    assert resolved_parents[1]["parent_id"] == "p_2"
    assert resolved_parents[1]["child_score"] == 0.78


def test_parent_child_search_and_resolve():
    retriever = ParentChildRetriever()
    retriever.add_parent_chunks([
        {"parent_id": "p_10", "text": "Complete parent section detailing polymorphism and vtables."},
    ])

    mock_child_retriever = MagicMock()
    mock_child_retriever.search.return_value = [
        {"child_id": "c_10_1", "parent_id": "p_10", "score": 0.95, "text": "vtables"}
    ]

    results = retriever.search_and_resolve(
        child_retriever=mock_child_retriever,
        query="what is a vtable?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["parent_id"] == "p_10"
    assert "Complete parent section" in results[0]["text"]
    assert results[0]["matched_child_id"] == "c_10_1"


def test_parent_child_search_resolve_and_rerank():
    retriever = ParentChildRetriever()
    retriever.add_parent_chunks([
        {"parent_id": "p_a", "text": "Parent A discussion."},
        {"parent_id": "p_b", "text": "Parent B discussion."},
    ])

    mock_child_retriever = MagicMock()
    mock_child_retriever.search.return_value = [
        {"child_id": "ca", "parent_id": "p_a", "score": 0.80},
        {"child_id": "cb", "parent_id": "p_b", "score": 0.75},
    ]

    mock_reranker = MagicMock()
    # Reranker prefers p_b over p_a
    mock_reranker.rerank.return_value = [
        {"parent_id": "p_b", "text": "Parent B discussion.", "rerank_score": 0.99},
        {"parent_id": "p_a", "text": "Parent A discussion.", "rerank_score": 0.12},
    ]

    reranked = retriever.search_resolve_and_rerank(
        child_retriever=mock_child_retriever,
        reranker=mock_reranker,
        query="test query",
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0]["parent_id"] == "p_b"
    assert reranked[0]["rerank_score"] == 0.99


def test_resolve_parents_as_documents():
    retriever = ParentChildRetriever()
    retriever.add_parent_chunks([
        {"parent_id": "p_doc", "text": "Parent document content", "doc_type": "parent"}
    ])

    child_docs = [
        Document(page_content="child snippet", metadata={"child_id": "c1", "parent_id": "p_doc"})
    ]

    resolved_docs = retriever.resolve_parents_as_documents(child_docs)
    assert len(resolved_docs) == 1
    assert isinstance(resolved_docs[0], Document)
    assert resolved_docs[0].page_content == "Parent document content"
    assert resolved_docs[0].metadata["parent_id"] == "p_doc"
