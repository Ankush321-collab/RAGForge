import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from backend.app.retrieval.query_transform import QueryExpander, expand_query, multi_query_search
from backend.app.retrieval.hybrid import HybridRetriever


def test_expand_query_helper():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="1. What is polymorphism in Python?\n2. Object oriented polymorphism concepts\n3. Method overriding and polymorphism"
    )

    expanded = expand_query("What is polymorphism?", num_queries=3, llm=mock_llm)
    assert expanded["original_query"] == "What is polymorphism?"
    assert len(expanded["expanded_queries"]) == 3
    assert len(expanded["all_queries"]) == 4


def test_multi_query_search_rrf_fusion():
    mock_retriever = MagicMock()
    # Query 1 finds doc_a and doc_b
    # Query 2 finds doc_b and doc_c
    mock_retriever.search.side_effect = [
        [{"chunk_id": "doc_a", "text": "About encapsulation", "score": 0.9},
         {"chunk_id": "doc_b", "text": "About polymorphism", "score": 0.8}],
        [{"chunk_id": "doc_b", "text": "About polymorphism", "score": 0.95},
         {"chunk_id": "doc_c", "text": "About inheritance", "score": 0.7}],
    ]

    queries = ["OOP encapsulation", "OOP polymorphism"]
    fused_hits = multi_query_search(
        retriever=mock_retriever,
        queries=queries,
        top_k=3,
        candidate_k=10,
        k_rrf=60,
    )

    # doc_b appeared in both queries, so it should rank highest in RRF score
    assert len(fused_hits) == 3
    assert fused_hits[0]["chunk_id"] == "doc_b"
    assert "rrf_score" in fused_hits[0]
    assert fused_hits[0]["source_ranks"] == {0: 2, 1: 1}


def test_multi_query_search_empty():
    assert multi_query_search(retriever=MagicMock(), queries=[]) == []


def test_hybrid_retriever_multi_query_search():
    mock_hybrid = HybridRetriever()
    mock_hybrid.search = MagicMock()
    mock_hybrid.search.side_effect = [
        [{"chunk_id": "h1", "text": "Hit 1"}],
        [{"chunk_id": "h2", "text": "Hit 2"}, {"chunk_id": "h1", "text": "Hit 1"}],
    ]

    results = mock_hybrid.multi_query_search(["query 1", "query 2"], top_k=2)
    assert len(results) == 2
    # h1 appeared in both lists, should have highest score
    assert results[0]["chunk_id"] == "h1"
