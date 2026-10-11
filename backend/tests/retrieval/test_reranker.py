import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from backend.app.retrieval.reranker import CrossEncoderReranker, hybrid_search_and_rerank


def test_reranker_with_mock_model():
    mock_model = MagicMock()
    # Return scores for 3 candidates: doc 1 gets 0.1, doc 2 gets 0.95, doc 3 gets 0.4
    mock_model.predict.return_value = [0.10, 0.95, 0.40]

    reranker = CrossEncoderReranker(model=mock_model)

    candidates = [
        {"chunk_id": "c1", "text": "Low relevance document text"},
        {"chunk_id": "c2", "text": "High relevance exact match document text"},
        {"chunk_id": "c3", "text": "Medium relevance document text"},
    ]

    reranked = reranker.rerank(query="exact match", documents=candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["rerank_score"] == 0.95
    assert reranked[0]["rerank_rank"] == 1

    assert reranked[1]["chunk_id"] == "c3"
    assert reranked[1]["rerank_score"] == 0.40
    assert reranked[1]["rerank_rank"] == 2


def test_reranker_as_documents():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.85]

    reranker = CrossEncoderReranker(model=mock_model)
    docs = [Document(page_content="Rerank test doc", metadata={"id": "doc1"})]

    results = reranker.rerank_as_documents("query", docs, top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].page_content == "Rerank test doc"
    assert results[0].metadata["id"] == "doc1"
    assert results[0].metadata["rerank_score"] == 0.85


def test_hybrid_search_and_rerank_pipeline():
    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": "cand_1", "text": "Candidate 1 from hybrid"},
        {"chunk_id": "cand_2", "text": "Candidate 2 from hybrid"},
    ]

    mock_model = MagicMock()
    # cand_2 is more relevant than cand_1
    mock_model.predict.return_value = [0.2, 0.9]

    reranker = CrossEncoderReranker(model=mock_model)

    final_results = hybrid_search_and_rerank(
        hybrid_retriever=mock_hybrid,
        reranker=reranker,
        query="test query",
        top_k=1,
        candidate_k=2,
    )

    mock_hybrid.search.assert_called_once_with(query="test query", top_k=2, candidate_k=2)
    assert len(final_results) == 1
    assert final_results[0]["chunk_id"] == "cand_2"
    assert final_results[0]["rerank_score"] == 0.9


def test_reranker_empty_input():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", []) == []
    assert reranker.rerank_as_documents("query", []) == []
