import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend.app.retrieval.hyde import (
    HyDEGenerator,
    HyDERetriever,
    hybrid_hyde_search,
)


def test_hyde_generator():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Polymorphism in object-oriented programming allows objects of different classes to respond to the same method call."
    )

    generator = HyDEGenerator(llm=mock_llm)
    result = generator.generate_hypothetical_document("What is polymorphism?")

    assert result["original_query"] == "What is polymorphism?"
    assert "Polymorphism in object-oriented programming" in result["hypothetical_document"]
    mock_llm.invoke.assert_called_once()


def test_hyde_retriever_search():
    mock_generator = MagicMock()
    mock_generator.generate_hypothetical_document.return_value = {
        "original_query": "Explain inheritance",
        "hypothetical_document": "Inheritance allows a derived class to inherit fields and methods from a base class.",
    }

    mock_vector_store = MagicMock()
    mock_vector_store.vector_search.return_value = [
        {"chunk_id": "c1", "text": "Base class and derived class inheritance principles.", "score": 0.88},
        {"chunk_id": "c2", "text": "Subclass syntax in C++.", "score": 0.74},
    ]

    retriever = HyDERetriever(vector_store=mock_vector_store, generator=mock_generator)
    hits = retriever.search("Explain inheritance", top_k=2)

    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["retrieval_strategy"] == "hyde"
    assert hits[0]["original_query"] == "Explain inheritance"
    assert "Inheritance allows a derived class" in hits[0]["hypothetical_document"]
    mock_vector_store.vector_search.assert_called_once_with(
        "Inheritance allows a derived class to inherit fields and methods from a base class.",
        top_k=2,
    )


def test_hyde_retriever_as_documents():
    mock_generator = MagicMock()
    mock_generator.generate_hypothetical_document.return_value = {
        "original_query": "Explain abstraction",
        "hypothetical_document": "Abstraction is hiding internal details.",
    }

    mock_vector_store = MagicMock()
    mock_vector_store.vector_search.return_value = [
        {"chunk_id": "c_abs", "text": "Abstraction in OOPS hides implementation.", "page_number": 5}
    ]

    retriever = HyDERetriever(vector_store=mock_vector_store, generator=mock_generator)
    docs = retriever.search_as_documents("Explain abstraction", top_k=1)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Abstraction in OOPS hides implementation."
    assert docs[0].metadata["chunk_id"] == "c_abs"
    assert docs[0].metadata["retrieval_strategy"] == "hyde"


def test_hybrid_hyde_search_pipeline():
    mock_hybrid = MagicMock()
    mock_hybrid.bm25_retriever.search.return_value = [
        {"chunk_id": "c_bm25", "text": "BM25 keyword match for polymorphism"}
    ]

    mock_hyde = MagicMock()
    mock_hyde.search.return_value = [
        {"chunk_id": "c_hyde", "text": "HyDE semantic vector hit for polymorphism"}
    ]

    fused = hybrid_hyde_search(
        hybrid_retriever=mock_hybrid,
        hyde_retriever=mock_hyde,
        query="What is polymorphism?",
        top_k=2,
        candidate_k=5,
    )

    assert len(fused) == 2
    assert fused[0]["retrieval_strategy"] == "hybrid_hyde"
    assert "rrf_score" in fused[0]


def test_hyde_empty_query():
    retriever = HyDERetriever()
    assert retriever.search("") == []
    assert retriever.search_as_documents("") == []
