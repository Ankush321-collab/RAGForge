import pytest
from backend.app.retrieval.bm25 import BM25Retriever, matches_metadata_filter
from backend.app.retrieval.hybrid import HybridRetriever


def test_matches_metadata_filter_operators():
    doc = {
        "document_id": "doc_101",
        "page_number": 42,
        "year": 2025,
        "document_type": "research_paper",
        "author": "Alice",
    }

    # Equality
    assert matches_metadata_filter(doc, {"year": 2025})
    assert not matches_metadata_filter(doc, {"year": 2024})

    # Range operators
    assert matches_metadata_filter(doc, {"year": {"$gte": 2024}})
    assert matches_metadata_filter(doc, {"year": {"$lte": 2025}})
    assert not matches_metadata_filter(doc, {"year": {"$gt": 2025}})
    assert not matches_metadata_filter(doc, {"page_number": {"$lt": 40}})

    # Set inclusion
    assert matches_metadata_filter(doc, {"document_type": {"$in": ["notes", "research_paper"]}})
    assert not matches_metadata_filter(doc, {"document_type": {"$in": ["textbook", "slides"]}})


def test_bm25_metadata_filtering():
    corpus = [
        {"chunk_id": "c1", "text": "Object-Oriented Programming in C++ with polymorphism.", "year": 2024, "section": "OOP"},
        {"chunk_id": "c2", "text": "Functional Programming in Haskell and pure functions.", "year": 2025, "section": "FP"},
        {"chunk_id": "c3", "text": "Object-Oriented Design patterns in Python.", "year": 2022, "section": "OOP"},
    ]

    retriever = BM25Retriever(corpus)

    # Search with year filter
    results_2024 = retriever.search("Object-Oriented", filter={"year": {"$gte": 2023}})
    assert len(results_2024) == 1
    assert results_2024[0]["chunk_id"] == "c1"

    # Search with section filter
    results_fp = retriever.search("Programming", filter={"section": "FP"})
    assert len(results_fp) == 1
    assert results_fp[0]["chunk_id"] == "c2"


def test_hybrid_metadata_filtering():
    corpus = [
        {"chunk_id": "c1", "text": "Deep learning architectures and transformers.", "document_id": "dl_book", "page_number": 5},
        {"chunk_id": "c2", "text": "Deep learning vision systems using CNNs.", "document_id": "vision_paper", "page_number": 12},
    ]

    bm25 = BM25Retriever(corpus)
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_store=None)

    hits = hybrid.search("Deep learning", mode="bm25_only", filter={"document_id": "dl_book"})
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
