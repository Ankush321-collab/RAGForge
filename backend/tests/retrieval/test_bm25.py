import pytest
from langchain_core.documents import Document

from backend.app.retrieval.bm25 import BM25Retriever, default_tokenizer


def test_default_tokenizer():
    tokens = default_tokenizer("GPT-4 has 100ms latency and 4096 context_size!")
    assert "gpt-4" in tokens
    assert "100ms" in tokens
    assert "4096" in tokens
    assert "context_size" in tokens


def test_bm25_empty_corpus():
    retriever = BM25Retriever()
    results = retriever.search("test query")
    assert results == []
    docs = retriever.search_as_documents("test query")
    assert docs == []


def test_bm25_search_chunks():
    corpus = [
        {
            "chunk_id": "c_1",
            "text": "Antigravity agents implement dense vector retrieval using ChromaDB.",
            "page_number": 1,
        },
        {
            "chunk_id": "c_2",
            "text": "BM25 is a probabilistic sparse keyword ranking function based on TF-IDF.",
            "page_number": 2,
        },
        {
            "chunk_id": "c_3",
            "text": "Cross-encoders perform reranking on top candidates from hybrid search.",
            "page_number": 3,
        },
    ]

    retriever = BM25Retriever(corpus)

    # Keyword match for BM25
    hits = retriever.search("TF-IDF BM25 ranking", top_k=2)
    assert len(hits) > 0
    assert hits[0]["chunk_id"] == "c_2"
    assert hits[0]["score"] > 0
    assert hits[0]["page_number"] == 2

    # Query with no match
    no_hits = retriever.search("Quantum entanglement teleportation")
    assert len(no_hits) == 0


def test_bm25_langchain_documents():
    docs = [
        Document(
            page_content="Model GPT-4 was evaluated on technical benchmarks with 92% accuracy.",
            metadata={"doc_id": "doc_gpt4", "source": "paper_v1.pdf"},
        ),
        Document(
            page_content="ResNet-50 is a convolutional neural network for image recognition.",
            metadata={"doc_id": "doc_resnet", "source": "vision.pdf"},
        ),
    ]

    retriever = BM25Retriever(docs)
    results = retriever.search_as_documents("GPT-4 benchmarks", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["doc_id"] == "doc_gpt4"
    assert results[0].metadata["source"] == "paper_v1.pdf"
    assert "score" in results[0].metadata
    assert results[0].metadata["score"] > 0
    assert "GPT-4" in results[0].page_content


def test_bm25_score_ordering():
    corpus = [
        {"chunk_id": "1", "text": "database database database open search database"},
        {"chunk_id": "2", "text": "database system"},
        {"chunk_id": "3", "text": "completely unrelated topic on biology"},
    ]
    retriever = BM25Retriever(corpus)
    hits = retriever.search("database", top_k=3)

    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "1"
    assert hits[1]["chunk_id"] == "2"
    assert hits[0]["score"] > hits[1]["score"]


def test_bm25_reindex_and_clear():
    retriever = BM25Retriever()
    corpus1 = [{"chunk_id": "1", "text": "alpha beta gamma"}]
    retriever.index(corpus1)
    assert len(retriever.search("alpha")) == 1

    retriever.clear()
    assert len(retriever.search("alpha")) == 0
    assert retriever.documents == []
    assert retriever.bm25 is None
