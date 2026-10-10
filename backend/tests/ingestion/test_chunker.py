import pytest
from backend.app.ingestion.chunker import (
    chunk_fixed,
    chunk_recursive,
    chunk_structure_aware,
    chunk_semantic,
    chunk_parent_child,
)


@pytest.fixture
def sample_pages():
    return [
        {
            "page_number": 1,
            "text": (
                "# Section 1: Introduction\n\n"
                "This is a paragraph about vector RAG systems.\n\n"
                "| Header 1 | Header 2 |\n"
                "| --- | --- |\n"
                "| Value 1 | Value 2 |\n\n"
                "Conclusion sentence 1. Conclusion sentence 2."
            ),
        }
    ]


def test_chunk_fixed(sample_pages):
    chunks = chunk_fixed(sample_pages, chunk_size=50, overlap=10, document_id="doc1")
    assert len(chunks) > 0
    assert chunks[0]["strategy"] == "fixed"
    assert chunks[0]["document_id"] == "doc1"


def test_chunk_recursive(sample_pages):
    chunks = chunk_recursive(sample_pages, chunk_size=100, overlap=10, document_id="doc1")
    assert len(chunks) > 0
    assert chunks[0]["strategy"] == "recursive"


def test_chunk_structure_aware(sample_pages):
    chunks = chunk_structure_aware(sample_pages, max_chunk_size=150, document_id="doc1")
    assert len(chunks) > 0
    assert any("Header 1" in c["text"] for c in chunks)
    assert chunks[0]["strategy"] == "structure_aware"


def test_chunk_semantic(sample_pages):
    chunks = chunk_semantic(sample_pages, max_chunk_size=100, document_id="doc1")
    assert len(chunks) > 0
    assert chunks[0]["strategy"] == "semantic"


def test_chunk_parent_child(sample_pages):
    parents, children = chunk_parent_child(
        sample_pages, parent_size=200, child_size=50, child_overlap=10, document_id="doc1"
    )
    assert len(parents) > 0
    assert len(children) > len(parents)
    assert "parent_id" in children[0]
    assert children[0]["parent_id"] == parents[0]["parent_id"]
