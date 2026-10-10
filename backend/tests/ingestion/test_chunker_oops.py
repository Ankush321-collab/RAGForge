from pathlib import Path
import pytest

from backend.app.ingestion.loader import load_document
from backend.app.ingestion.chunker import (
    chunk_fixed,
    chunk_recursive,
    chunk_structure_aware,
    chunk_semantic,
    chunk_parent_child,
)
from backend.app.retrieval.parent_child import ParentChildRetriever


PDF_PATH = Path("data/raw/oops_notes.pdf")


@pytest.mark.skipif(not PDF_PATH.exists(), reason="oops_notes.pdf not found in data/raw")
def test_oops_notes_chunking_all_strategies():
    doc = load_document(PDF_PATH)
    pages = doc["pages"]
    doc_id = doc["document_id"]

    assert len(pages) > 0, "PDF pages should not be empty"

    # 1. Fixed Chunking
    fixed_chunks = chunk_fixed(pages, chunk_size=500, overlap=50, document_id=doc_id)
    assert len(fixed_chunks) > 0
    assert fixed_chunks[0]["strategy"] == "fixed"

    # 2. Recursive Chunking
    recursive_chunks = chunk_recursive(pages, chunk_size=500, overlap=50, document_id=doc_id)
    assert len(recursive_chunks) > 0
    assert recursive_chunks[0]["strategy"] == "recursive"

    # 3. Structure-Aware Chunking
    structure_chunks = chunk_structure_aware(pages, max_chunk_size=800, document_id=doc_id)
    assert len(structure_chunks) > 0
    assert structure_chunks[0]["strategy"] == "structure_aware"

    # 4. Semantic Chunking
    semantic_chunks = chunk_semantic(pages, max_chunk_size=600, document_id=doc_id)
    assert len(semantic_chunks) > 0
    assert semantic_chunks[0]["strategy"] == "semantic"

    # 5. Parent-Child Chunking
    parents, children = chunk_parent_child(
        pages, parent_size=1500, child_size=300, child_overlap=50, document_id=doc_id
    )
    assert len(parents) > 0
    assert len(children) >= len(parents)

    # Test ParentChildRetriever with real oops_notes chunks
    retriever = ParentChildRetriever()
    retriever.add_parent_chunks(parents)

    # Mock a search hit on the first child
    first_child = children[0]
    child_hits = [
        {
            "child_id": first_child["child_id"],
            "parent_id": first_child["parent_id"],
            "score": 0.95,
            "text": first_child["text"],
        }
    ]

    resolved = retriever.resolve_parents(child_hits)
    assert len(resolved) == 1
    assert resolved[0]["parent_id"] == first_child["parent_id"]
    assert resolved[0]["score"] == 0.95
    assert len(resolved[0]["text"]) >= len(first_child["text"])
