import pytest
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
    assert resolved_parents[0]["score"] == 0.92
    assert resolved_parents[1]["parent_id"] == "p_2"
    assert resolved_parents[1]["score"] == 0.78
