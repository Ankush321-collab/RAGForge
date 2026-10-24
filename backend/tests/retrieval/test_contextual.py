from backend.app.retrieval.contextual import ContextualRetriever


def test_contextual_prefix_building():
    prefix = ContextualRetriever.build_context_prefix(
        document_title="ML_Systems",
        page_number=14,
        section_heading="Ablation Analysis",
        summary="This document evaluates retrieval performance across 5 model benchmarks.",
    )
    assert "Document: ML_Systems" in prefix
    assert "Page: 14" in prefix
    assert "Section: Ablation Analysis" in prefix
    assert "Context: This document evaluates retrieval performance across 5 model benchmarks." in prefix


def test_contextualize_chunk_preserves_original_text():
    raw_chunk = {
        "chunk_id": "c_42",
        "document_id": "paper_2025",
        "page_number": 8,
        "text": "The model achieved 94.2% accuracy on benchmark test sets.",
    }

    retriever = ContextualRetriever(document_summary="Evaluation of multimodal vision transformers.")
    contextual_chunk = retriever.contextualize_chunk(raw_chunk, section_heading="Results")

    assert contextual_chunk["original_text"] == "The model achieved 94.2% accuracy on benchmark test sets."
    assert "Section: Results" in contextual_chunk["contextualized_text"]
    assert "accuracy on benchmark test sets" in contextual_chunk["text"]
    assert contextual_chunk["strategy"] == "contextual"


def test_contextualize_chunks_batch():
    chunks = [
        {"chunk_id": "c1", "text": "Passage one.", "page_number": 1},
        {"chunk_id": "c2", "text": "Passage two.", "page_number": 2},
    ]

    retriever = ContextualRetriever(document_title="Doc1")
    res = retriever.contextualize_chunks(chunks, section_headings=["Intro", "Methods"])

    assert len(res) == 2
    assert "Section: Intro" in res[0]["text"]
    assert "Section: Methods" in res[1]["text"]
