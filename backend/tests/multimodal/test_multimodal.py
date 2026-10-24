from pathlib import Path
import pytest

from backend.app.multimodal.image_extractor import PDFImageExtractor
from backend.app.multimodal.vision import VisionModel, MultimodalImageRetriever
from backend.app.multimodal.table_processor import TableProcessor


def test_table_markdown_generation():
    headers = ["Model", "Accuracy", "F1"]
    rows = [
        ["Model A", "91.2", "90.1"],
        ["Model B", "94.5", "93.7"],
    ]
    markdown = TableProcessor.table_to_markdown(headers, rows)
    assert "| Model | Accuracy | F1 |" in markdown
    assert "| Model A | 91.2 | 90.1 |" in markdown
    assert "| Model B | 94.5 | 93.7 |" in markdown


def test_table_search():
    tables = [
        {
            "table_id": "tbl_1",
            "headers": ["Model", "Accuracy", "F1_Score"],
            "markdown": "| Model | Accuracy | F1_Score |\n| Model B | 94.5 | 93.7 |",
        },
        {
            "table_id": "tbl_2",
            "headers": ["Algorithm", "Latency_ms"],
            "markdown": "| Algorithm | Latency_ms |\n| QuickSort | 12.4 |",
        },
    ]

    processor = TableProcessor()
    hits = processor.search_tables("Which model has highest F1_Score?", tables, top_k=1)
    assert len(hits) == 1
    assert hits[0]["table_id"] == "tbl_1"


def test_vision_model_and_image_retrieval():
    images = [
        {
            "image_id": "fig_retrieval",
            "document_id": "rag_doc",
            "page_number": 3,
            "width": 640,
            "height": 480,
            "s3_key": "documents/rag_doc/images/fig_retrieval.png",
        },
        {
            "image_id": "fig_eval",
            "document_id": "rag_doc",
            "page_number": 8,
            "width": 800,
            "height": 600,
            "s3_key": "documents/rag_doc/images/fig_eval.png",
        },
    ]

    hints = {
        "fig_retrieval": "hybrid retrieval pipeline architecture diagram showing vector and BM25 fusion",
        "fig_eval": "evaluation bar chart comparing MRR and Recall benchmarks",
    }

    retriever = MultimodalImageRetriever()
    indexed_cnt = retriever.index_images(images, context_hints=hints)
    assert indexed_cnt == 2

    # Search for architecture diagram
    results = retriever.search_images("retrieval pipeline architecture diagram", top_k=1)
    assert len(results) == 1
    assert results[0]["image_id"] == "fig_retrieval"
    assert "url" in results[0]
