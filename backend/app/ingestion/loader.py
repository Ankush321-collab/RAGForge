from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path):
    """Extract one normalized page record per PDF page."""
    reader = PdfReader(pdf_path)
    return [
        {
            "page_number": page_number,
            "text": page.extract_text() or "",
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def get_metadata(pdf_path):
    """Return PDF metadata without performing other ingestion work."""
    reader = PdfReader(pdf_path)
    return dict(reader.metadata or {})


def load_document(pdf_path):
    """Load the Phase 1A document contract used by later stages."""
    path = Path(pdf_path)
    return {
        "document_id": path.stem,
        "source": str(path),
        "metadata": get_metadata(path),
        "pages": extract_text(path),
    }


if __name__ == "__main__":
    document = load_document("data/raw/oops_notes.pdf")
    for page in document["pages"]:
        print("Page", page["page_number"])
        print(page["text"])
        print("-" * 50)
    print("\nMETADATA")
    print(document["metadata"])
