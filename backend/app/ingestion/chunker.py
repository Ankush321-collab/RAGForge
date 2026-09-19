from .loader import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages, chunk_size=500, overlap=50, document_id=None):
    """Split Phase 1A page records into Phase 1B chunk records."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = []

    for page in pages:
        for text in splitter.split_text(page.get("text", "")):
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "document_id": document_id,
                    "page_number": page["page_number"],
                    "text": text,
                }
            )

    return chunks


def chunk_document(document, chunk_size=500, overlap=50):
    """Run Phase 1B against the document returned by ``load_document``."""
    return chunk_pages(
        document["pages"],
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=document.get("document_id"),
    )


chunk_page = chunk_pages


if __name__ == "__main__":
    pages = extract_text("data/raw/oops_notes.pdf")
    for chunk in chunk_pages(pages):
        print("Chunk ID:", chunk["chunk_id"])
        print("Page Number:", chunk["page_number"])
        print(chunk["text"])
        print("-" * 50)
