from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .chunker import chunk_document
from .loader import load_document


@lru_cache(maxsize=1)
def get_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the embedding model once, when Phase 1C is actually used."""
    return SentenceTransformer(model_name)


def embed(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create one embedding for a chunk of text."""
    return get_model(model_name).encode(text)


def embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Attach embeddings to Phase 1B chunk records."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_model(model_name).encode(texts) if texts else []

    return [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]


def embed_document(
    pdf_path,
    chunk_size=500,
    overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """Run the connected loader -> chunker -> embedding pipeline."""
    document = load_document(pdf_path)
    chunks = chunk_document(
        document,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return embed_chunks(chunks, model_name=model_name)


if __name__ == "__main__":
    chunks = embed_document("data/raw/oops_notes.pdf")
    for chunk in chunks:
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print("Text:", chunk["text"])
        print("Embedding dimensions:", len(chunk["embedding"]))
        print("-" * 50)
