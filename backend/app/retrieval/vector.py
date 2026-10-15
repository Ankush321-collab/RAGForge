from collections.abc import Iterable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..ingestion.embedding import embed_document, get_model


class SentenceTransformerEmbeddings(Embeddings):
    """Adapt the Phase 1C SentenceTransformer model to LangChain."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = get_model(self.model_name).encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return get_model(self.model_name).encode(text).tolist()


def chunks_to_documents(chunks: Iterable[dict]) -> list[Document]:
    """Convert Phase 1B/1C/10 chunks into LangChain documents."""
    docs = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id") or chunk.get("child_id") or chunk.get("id") or str(idx)
        metadata = {
            "chunk_id": chunk_id,
            "document_id": chunk.get("document_id"),
            "page_number": chunk.get("page_number", 1),
        }
        if "parent_id" in chunk:
            metadata["parent_id"] = chunk["parent_id"]
        if "strategy" in chunk:
            metadata["strategy"] = chunk["strategy"]
        if "original_text" in chunk:
            metadata["original_text"] = chunk["original_text"]
        if "context" in chunk:
            metadata["context"] = chunk["context"]
        docs.append(Document(page_content=chunk["text"], metadata=metadata))
    return docs


def create_vector_store(
    chunks: Iterable[dict],
    collection_name="ragforge_chunks",
    persist_directory="data/chroma",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """Create persistent ChromaDB from chunks, computing embeddings if not precomputed."""
    chunks = list(chunks)
    documents = chunks_to_documents(chunks)
    
    if chunks and "embedding" in chunks[0]:
        embeddings = [
            chunk["embedding"].tolist() if hasattr(chunk["embedding"], "tolist") else chunk["embedding"]
            for chunk in chunks
        ]
    elif documents:
        emb_fn = SentenceTransformerEmbeddings(model_name)
        embeddings = emb_fn.embed_documents([doc.page_content for doc in documents])
    else:
        embeddings = []

    ids = [
        f"{chunk.get('document_id', 'document')}-{chunk.get('chunk_id') or chunk.get('child_id') or idx}"
        for idx, chunk in enumerate(chunks, start=1)
    ]

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=SentenceTransformerEmbeddings(model_name),
        persist_directory=str(Path(persist_directory)) if persist_directory else None,
    )
    if documents and embeddings:
        vector_store._collection.upsert(
            ids=ids,
            documents=[document.page_content for document in documents],
            metadatas=[document.metadata for document in documents],
            embeddings=embeddings,
        )
    return vector_store


def create_vector_store_from_pdf(
    pdf_path,
    collection_name="ragforge_chunks",
    persist_directory="data/chroma",
    chunk_size=500,
    overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """Run loader -> chunker -> embedding -> ChromaDB for one PDF."""
    chunks = embed_document(
        pdf_path,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
    )
    return create_vector_store(
        chunks,
        collection_name=collection_name,
        persist_directory=persist_directory,
        model_name=model_name,
    )


def similarity_search(vector_store, query: str, k=5, **kwargs) -> list[Document]:
    """Return the top-k documents from ChromaDB for a query with optional filtering."""
    return vector_store.similarity_search(query, k=k, **kwargs)
