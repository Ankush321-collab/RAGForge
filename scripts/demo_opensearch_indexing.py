import os
import sys

# Ensure backend package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.ingestion.loader import load_document
from backend.app.ingestion.chunker import chunk_recursive
from backend.app.retrieval.opensearch import OpenSearchVectorStore


def main():
    print("==================================================")
    print("  AWS OpenSearch Vector & BM25 Search Demo")
    print("==================================================")

    store = OpenSearchVectorStore()

    # Load and chunk first 5 pages of oops_notes.pdf
    doc = load_document("data/raw/oops_notes.pdf")
    pages = doc["pages"][:5]
    doc_id = doc["document_id"]

    chunks = chunk_recursive(pages, chunk_size=400, overlap=50, document_id=doc_id)
    print(f"Loaded document '{doc_id}' ({len(pages)} pages). Generated {len(chunks)} chunks.")

    print("\nIndexing chunks into OpenSearch Serverless ('ragforge-chunks')...")
    indexed_cnt = store.index_chunks(chunks)
    print(f"Successfully indexed {indexed_cnt} chunks into AWS OpenSearch Serverless!")

    # Test Vector Search
    query = "What is object oriented programming?"
    print(f"\n--- Testing Vector Search (Query: '{query}') ---")
    vector_results = store.vector_search(query, top_k=2)
    for idx, hit in enumerate(vector_results, start=1):
        print(f"\n[Hit {idx}] Score: {hit['score']:.4f} | Page: {hit['page_number']}")
        print(f"Snippet: {repr(hit['text'][:120])}...")

    # Test BM25 Keyword Search
    keyword = "class inheritance"
    print(f"\n--- Testing BM25 Keyword Search (Query: '{keyword}') ---")
    bm25_results = store.bm25_search(keyword, top_k=2)
    for idx, hit in enumerate(bm25_results, start=1):
        print(f"\n[Hit {idx}] Score: {hit['score']:.4f} | Page: {hit['page_number']}")
        print(f"Snippet: {repr(hit['text'][:120])}...")


if __name__ == "__main__":
    main()
