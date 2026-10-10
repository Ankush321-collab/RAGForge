import os
import sys
import time
from pathlib import Path

# Ensure backend package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.storage.s3 import S3Storage
from backend.app.ingestion.loader import load_document
from backend.app.ingestion.chunker import chunk_parent_child
from backend.app.retrieval.opensearch import OpenSearchVectorStore
from backend.app.retrieval.parent_child import ParentChildRetriever


def run_end_to_end_test(max_pages: int | None = None):
    print("======================================================================")
    print("   RAGForge Phase 1 -> Phase 3 End-to-End Pipeline A to Z Test")
    print("======================================================================\n")

    pdf_path = Path("data/raw/oops_notes.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return False

    # -------------------------------------------------------------------------
    # STEP 1: S3 Upload & Storage Verification (Phase 1 / Phase 2A)
    # -------------------------------------------------------------------------
    print("[STEP 1/6] Uploading document to S3 bucket...")
    storage = S3Storage()
    storage.ensure_bucket()

    doc_id = "test_e2e_oops_full"
    object_key = storage.build_object_key(doc_id, pdf_path.name)
    
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    storage.upload_bytes(file_bytes, object_key, "application/pdf")
    print(f"  -> Uploaded '{pdf_path.name}' ({len(file_bytes)} bytes) to S3 at key: '{object_key}'")

    # Generate Presigned URL
    presigned_url = storage.generate_presigned_url(object_key, expires_in=300)
    print(f"  -> Generated Presigned URL: {presigned_url[:80]}...\n")

    # -------------------------------------------------------------------------
    # STEP 2: Document Ingestion & Page Extraction (Phase 1A)
    # -------------------------------------------------------------------------
    print("[STEP 2/6] Parsing ALL PDF document pages...")
    doc = load_document(pdf_path)
    pages = doc["pages"] if max_pages is None else doc["pages"][:max_pages]
    print(f"  -> Extracted ALL {len(pages)} pages from '{pdf_path.name}'\n")

    # -------------------------------------------------------------------------
    # STEP 3: Advanced Parent-Child Chunking (Phase 3 Chunking)
    # -------------------------------------------------------------------------
    print("[STEP 3/6] Generating Parent-Child Chunks across all pages...")
    parent_chunks, child_chunks = chunk_parent_child(
        pages, parent_size=1500, child_size=300, child_overlap=50, document_id=doc_id
    )
    print(f"  -> Generated {len(parent_chunks)} Parent Chunks & {len(child_chunks)} Child Chunks.\n")

    # Store Parent Chunks in Docstore
    retriever = ParentChildRetriever()
    retriever.add_parent_chunks(parent_chunks)

    # -------------------------------------------------------------------------
    # STEP 4: Embedding & AWS OpenSearch Serverless Indexing (Phase 3 Indexing)
    # -------------------------------------------------------------------------
    print("[STEP 4/6] Indexing ALL Child Chunks into AWS OpenSearch Serverless ('ragforge-chunks')...")
    opensearch_store = OpenSearchVectorStore()
    indexed_cnt = opensearch_store.index_chunks(child_chunks)
    print(f"  -> Successfully indexed ALL {indexed_cnt} child chunks into OpenSearch!\n")

    # Allow brief propagation window for Serverless indexing
    time.sleep(2)

    # -------------------------------------------------------------------------
    # STEP 5: Vector Similarity Search & BM25 Keyword Search
    # -------------------------------------------------------------------------
    print("[STEP 5/6] Querying AWS OpenSearch Serverless...")
    
    # 5A. Vector Search
    query = "What is object oriented programming?"
    print(f"  -> Running Vector Search for query: '{query}'")
    vector_hits = opensearch_store.vector_search(query, top_k=3)
    assert len(vector_hits) > 0, "Vector search returned no results!"
    print(f"     [Vector Top Hit] Score: {vector_hits[0]['score']:.4f} | Page: {vector_hits[0]['page_number']}")
    print(f"     Snippet: {repr(vector_hits[0]['text'][:100])}...")

    # 5B. BM25 Keyword Search
    keyword_query = "class inheritance"
    print(f"\n  -> Running BM25 Keyword Search for query: '{keyword_query}'")
    bm25_hits = opensearch_store.bm25_search(keyword_query, top_k=3)
    assert len(bm25_hits) > 0, "BM25 search returned no results!"
    print(f"     [BM25 Top Hit] Score: {bm25_hits[0]['score']:.4f} | Page: {bm25_hits[0]['page_number']}")
    print(f"     Snippet: {repr(bm25_hits[0]['text'][:100])}...\n")

    # -------------------------------------------------------------------------
    # STEP 6: Parent-Child Context Resolution
    # -------------------------------------------------------------------------
    print("[STEP 6/6] Resolving Child Search Hits to Full Parent Context...")
    resolved_parents = retriever.resolve_parents(vector_hits)
    assert len(resolved_parents) > 0, "Parent resolution failed!"

    top_parent = resolved_parents[0]
    print(f"  -> Successfully resolved Child Hit to Parent ID: '{top_parent.get('parent_id')}'")
    print(f"  -> Matched Child Length  : {len(vector_hits[0]['text'])} chars")
    print(f"  -> Resolved Parent Length : {len(top_parent['text'])} chars")
    print(f"  -> Parent Text Snippet    : {repr(top_parent['text'][:150])}...\n")

    print("======================================================================")
    print(f"   SUCCESS! ALL {len(pages)} PAGES PROCESSED AND INDEXED SUCCESSFULLY!")
    print("======================================================================\n")
    return True


if __name__ == "__main__":
    success = run_end_to_end_test()
    if not success:
        sys.exit(1)
