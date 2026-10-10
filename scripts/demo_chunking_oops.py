from pathlib import Path
from backend.app.ingestion.loader import load_document
from backend.app.ingestion.chunker import (
    chunk_fixed,
    chunk_recursive,
    chunk_structure_aware,
    chunk_semantic,
    chunk_parent_child,
)
from backend.app.retrieval.parent_child import ParentChildRetriever


def main():
    pdf_path = Path("data/raw/oops_notes.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    print("==================================================")
    print("  RAGForge Phase 3 Chunking Benchmarking Demo")
    print(f"  Target File: {pdf_path}")
    print("==================================================\n")

    doc = load_document(pdf_path)
    pages = doc["pages"]
    doc_id = doc["document_id"]

    print(f"Loaded PDF Document ID: '{doc_id}'")
    print(f"Total Pages Extracted: {len(pages)}\n")

    # 1. Fixed Chunking
    fixed = chunk_fixed(pages, chunk_size=500, overlap=50, document_id=doc_id)
    avg_fixed = sum(len(c["text"]) for c in fixed) / len(fixed) if fixed else 0

    # 2. Recursive Chunking
    recursive = chunk_recursive(pages, chunk_size=500, overlap=50, document_id=doc_id)
    avg_rec = sum(len(c["text"]) for c in recursive) / len(recursive) if recursive else 0

    # 3. Structure-Aware Chunking
    structured = chunk_structure_aware(pages, max_chunk_size=800, document_id=doc_id)
    avg_struct = sum(len(c["text"]) for c in structured) / len(structured) if structured else 0

    # 4. Semantic Chunking
    semantic = chunk_semantic(pages, max_chunk_size=600, document_id=doc_id)
    avg_sem = sum(len(c["text"]) for c in semantic) / len(semantic) if semantic else 0

    # 5. Parent-Child Chunking
    parents, children = chunk_parent_child(
        pages, parent_size=1500, child_size=300, child_overlap=50, document_id=doc_id
    )
    avg_parent = sum(len(p["text"]) for p in parents) / len(parents) if parents else 0
    avg_child = sum(len(c["text"]) for c in children) / len(children) if children else 0

    print("-----------------------------------------------------------------------------------------")
    print(f"{'Strategy':<25} | {'Total Chunks':<15} | {'Avg Chunk Length (chars)':<25}")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'1. Fixed Chunking':<25} | {len(fixed):<15} | {avg_fixed:<25.1f}")
    print(f"{'2. Recursive Chunking':<25} | {len(recursive):<15} | {avg_rec:<25.1f}")
    print(f"{'3. Structure-Aware':<25} | {len(structured):<15} | {avg_struct:<25.1f}")
    print(f"{'4. Semantic Chunking':<25} | {len(semantic):<15} | {avg_sem:<25.1f}")
    print(f"{'5. Parent Chunks':<25} | {len(parents):<15} | {avg_parent:<25.1f}")
    print(f"{'   Child Chunks':<25} | {len(children):<15} | {avg_child:<25.1f}")
    print("-----------------------------------------------------------------------------------------\n")

    # Demo Parent-Child Retrieval
    retriever = ParentChildRetriever()
    retriever.add_parent_chunks(parents)

    sample_child = children[0]
    hits = [{"child_id": sample_child["child_id"], "parent_id": sample_child["parent_id"], "score": 0.94, "text": sample_child["text"]}]
    resolved = retriever.resolve_parents(hits)

    print("--- Parent-Child Retrieval Demonstration ---")
    print(f"Matched Child Chunk ID : {sample_child['child_id']} (Len: {len(sample_child['text'])})")
    print(f"Child Snippet Preview  : {repr(sample_child['text'][:80])}...")
    print(f"Resolved Parent ID     : {resolved[0]['parent_id']} (Len: {len(resolved[0]['text'])})")
    print(f"Parent Context Preview : {repr(resolved[0]['text'][:150])}...")
    print("==================================================\n")


if __name__ == "__main__":
    main()
