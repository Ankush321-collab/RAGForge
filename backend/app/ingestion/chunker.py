import re
from typing import Any, Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader import extract_text


def chunk_fixed(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 50,
    document_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Fixed-size character chunking with exact stride/overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    stride = chunk_size - overlap
    chunks = []

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_number", 1)
        if not text:
            continue

        start = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size]
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "document_id": document_id,
                    "page_number": page_num,
                    "text": chunk_text,
                    "strategy": "fixed",
                }
            )
            start += stride

    return chunks


def chunk_recursive(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 50,
    document_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Recursive character chunking using hierarchical separators (\\n\\n, \\n, space)."""
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
                    "page_number": page.get("page_number", 1),
                    "text": text,
                    "strategy": "recursive",
                }
            )

    return chunks


# Alias existing chunk_pages to chunk_recursive for backwards compatibility
chunk_pages = chunk_recursive


def chunk_structure_aware(
    pages: List[Dict[str, Any]],
    max_chunk_size: int = 800,
    document_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Structure-aware chunking preserving headings, table blocks, and code blocks."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    chunks = []

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_number", 1)
        if not text.strip():
            continue

        # Split text into logical blocks (paragraphs, markdown tables, headers, code blocks)
        raw_blocks = re.split(r"(\n\s*\n|(?=^#{1,6}\s)|(?=^\|.*\|$))", text, flags=re.MULTILINE)
        blocks = [b.strip() for b in raw_blocks if b and b.strip()]

        current_chunk_text = ""

        for block in blocks:
            # If adding this block exceeds max_chunk_size and current_chunk_text is not empty, flush current chunk
            if len(current_chunk_text) + len(block) + 1 > max_chunk_size and current_chunk_text:
                chunks.append(
                    {
                        "chunk_id": len(chunks) + 1,
                        "document_id": document_id,
                        "page_number": page_num,
                        "text": current_chunk_text.strip(),
                        "strategy": "structure_aware",
                    }
                )
                current_chunk_text = ""

            # If a single block exceeds max_chunk_size, append it as its own standalone block or split recursively
            if len(block) > max_chunk_size:
                if current_chunk_text:
                    chunks.append(
                        {
                            "chunk_id": len(chunks) + 1,
                            "document_id": document_id,
                            "page_number": page_num,
                            "text": current_chunk_text.strip(),
                            "strategy": "structure_aware",
                        }
                    )
                    current_chunk_text = ""

                # Sub-split oversized structural block
                sub_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=50)
                for sub_text in sub_splitter.split_text(block):
                    chunks.append(
                        {
                            "chunk_id": len(chunks) + 1,
                            "document_id": document_id,
                            "page_number": page_num,
                            "text": sub_text.strip(),
                            "strategy": "structure_aware",
                        }
                    )
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + block
                else:
                    current_chunk_text = block

        if current_chunk_text.strip():
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "document_id": document_id,
                    "page_number": page_num,
                    "text": current_chunk_text.strip(),
                    "strategy": "structure_aware",
                }
            )

    return chunks


def chunk_semantic(
    pages: List[Dict[str, Any]],
    max_chunk_size: int = 600,
    document_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Semantic sentence-boundary chunking grouping sentences up to max_chunk_size."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    chunks = []

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_number", 1)
        if not text.strip():
            continue

        # Split into sentences using sentence boundary regex
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) + 1 > max_chunk_size and current_sentences:
                chunks.append(
                    {
                        "chunk_id": len(chunks) + 1,
                        "document_id": document_id,
                        "page_number": page_num,
                        "text": " ".join(current_sentences),
                        "strategy": "semantic",
                    }
                )
                current_sentences = []
                current_len = 0

            current_sentences.append(sentence)
            current_len += len(sentence) + 1

        if current_sentences:
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "document_id": document_id,
                    "page_number": page_num,
                    "text": " ".join(current_sentences),
                    "strategy": "semantic",
                }
            )

    return chunks


def chunk_parent_child(
    pages: List[Dict[str, Any]],
    parent_size: int = 1500,
    child_size: int = 300,
    child_overlap: int = 50,
    document_id: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates large Parent Chunks (for LLM context) and small Child Chunks (for embedding indexing).
    
    Returns:
        (parent_chunks, child_chunks) tuple.
    """
    if parent_size <= child_size:
        raise ValueError("parent_size must be strictly greater than child_size")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap)

    parent_chunks = []
    child_chunks = []

    parent_idx = 1
    child_idx = 1

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_number", 1)
        if not text.strip():
            continue

        for parent_text in parent_splitter.split_text(text):
            p_id = f"parent_{parent_idx}"
            parent_chunk = {
                "parent_id": p_id,
                "document_id": document_id,
                "page_number": page_num,
                "text": parent_text,
                "strategy": "parent_child",
            }
            parent_chunks.append(parent_chunk)

            # Split parent chunk into smaller child chunks
            for child_text in child_splitter.split_text(parent_text):
                c_id = f"child_{child_idx}"
                child_chunk = {
                    "child_id": c_id,
                    "parent_id": p_id,
                    "document_id": document_id,
                    "page_number": page_num,
                    "text": child_text,
                    "strategy": "parent_child",
                }
                child_chunks.append(child_chunk)
                child_idx += 1

            parent_idx += 1

    return parent_chunks, child_chunks


def chunk_document(document: Dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Run Phase 1B/3 against the document returned by ``load_document``."""
    return chunk_pages(
        document["pages"],
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=document.get("document_id"),
    )


chunk_page = chunk_pages
