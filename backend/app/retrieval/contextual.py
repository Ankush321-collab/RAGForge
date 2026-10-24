from typing import Any, Dict, List, Optional
import os


class ContextualRetriever:
    """Phase 12: Contextual Retrieval.
    
    Generates situational context for each chunk before embedding, while
    preserving the raw original text for precise ground-truth citations.
    """

    def __init__(
        self,
        document_title: str = "Technical Document",
        document_summary: Optional[str] = None,
    ) -> None:
        self.document_title = document_title
        self.document_summary = document_summary

    @staticmethod
    def build_context_prefix(
        document_title: str,
        page_number: int,
        section_heading: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> str:
        """Create a compact situational context header for a chunk."""
        parts = [f"Document: {document_title}", f"Page: {page_number}"]
        if section_heading:
            parts.append(f"Section: {section_heading}")
        if summary:
            # First sentence of summary
            summary_first = summary.strip().split(".")[0] + "."
            parts.append(f"Context: {summary_first}")
        return " | ".join(parts)

    def contextualize_chunk(
        self,
        chunk: Dict[str, Any],
        section_heading: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach contextualized text to a chunk while saving original text."""
        updated = chunk.copy()
        raw_text = updated.get("original_text") or updated.get("text", "")
        updated["original_text"] = raw_text

        page_num = updated.get("page_number", 1)
        prefix = self.build_context_prefix(
            document_title=updated.get("document_id") or self.document_title,
            page_number=page_num,
            section_heading=section_heading or updated.get("section"),
            summary=self.document_summary,
        )

        contextualized = f"[{prefix}]\n{raw_text}"
        updated["contextualized_text"] = contextualized
        # Set text to contextualized version for downstream embedding & vector indexing
        updated["text"] = contextualized
        updated["strategy"] = "contextual"
        return updated

    def contextualize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        section_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch contextualize a sequence of document chunks."""
        contextualized_list = []
        for idx, chunk in enumerate(chunks):
            heading = section_headings[idx] if section_headings and idx < len(section_headings) else None
            contextualized_list.append(self.contextualize_chunk(chunk, section_heading=heading))
        return contextualized_list
