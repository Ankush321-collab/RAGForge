from typing import Any, Dict, List, Optional


class ParentChildRetriever:
    """Manages parent chunk storage and maps retrieved child chunks back to parent contexts."""

    def __init__(self) -> None:
        self._docstore: Dict[str, Dict[str, Any]] = {}

    def add_parent_chunks(self, parent_chunks: List[Dict[str, Any]]) -> None:
        """Store parent chunks in docstore keyed by parent_id."""
        for chunk in parent_chunks:
            parent_id = chunk.get("parent_id")
            if parent_id:
                self._docstore[parent_id] = chunk

    def get_parent_chunk(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve parent chunk by ID."""
        return self._docstore.get(parent_id)

    def resolve_parents(self, child_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Maps child hits to their parent chunks, removing duplicates while retaining hit order.
        
        If a child hit doesn't specify parent_id or the parent is not in docstore,
        the child hit itself is retained.
        """
        resolved = []
        seen_parents = set()

        for hit in child_hits:
            parent_id = hit.get("parent_id")
            if parent_id and parent_id in self._docstore:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_doc = self._docstore[parent_id].copy()
                    parent_doc["score"] = hit.get("score")
                    parent_doc["matched_child_id"] = hit.get("child_id")
                    resolved.append(parent_doc)
            else:
                resolved.append(hit)

        return resolved

    def clear(self) -> None:
        """Clear docstore."""
        self._docstore.clear()
