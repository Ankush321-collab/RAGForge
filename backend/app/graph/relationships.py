import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .entities import Entity


@dataclass
class Relationship:
    """A directed edge between two entities in the Knowledge Graph."""
    source: str
    target: str
    relation_type: str = "RELATED_TO"
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "properties": self.properties,
        }

    def __str__(self) -> str:
        return f"({self.source}) -[{self.relation_type}]-> ({self.target})"


class RelationshipExtractor:
    """Phase 15: Extract typed relationships between entities found in text."""

    RELATION_CLUES = [
        (r"\b(implements?|implementing)\b", "IMPLEMENTS"),
        (r"\b(uses?|using|utilizes?)\b", "USES"),
        (r"\b(enables?|facilitates?)\b", "ENABLES"),
        (r"\b(developed by|created by|authored by)\b", "DEVELOPED_BY"),
        (r"\b(inherits from|subclass of|derived from)\b", "SUBCLASS_OF"),
        (r"\b(evaluates? on|tested on)\b", "EVALUATES_ON"),
        (r"\b(includes?|consists of|contains?)\b", "INCLUDES"),
    ]

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
    ) -> List[Relationship]:
        """Extract relationships between entities co-occurring in sentences."""
        if len(entities) < 2:
            return []

        relationships: List[Relationship] = []
        sentences = re.split(r"[.\n;]", text)

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # Identify entities in this sentence
            present_entities = [
                e for e in entities if e.name.lower() in sent_str.lower()
            ]

            if len(present_entities) >= 2:
                for i in range(len(present_entities) - 1):
                    src = present_entities[i].name
                    tgt = present_entities[i + 1].name
                    if src.lower() == tgt.lower():
                        continue

                    # Determine relation type from sentence keywords
                    detected_type = "RELATED_TO"
                    for pattern, r_type in self.RELATION_CLUES:
                        if re.search(pattern, sent_str, flags=re.IGNORECASE):
                            detected_type = r_type
                            break

                    relationships.append(Relationship(
                        source=src,
                        target=tgt,
                        relation_type=detected_type,
                        properties={"context": sent_str[:120]},
                    ))

        return relationships
