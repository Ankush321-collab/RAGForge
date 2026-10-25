import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Entity:
    """A semantic entity in the Knowledge Graph."""
    name: str
    entity_type: str = "Concept"
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
        }


class EntityExtractor:
    """Phase 15: Rule-based and pattern entity extraction for Graph RAG."""

    COMMON_PATTERNS = [
        (r"\b(GPT-4|GPT-3\.5|Claude|Mistral|Llama|BERT|ResNet|EfficientNet)\b", "Model"),
        (r"\b(OpenAI|Anthropic|Meta|Google|Microsoft|Amazon)\b", "Organization"),
        (r"\b(Polymorphism|Inheritance|Encapsulation|Abstraction|Virtual Functions?|vtable|vptr|Dynamic Dispatch)\b", "Concept"),
        (r"\b(Process|Thread|Semaphore|Mutex|Deadlock|Virtual Memory|Paging|Page Fault|CPU Scheduling|Context Switch|PCB|Kernel|User Mode|System Call|Demand Paging|Thrashing|Banker's Algorithm)\b", "OS_Concept"),
        (r"\b(BM25|Dense Retrieval|Sparse Retrieval|Reranker|Cross-Encoder|HyDE|RRF)\b", "Technique"),
        (r"\b(ChromaDB|OpenSearch|Elasticsearch|Neo4j|S3|PostgreSQL)\b", "Database"),
        (r"\b(Recall@[0-9]+|MRR|NDCG|Precision@[0-9]+|F1[-_ ]Score|Accuracy)\b", "Metric"),
    ]

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract recognized semantic entities from text."""
        entities: Dict[str, Entity] = {}

        for pattern, entity_type in self.COMMON_PATTERNS:
            matches = re.finditer(pattern, text, flags=re.IGNORECASE)
            for m in matches:
                matched_name = m.group(0).strip()
                canonical_name = matched_name.title() if len(matched_name) > 4 else matched_name.upper()
                if canonical_name not in entities:
                    entities[canonical_name] = Entity(
                        name=canonical_name,
                        entity_type=entity_type,
                        description=f"{entity_type} mentioned in passage",
                    )

        # Capitalized technical terms heuristics (e.g., "Dynamic Dispatch")
        cap_terms = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
        for term in cap_terms:
            if term not in entities and len(term) > 5:
                entities[term] = Entity(name=term, entity_type="Concept")

        return list(entities.values())
