import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .entities import Entity, EntityExtractor
from .relationships import Relationship, RelationshipExtractor

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """Phase 15 & 15A: Knowledge Graph store supporting in-memory local graph & Neo4j.
    
    Provides offline/local graph construction, neighbor exploration, and
    graph context retrieval for Graph-augmented RAG.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        from backend.app.core.config import get_settings
        settings = get_settings()
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.driver = None

        # Local in-memory graph storage
        self.nodes: Dict[str, Entity] = {}
        self.edges: List[Relationship] = []
        self.adjacency: Dict[str, List[Tuple[Relationship, str]]] = {}

        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()

        if self.neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri, auth=(self.username or "", self.password or "")
                )
            except Exception as e:
                logger.warning("Neo4j driver initialization skipped (%s); running local in-memory graph.", e)

    def add_entity(self, entity: Entity) -> None:
        """Register a node in the graph."""
        self.nodes[entity.name.lower()] = entity
        if entity.name.lower() not in self.adjacency:
            self.adjacency[entity.name.lower()] = []

    def add_relationship(self, rel: Relationship) -> None:
        """Register a directed edge in the graph."""
        self.edges.append(rel)
        src_key = rel.source.lower()
        tgt_key = rel.target.lower()

        if src_key not in self.adjacency:
            self.adjacency[src_key] = []
        if tgt_key not in self.adjacency:
            self.adjacency[tgt_key] = []

        self.adjacency[src_key].append((rel, tgt_key))
        self.adjacency[tgt_key].append((rel, src_key))

    def build_from_text(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract and index entities and relationships from text."""
        entities = self.entity_extractor.extract_entities(text)
        for e in entities:
            self.add_entity(e)

        rels = self.relationship_extractor.extract_relationships(text, entities)
        for r in rels:
            self.add_relationship(r)

        return entities, rels

    def query_subgraph(self, entity_name: str, max_depth: int = 1) -> List[Relationship]:
        """Find 1-or-2 hop relationships connected to the given entity."""
        key = entity_name.lower()
        matched_edges: List[Relationship] = []
        visited_nodes: Set[str] = {key}

        current_frontier = [key]
        for _ in range(max_depth):
            next_frontier = []
            for node in current_frontier:
                for rel, neighbor in self.adjacency.get(node, []):
                    if rel not in matched_edges:
                        matched_edges.append(rel)
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_frontier.append(neighbor)
            current_frontier = next_frontier

        return matched_edges

    def get_graph_context_for_query(self, query: str, max_depth: int = 1) -> str:
        """Phase 15A: Extract relevant entity relationships to enrich prompt context."""
        query_lower = query.lower()
        matched_keys = [k for k in self.nodes.keys() if k in query_lower]
        query_entities = [self.nodes[k] for k in matched_keys]

        if not query_entities:
            query_entities = self.entity_extractor.extract_entities(query)

        all_rels: List[Relationship] = []
        for e in query_entities:
            rels = self.query_subgraph(e.name, max_depth=max_depth)
            for r in rels:
                if r not in all_rels:
                    all_rels.append(r)

        if not all_rels:
            return ""

        context_lines = [
            f"- {rel.source} {rel.relation_type.lower().replace('_', ' ')} {rel.target}"
            for rel in all_rels[:10]
        ]
        return "Knowledge Graph Relationships:\n" + "\n".join(context_lines)
