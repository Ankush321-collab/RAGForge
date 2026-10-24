from backend.app.graph.entities import Entity, EntityExtractor
from backend.app.graph.relationships import Relationship, RelationshipExtractor
from backend.app.graph.neo4j import KnowledgeGraphStore


def test_entity_extraction():
    extractor = EntityExtractor()
    text = "OpenAI released GPT-4 which implements Polymorphism and utilizes BM25 for sparse retrieval."
    entities = extractor.extract_entities(text)
    names = [e.name for e in entities]

    assert any("Openai" in n or "OPENAI" in n for n in names)
    assert any("Gpt-4" in n or "GPT-4" in n for n in names)
    assert any("Polymorphism" in n for n in names)
    assert any("BM25" in n for n in names)


def test_relationship_extraction():
    extractor = RelationshipExtractor()
    entities = [
        Entity(name="GPT-4", entity_type="Model"),
        Entity(name="OpenAI", entity_type="Organization"),
    ]
    text = "GPT-4 was developed by OpenAI for reasoning tasks."
    rels = extractor.extract_relationships(text, entities)

    assert len(rels) > 0
    assert rels[0].source == "GPT-4"
    assert rels[0].target == "OpenAI"
    assert rels[0].relation_type == "DEVELOPED_BY"


def test_graph_store_local_subgraph_and_context():
    store = KnowledgeGraphStore()

    passage = (
        "Polymorphism enables Virtual Functions. "
        "Virtual Functions use vtable for dynamic dispatch."
    )
    store.build_from_text(passage)

    # Query context
    context = store.get_graph_context_for_query("Explain Virtual Functions and dynamic dispatch")
    assert "Knowledge Graph Relationships:" in context
    assert "Virtual Functions" in context
