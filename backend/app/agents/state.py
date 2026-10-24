from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """Phase 16A: LangGraph execution state for multimodal agentic RAG."""
    original_query: str
    chat_history: List[Dict[str, str]]
    rewritten_query: str
    query_type: str  # TEXT, IMAGE, TABLE, GRAPH, HYBRID
    retrieved_documents: List[Dict[str, Any]]
    retrieved_images: List[Dict[str, Any]]
    retrieved_tables: List[Dict[str, Any]]
    graph_context: str
    final_context: str
    answer: str
    citations: List[Dict[str, Any]]
    grounding_score: float
    is_valid: bool
    iterations: int
