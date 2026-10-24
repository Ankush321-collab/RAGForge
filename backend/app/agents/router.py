import re
from typing import Any, Dict, List, Optional


class AgenticQueryRouter:
    """Phase 16: Agentic Query Router.
    
    Classifies queries into TEXT, IMAGE, TABLE, GRAPH, or HYBRID routes
    to dispatch the optimal retrieval strategy.
    """

    IMAGE_KEYWORDS = [
        r"\bfigure\b", r"\bdiagram\b", r"\bchart\b", r"\bimage\b",
        r"\barchitecture diagram\b", r"\bflowchart\b", r"\bschematic\b",
        r"\bplot\b", r"\bvisual\b", r"\billustration\b",
    ]

    TABLE_KEYWORDS = [
        r"\btable\b", r"\bbenchmark\b", r"\baccuracy\b", r"\bmetrics\b",
        r"\bf1[-_ ]score\b", r"\blatency\b", r"\bpercentage\b", r"\bscore\b",
        r"\bhighest\b", r"\blowest\b", r"\bcomparison\b", r"\brows?\b", r"\bcolumns?\b",
    ]

    GRAPH_KEYWORDS = [
        r"\bhow are .* related\b", r"\brelationship between\b", r"\bconnects? to\b",
        r"\bhierarchy\b", r"\bontology\b", r"\bentit(y|ies)\b", r"\bdepends? on\b",
        r"\bdeveloped by\b", r"\bsubclass of\b", r"\bderived from\b",
    ]

    def classify_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Classify a query into its primary retrieval route."""
        q_lower = query.lower()

        # 1. Image Route Check
        is_image = any(re.search(pat, q_lower) for pat in self.IMAGE_KEYWORDS)

        # 2. Table Route Check
        is_table = any(re.search(pat, q_lower) for pat in self.TABLE_KEYWORDS)

        # 3. Graph Route Check
        is_graph = any(re.search(pat, q_lower) for pat in self.GRAPH_KEYWORDS)

        # Hybrid if multiple modalities detected
        active_modalities = sum([is_image, is_table, is_graph])

        if active_modalities > 1:
            route = "HYBRID"
            reason = "Query references multiple modalities or cross-modal relationships."
        elif is_image:
            route = "IMAGE"
            reason = "Query explicitly references visual diagrams, figures, or charts."
        elif is_table:
            route = "TABLE"
            reason = "Query seeks structured tabular, numerical, or benchmark comparisons."
        elif is_graph:
            route = "GRAPH"
            reason = "Query asks about relationships, hierarchies, or connections between entities."
        else:
            route = "TEXT"
            reason = "Standard textual semantic and keyword retrieval question."

        return {
            "query": query,
            "route": route,
            "reason": reason,
            "recommended_strategy": self._recommended_strategy(route),
        }

    @staticmethod
    def _recommended_strategy(route: str) -> str:
        strategies = {
            "TEXT": "Hybrid (BM25 + Vector) with Parent-Child resolution",
            "IMAGE": "Multimodal Image Retrieval + Vision model description",
            "TABLE": "Structured Tabular Match + Markdown synthesis",
            "GRAPH": "Knowledge Graph traversal + Vector hybrid fusion",
            "HYBRID": "Cross-modal Reciprocal Rank Fusion",
        }
        return strategies.get(route, "Hybrid")
