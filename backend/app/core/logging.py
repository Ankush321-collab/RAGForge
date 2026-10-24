import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("ragforge")


class QueryTracer:
    """Phase 20: Latency, token, and stage observability tracer for RAG queries."""

    def __init__(self, query: str, request_id: Optional[str] = None) -> None:
        self.request_id = request_id or str(uuid4())[:8]
        self.original_query = query
        self.rewritten_query: Optional[str] = None
        self.route: str = "TEXT"
        self.retrieval_strategy: str = "hybrid"
        self.retrieved_document_ids: List[str] = []
        self.reranker_scores: List[float] = []
        self.timings: Dict[str, float] = {}
        self.final_answer_length: int = 0
        self.start_time = time.time()

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        self.timings[stage_name] = round(duration_ms, 2)

    def finish(self, answer: str = "", citations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        total_ms = (time.time() - self.start_time) * 1000
        self.timings["total_ms"] = round(total_ms, 2)
        self.final_answer_length = len(answer)

        trace_log = {
            "request_id": self.request_id,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query or self.original_query,
            "route": self.route,
            "retrieval_strategy": self.retrieval_strategy,
            "retrieved_document_ids": self.retrieved_document_ids,
            "reranker_scores": self.reranker_scores,
            "timings_ms": self.timings,
            "answer_length": self.final_answer_length,
            "citation_count": len(citations or []),
        }

        logger.info("[QUERY_TRACE] %s", json.dumps(trace_log))
        return trace_log


def configure_logging(level: int = logging.INFO) -> None:
    """Configure system-wide unified logger format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
