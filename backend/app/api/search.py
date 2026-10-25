import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.logging import QueryTracer
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.parent_child import ParentChildRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.vector import get_vector_store

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=100)
    mode: str = Field(default="hybrid", description="'hybrid', 'vector_only', or 'bm25_only'")
    filter: Optional[Dict[str, Any]] = None
    document_id: Optional[str] = None
    rerank: bool = True


class SearchHit(BaseModel):
    id: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page_number: Optional[int] = 1
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_hits: int
    execution_time_ms: float
    hits: List[SearchHit]


@router.post("/search", response_model=SearchResponse)
def execute_search(req: SearchRequest) -> SearchResponse:
    tracer = QueryTracer(req.query)
    t0 = time.time()

    search_filter = req.filter or {}
    if req.document_id and req.document_id != "all":
        search_filter["document_id"] = req.document_id

    try:
        from backend.app.api.chat import get_vector_store
        vector_store = get_vector_store()
    except Exception:
        vector_store = None

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=None,
    )

    t_ret_start = time.time()
    try:
        hits = hybrid_retriever.search(
            query=req.query,
            top_k=req.top_k if not req.rerank else req.candidate_k,
            mode=req.mode,
            filter=search_filter if search_filter else None,
        )
    except Exception as e:
        hits = []

    tracer.record_stage("retrieval", (time.time() - t_ret_start) * 1000)

    # Optional Reranker
    if req.rerank and hits:
        t_rerank_start = time.time()
        try:
            reranker = CrossEncoderReranker()
            hits = reranker.rerank(req.query, hits, top_k=req.top_k)
        except Exception:
            hits = hits[:req.top_k]
        tracer.record_stage("reranking", (time.time() - t_rerank_start) * 1000)
    else:
        hits = hits[:req.top_k]

    total_time_ms = round((time.time() - t0) * 1000, 2)
    tracer.finish()

    formatted_hits = [
        SearchHit(
            id=str(h.get("chunk_id") or h.get("id") or idx),
            chunk_id=h.get("chunk_id"),
            document_id=h.get("document_id"),
            page_number=h.get("page_number", 1),
            score=h.get("score") or h.get("rrf_score"),
            rerank_score=h.get("rerank_score"),
            text=h.get("text", ""),
        )
        for idx, h in enumerate(hits, start=1)
    ]

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        total_hits=len(formatted_hits),
        execution_time_ms=total_time_ms,
        hits=formatted_hits,
    )
