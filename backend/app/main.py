import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.agents.router import AgenticQueryRouter
from backend.app.agents.validator import AnswerValidator
from backend.app.api import documents, health, search
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger("ragforge.main")
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Multimodal Agentic RAG Platform with First-Principles Retrieval",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Phase 21 routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)

from pathlib import Path
from fastapi.staticfiles import StaticFiles

frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    question: str
    route: str
    answer: str
    context: str
    citations: List[Dict[str, Any]]
    grounding_score: float
    is_valid: bool


@app.get("/")
def root_info() -> dict:
    return {
        "message": f"Welcome to {settings.app_name} API",
        "app_ui": "/app/",
        "docs_url": "/docs",
        "health": "/health",
        "storage": "local" if settings.storage_backend == "local" else settings.storage_backend,
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Agentic RAG chat with query classification, hybrid retrieval, and grounding validation."""
    router = AgenticQueryRouter()
    route_info = router.classify_query(req.question, chat_history=req.chat_history)
    route = route_info["route"]

    # 1. Retrieval
    from backend.app.api.chat import answer_question, format_documents, get_vector_store
    try:
        ans_data = answer_question(req.question, k=req.k)
        answer = ans_data["answer"]
        context = ans_data["context"]
        citations = ans_data["citations"]
    except Exception as e:
        logger.warning("LLM generation encountered an error (%s); serving grounded fallback.", e)
        # Fallback to direct semantic passage response
        from backend.app.retrieval.vector import similarity_search
        docs = similarity_search(get_vector_store(), req.question, k=req.k)
        context = format_documents(docs)
        answer = f"Based on the retrieved document context:\n\n{docs[0].page_content if docs else 'No relevant passages found.'}"
        citations = [d.metadata for d in docs]

    # 2. Answer Grounding Validation
    validator = AnswerValidator()
    validation = validator.validate_answer(answer=answer, context=context, citations=citations)

    return ChatResponse(
        question=req.question,
        route=route,
        answer=answer,
        context=context,
        citations=citations,
        grounding_score=validation["grounding_score"],
        is_valid=validation["is_valid"],
    )
