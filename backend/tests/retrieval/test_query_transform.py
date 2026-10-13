import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from backend.app.retrieval.query_transform import (
    QueryRewriter,
    rewrite_query,
    QueryExpander,
    expand_query,
    multi_query_search,
)


def test_query_rewriter_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="What are the key limitations and trade-offs of Retrieval Augmented Generation?"
    )

    rewriter = QueryRewriter(llm=mock_llm)
    result = rewriter.rewrite(
        query="what about its limitations?",
        chat_history=[
            {"role": "user", "content": "Tell me about Retrieval Augmented Generation."},
            {"role": "assistant", "content": "RAG combines retrieval with LLM generation."},
        ],
    )

    # Validate output contract
    assert result["original_query"] == "what about its limitations?"
    assert result["rewritten_query"] == "What are the key limitations and trade-offs of Retrieval Augmented Generation?"
    mock_llm.invoke.assert_called_once()


def test_query_rewriter_preserves_original_query_on_empty_or_no_llm():
    # Without LLM, fallback should return original query safely
    rewriter = QueryRewriter(llm=None)
    result = rewriter.rewrite("What is polymorphism?")
    assert result["original_query"] == "What is polymorphism?"
    assert result["rewritten_query"] == "What is polymorphism?"

    # Empty query
    result_empty = rewriter.rewrite("")
    assert result_empty["original_query"] == ""
    assert result_empty["rewritten_query"] == ""


def test_rewrite_query_helper_function():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="How does inheritance work in object oriented programming?")

    result = rewrite_query(
        query="how does it work?",
        llm=mock_llm,
        chat_history=[("user", "Explain inheritance in OOP")],
    )

    assert result["original_query"] == "how does it work?"
    assert result["rewritten_query"] == "How does inheritance work in object oriented programming?"


def test_query_expander_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=(
            "1. How does retrieval augmented generation reduce hallucination in LLMs?\n"
            "2. How does external knowledge grounding improve model factuality?\n"
            "3. What mechanisms in RAG verify factual accuracy and citations?"
        )
    )

    expander = QueryExpander(llm=mock_llm)
    result = expander.expand(query="How does RAG reduce hallucinations?", num_queries=3)

    assert result["original_query"] == "How does RAG reduce hallucinations?"
    assert len(result["expanded_queries"]) == 3
    assert "How does retrieval augmented generation reduce hallucination in LLMs?" in result["expanded_queries"]
    assert len(result["all_queries"]) == 4
    assert result["all_queries"][0] == "How does RAG reduce hallucinations?"


def test_query_expander_no_llm_fallback():
    expander = QueryExpander(llm=None)
    result = expander.expand(query="Test query", num_queries=3)

    assert result["original_query"] == "Test query"
    assert result["expanded_queries"] == []
    assert result["all_queries"] == ["Test query"]
