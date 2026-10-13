import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.app.retrieval.hybrid import doc_to_dict, reciprocal_rank_fusion

load_dotenv()

DEFAULT_REWRITE_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in reformulating user queries for an information retrieval system.\n"
    "Your task is to rewrite ambiguous, conversational, or follow-up search queries into a standalone, fully-explicit search query.\n"
    "- Resolve all ambiguous pronouns ('it', 'this', 'that', 'they', 'its') and coreferences using the conversation context.\n"
    "- Preserve the original semantic intent, technical vocabulary, and keywords.\n"
    "- If the query is already clear and self-contained, return it essentially unchanged.\n"
    "- Do NOT answer the question.\n"
    "- Output ONLY the rewritten search query text, without explanations, quotes, or formatting."
)

DEFAULT_EXPAND_SYSTEM_PROMPT = (
    "You are an expert retrieval optimization system.\n"
    "Given a search query, generate {num_queries} diverse search queries covering different perspectives, "
    "synonyms, technical terms, and complementary angles of the question.\n"
    "Format: Provide each alternative query on a new line prefixed with a number and period (e.g., '1. ...').\n"
    "Do NOT answer the question. Output ONLY the numbered queries, without preamble or conversational text."
)


_UNSET = object()


def _resolve_llm(model_name: Optional[str] = None) -> Optional[Any]:
    """Resolve default LLM instance, attempting ChatMistralAI if api key is present."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        try:
            from langchain_mistralai import ChatMistralAI
            chosen_model = model_name or os.getenv("MISTRAL_MODEL", "ministral-8b-2512")
            return ChatMistralAI(model_name=chosen_model, temperature=0.0)
        except Exception:
            return None
    return None


class QueryRewriter:
    """Phase 7: Reformulates ambiguous or conversational queries into standalone retrieval queries."""

    def __init__(
        self,
        llm: Any = _UNSET,
        system_prompt: str = DEFAULT_REWRITE_SYSTEM_PROMPT,
        model_name: Optional[str] = None,
    ) -> None:
        self.llm = _resolve_llm(model_name=model_name) if llm is _UNSET else llm
        self.system_prompt = system_prompt


    def rewrite(
        self,
        query: str,
        chat_history: Optional[Sequence[Union[Dict[str, str], BaseMessage, tuple]]] = None,
    ) -> Dict[str, str]:
        """Rewrite input query into a self-contained search query.
        
        Returns:
            Dict containing 'original_query' and 'rewritten_query'.
            The original query is NEVER overwritten.
        """
        clean_query = query.strip()
        if not clean_query:
            return {"original_query": query, "rewritten_query": ""}

        if self.llm is None:
            # Fallback when LLM is unavailable: return original query
            return {
                "original_query": query,
                "rewritten_query": clean_query,
            }

        # Build prompt messages
        messages = [SystemMessage(content=self.system_prompt)]

        if chat_history:
            history_text_parts = []
            for item in chat_history:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    history_text_parts.append(f"{role.capitalize()}: {content}")
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    history_text_parts.append(f"{item[0].capitalize()}: {item[1]}")
                elif isinstance(item, BaseMessage):
                    history_text_parts.append(f"{item.type.capitalize()}: {item.content}")

            if history_text_parts:
                history_block = "\n".join(history_text_parts)
                messages.append(
                    HumanMessage(content=f"Conversation Context:\n{history_block}\n\nCurrent Question: {clean_query}")
                )
            else:
                messages.append(HumanMessage(content=f"Question: {clean_query}"))
        else:
            messages.append(HumanMessage(content=f"Question: {clean_query}"))

        try:
            response = self.llm.invoke(messages)
            rewritten_text = response.content if hasattr(response, "content") else str(response)
            rewritten_text = rewritten_text.strip().strip('"\'')
            if not rewritten_text:
                rewritten_text = clean_query
        except Exception:
            rewritten_text = clean_query

        return {
            "original_query": query,
            "rewritten_query": rewritten_text,
        }


def rewrite_query(
    query: str,
    llm: Any = _UNSET,
    chat_history: Optional[Sequence[Union[Dict[str, str], BaseMessage, tuple]]] = None,
) -> Dict[str, str]:
    """Functional convenience helper for Phase 7 query rewriting."""
    rewriter = QueryRewriter(llm=llm)
    return rewriter.rewrite(query=query, chat_history=chat_history)


class QueryExpander:
    """Phase 8: Generates multiple diverse retrieval queries from a single user query."""

    def __init__(
        self,
        llm: Any = _UNSET,
        system_prompt: str = DEFAULT_EXPAND_SYSTEM_PROMPT,
        model_name: Optional[str] = None,
    ) -> None:
        self.llm = _resolve_llm(model_name=model_name) if llm is _UNSET else llm
        self.system_prompt = system_prompt


    def expand(
        self,
        query: str,
        num_queries: int = 3,
        include_original: bool = True,
    ) -> Dict[str, Any]:
        """Generate alternative query formulations.
        
        Returns:
            Dict containing 'original_query', 'expanded_queries', and 'all_queries'.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "original_query": query,
                "expanded_queries": [],
                "all_queries": [query] if include_original else [],
            }

        if self.llm is None or num_queries <= 0:
            return {
                "original_query": query,
                "expanded_queries": [],
                "all_queries": [clean_query],
            }

        prompt = self.system_prompt.format(num_queries=num_queries)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Query: {clean_query}"),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            
            # Parse lines (extracting numbered items or lines)
            extracted: List[str] = []
            for line in raw_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove prefixes like "1.", "1)", "-", "*"
                cleaned = re.sub(r"^(\d+[\.\)]|[-*•])\s*", "", line).strip().strip('"\'')
                if cleaned and cleaned.lower() != clean_query.lower() and cleaned not in extracted:
                    extracted.append(cleaned)

            expanded = extracted[:num_queries]
        except Exception:
            expanded = []

        all_queries = [clean_query]
        for q in expanded:
            if q not in all_queries:
                all_queries.append(q)

        return {
            "original_query": query,
            "expanded_queries": expanded,
            "all_queries": all_queries if include_original else expanded,
        }


def expand_query(
    query: str,
    num_queries: int = 3,
    llm: Any = _UNSET,
    include_original: bool = True,
) -> Dict[str, Any]:
    """Functional convenience helper for Phase 8 query expansion."""
    expander = QueryExpander(llm=llm)
    return expander.expand(query=query, num_queries=num_queries, include_original=include_original)



def multi_query_search(
    retriever: Any,
    queries: Sequence[str],
    top_k: int = 5,
    candidate_k: int = 20,
    k_rrf: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Phase 8 Multi-Query Fusion: Executes search for each query and fuses rankings via RRF.
    
    Args:
        retriever: An object exposing a `.search(query, top_k)` or `.similarity_search(query, k)` method,
                   or a callable `retriever(query, top_k)`.
        queries: Sequence of distinct queries (e.g., original query + expanded queries).
        top_k: Number of final fused documents to return.
        candidate_k: Number of candidates retrieved per individual query.
        k_rrf: Constant for Reciprocal Rank Fusion.
        weights: Optional weighting for each query ranking list.
    """
    if not queries:
        return []

    ranked_lists: List[List[Dict[str, Any]]] = []

    for q in queries:
        q_clean = q.strip()
        if not q_clean:
            continue

        if hasattr(retriever, "search"):
            hits = retriever.search(q_clean, top_k=candidate_k)
        elif hasattr(retriever, "similarity_search"):
            raw_docs = retriever.similarity_search(q_clean, k=candidate_k)
            hits = [doc_to_dict(d) for d in raw_docs]
        elif callable(retriever):
            hits = retriever(q_clean, candidate_k)
        else:
            raise AttributeError("Retriever must support .search(), .similarity_search(), or be a callable.")

        normalized_hits = [doc_to_dict(h) for h in hits]
        ranked_lists.append(normalized_hits)

    if not ranked_lists:
        return []

    # Fuse all ranked lists using reciprocal rank fusion
    fused_results = reciprocal_rank_fusion(
        ranked_lists=ranked_lists,
        k=k_rrf,
        weights=weights,
    )

    return fused_results[:top_k]
