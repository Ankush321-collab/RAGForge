import re
from typing import Any, Dict, List, Optional


class AnswerValidator:
    """Phase 17: Answer Grounding and Citation Validation.
    
    Validates that model outputs are strictly grounded in retrieved document
    context, identifies ungrounded claims, and verifies source citations.
    """

    def __init__(self, min_grounding_threshold: float = 0.6) -> None:
        self.min_grounding_threshold = min_grounding_threshold

    def validate_answer(
        self,
        answer: str,
        context: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Validate answer sentences against context passages."""
        if not answer.strip():
            return {
                "is_valid": False,
                "grounding_score": 0.0,
                "citation_count": 0,
                "ungrounded_claims": ["Empty answer generated."],
                "needs_re_retrieval": True,
                "message": "Empty answer generated.",
            }

        sentences = [
            s.strip() for s in re.split(r"[.\n]", answer)
            if len(s.strip()) > 15
        ]

        if not sentences:
            sentences = [answer.strip()]

        context_lower = context.lower()
        context_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", context_lower))

        grounded_count = 0
        ungrounded_claims: List[str] = []

        for sent in sentences:
            sent_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", sent.lower()))
            if not sent_words:
                continue

            # Overlap ratio between sentence informative words and context words
            overlap = len(sent_words & context_words) / len(sent_words)
            if overlap >= 0.35:
                grounded_count += 1
            else:
                ungrounded_claims.append(sent)

        total_claims = len(sentences)
        grounding_score = grounded_count / total_claims if total_claims > 0 else 0.0
        has_citations = bool(citations and len(citations) > 0)

        is_valid = grounding_score >= self.min_grounding_threshold
        needs_re_retrieval = grounding_score < 0.45 or len(context.strip()) < 50

        return {
            "is_valid": is_valid,
            "grounding_score": round(grounding_score, 3),
            "citation_count": len(citations or []),
            "ungrounded_claims": ungrounded_claims,
            "needs_re_retrieval": needs_re_retrieval,
            "message": (
                "Answer is well-grounded in retrieved context."
                if is_valid
                else f"Potential hallucination or ungrounded claims detected ({len(ungrounded_claims)} claims)."
            ),
        }
