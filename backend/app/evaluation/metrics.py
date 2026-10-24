import math
import re
from typing import Any, Dict, List, Sequence, Set


def calculate_recall_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int = 5) -> float:
    """Calculate Recall@K: fraction of relevant items retrieved in top-k."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(str(gid).lower() for gid in ground_truth_ids)
    hits = sum(1 for rid in top_k if str(rid).lower() in gt_set)
    return min(1.0, hits / len(gt_set))


def calculate_precision_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int = 5) -> float:
    """Calculate Precision@K: fraction of top-k items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(str(gid).lower() for gid in ground_truth_ids)
    hits = sum(1 for rid in top_k if str(rid).lower() in gt_set)
    return hits / k


def calculate_mrr(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1 / rank of first relevant item."""
    gt_set = set(str(gid).lower() for gid in ground_truth_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if str(rid).lower() in gt_set:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    gt_set = set(str(gid).lower() for gid in ground_truth_ids)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if str(rid).lower() in gt_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG (all relevant items ranked at top)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gt_set), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_faithfulness(answer: str, context: str) -> float:
    """Calculate Answer Faithfulness: ratio of claims supported by context."""
    if not answer.strip() or not context.strip():
        return 0.0

    claims = [c.strip() for c in re.split(r"[.\n]", answer) if len(c.strip()) > 15]
    if not claims:
        claims = [answer.strip()]

    context_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", context.lower()))
    supported = 0
    for claim in claims:
        claim_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", claim.lower()))
        if claim_words and (len(claim_words & context_words) / len(claim_words)) >= 0.35:
            supported += 1

    return supported / len(claims) if claims else 0.0
