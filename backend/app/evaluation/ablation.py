from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .dataset import EvaluationDataset, EvaluationSample
from .metrics import (
    calculate_faithfulness,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)


class AblationStudy:
    """Phase 19: Ablation Study benchmarking retrieval strategies."""

    def __init__(self, dataset: Optional[EvaluationDataset] = None) -> None:
        self.dataset = dataset or EvaluationDataset()

    def run_strategy_benchmark(
        self,
        strategy_name: str,
        retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
        k: int = 5,
    ) -> Dict[str, float]:
        """Run benchmark for a given retrieval pipeline strategy."""
        recalls: List[float] = []
        mrrs: List[float] = []
        ndcgs: List[float] = []

        for sample in self.dataset.get_samples():
            results = retriever_fn(sample.question, k)
            retrieved_ids = [
                str(r.get("chunk_id") or r.get("id") or r.get("page_number"))
                for r in results
            ]
            # Ground truth targets
            targets = [str(sample.relevant_page), sample.relevant_doc_id]

            recalls.append(calculate_recall_at_k(retrieved_ids, targets, k=k))
            mrrs.append(calculate_mrr(retrieved_ids, targets))
            ndcgs.append(calculate_ndcg_at_k(retrieved_ids, targets, k=k))

        return {
            "strategy": strategy_name,
            "recall@5": round(float(np.mean(recalls)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "ndcg@5": round(float(np.mean(ndcgs)), 4),
        }

    @staticmethod
    def format_ablation_markdown_table(benchmark_results: List[Dict[str, Any]]) -> str:
        """Format ablation comparison results as a Markdown table."""
        headers = ["Strategy", "Recall@5", "MRR", "NDCG@5"]
        sep = ["---", "---", "---", "---"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]

        for res in benchmark_results:
            lines.append(
                f"| {res['strategy']} | {res['recall@5']:.4f} | {res['mrr']:.4f} | {res['ndcg@5']:.4f} |"
            )

        return "\n".join(lines)
