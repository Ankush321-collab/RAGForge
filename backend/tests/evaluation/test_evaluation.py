from backend.app.evaluation.dataset import EvaluationDataset, EvaluationSample
from backend.app.evaluation.metrics import (
    calculate_faithfulness,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from backend.app.evaluation.ablation import AblationStudy


def test_retrieval_metrics():
    retrieved = ["doc_3", "doc_1", "doc_7", "doc_2"]
    ground_truth = ["doc_1", "doc_2"]

    # Recall@4: both doc_1 and doc_2 found -> 1.0
    recall = calculate_recall_at_k(retrieved, ground_truth, k=4)
    assert recall == 1.0

    # Precision@2: doc_1 is in top 2 (1/2 = 0.5)
    prec = calculate_precision_at_k(retrieved, ground_truth, k=2)
    assert prec == 0.5

    # MRR: first relevant is doc_1 at rank 2 -> 1/2 = 0.5
    mrr = calculate_mrr(retrieved, ground_truth)
    assert mrr == 0.5

    # NDCG
    ndcg = calculate_ndcg_at_k(retrieved, ground_truth, k=4)
    assert ndcg > 0.0


def test_faithfulness_metric():
    context = "The proposed Transformer achieves 94% BLEU score on WMT translation."
    faithful_answer = "The Transformer reached 94% BLEU score on translation."
    hallucinated_answer = "The ResNet model achieved state of the art on ImageNet."

    assert calculate_faithfulness(faithful_answer, context) >= 0.7
    assert calculate_faithfulness(hallucinated_answer, context) < 0.4


def test_ablation_study_table_formatting():
    results = [
        {"strategy": "Dense Vector", "recall@5": 0.6500, "mrr": 0.4200, "ndcg@5": 0.5500},
        {"strategy": "BM25 Sparse", "recall@5": 0.7000, "mrr": 0.4800, "ndcg@5": 0.6000},
        {"strategy": "Hybrid RRF", "recall@5": 0.8500, "mrr": 0.6400, "ndcg@5": 0.7500},
        {"strategy": "Hybrid + Reranker", "recall@5": 0.9500, "mrr": 0.8200, "ndcg@5": 0.8800},
    ]

    table = AblationStudy.format_ablation_markdown_table(results)
    assert "| Strategy | Recall@5 | MRR | NDCG@5 |" in table
    assert "| Hybrid + Reranker | 0.9500 | 0.8200 | 0.8800 |" in table
