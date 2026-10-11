import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.storage.s3 import S3Storage
from backend.app.retrieval.opensearch import OpenSearchVectorStore
from backend.app.retrieval.hybrid import reciprocal_rank_fusion
from backend.app.retrieval.reranker import CrossEncoderReranker
import boto3


def test_s3_pipeline(pdf_path: Path):
    print("\n" + "=" * 70)
    print(" [1/4] TESTING S3 STORAGE & ASSET PIPELINE")
    print("=" * 70)
    storage = S3Storage()
    storage.ensure_bucket()

    doc_id = "oops_hard_test"
    object_key = storage.build_object_key(doc_id, pdf_path.name)
    file_bytes = pdf_path.read_bytes()

    print(f"  -> Uploading '{pdf_path.name}' ({len(file_bytes):,} bytes) to bucket '{storage.bucket_name}'...")
    storage.upload_bytes(file_bytes, object_key, "application/pdf")
    print(f"  -> S3 Object uploaded successfully at key: {object_key}")

    presigned_url = storage.generate_presigned_url(object_key, expires_in=300)
    print(f"  -> Presigned URL generated: {presigned_url[:75]}...")

    # Verify download
    downloaded = storage.download_bytes(object_key)
    assert len(downloaded) == len(file_bytes), "Downloaded S3 bytes do not match uploaded bytes!"
    print(f"  -> S3 verification PASSED: roundtrip verified ({len(downloaded):,} bytes)")
    return True


def test_sqs_pipeline():
    print("\n" + "=" * 70)
    print(" [2/4] TESTING AWS SQS INGESTION QUEUE")
    print("=" * 70)
    load_dotenv()
    sqs_url = os.getenv("AWS_SQS_QUEUE_URL")
    region = os.getenv("S3_REGION", "ap-south-2")
    access_key = os.getenv("S3_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY")

    if not sqs_url:
        print("  -> AWS_SQS_QUEUE_URL not specified in .env. Skipping SQS.")
        return False

    print(f"  -> Target Queue URL: {sqs_url}")
    sqs_client = boto3.client(
        "sqs",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        response = sqs_client.send_message(
            QueueUrl=sqs_url,
            MessageBody='{"event": "DOCUMENT_INGESTION_TEST", "document": "oops_notes.pdf"}',
        )
        msg_id = response.get("MessageId")
        print(f"  -> SQS message successfully dispatched! MessageId: {msg_id}")

        # Receive message to verify queue reading
        recv_resp = sqs_client.receive_message(
            QueueUrl=sqs_url, MaxNumberOfMessages=1, WaitTimeSeconds=2
        )
        messages = recv_resp.get("Messages", [])
        assert len(messages) > 0, "No message received from SQS queue!"
        received_msg = messages[0]
        print(f"  -> SQS message consumed: {received_msg.get('Body')}")

        # Delete message to clean up queue
        sqs_client.delete_message(
            QueueUrl=sqs_url, ReceiptHandle=received_msg["ReceiptHandle"]
        )
        print("  -> SQS message acknowledged & deleted successfully from queue.")
        return True
    except Exception as e:
        print(f"  -> SQS Pipeline Error: {e}")
        return False


def test_opensearch_cluster():
    print("\n" + "=" * 70)
    print(" [3/4] TESTING OPENSEARCH CLUSTER & INDEX HEALTH")
    print("=" * 70)
    store = OpenSearchVectorStore()
    try:
        cluster_info = store.client.info()
        version = cluster_info.get("version", {}).get("number", "Serverless")
        print(f"  -> OpenSearch Cluster Connected! Version: {version}")
    except Exception:
        print("  -> Connected to AWS OpenSearch Serverless (AOSS) Endpoint.")

    count_resp = store.client.count(index=store.index_name)
    total_docs = count_resp.get("count", 0)
    print(f"  -> Index '{store.index_name}' contains {total_docs:,} indexed passages.")
    assert total_docs > 0, "OpenSearch index is empty!"
    return store


def run_hard_queries(store: OpenSearchVectorStore, reranker: CrossEncoderReranker):
    print("\n" + "=" * 70)
    print(" [4/4] EXECUTING 3 HARD OOP BENCHMARK TESTS")
    print("       (BM25 + Vector Similarity -> RRF Hybrid Fusion -> Cross-Encoder Rerank)")
    print("=" * 70)

    hard_test_cases = [
        {
            "id": "HARD TEST 1",
            "concept": "Virtual Functions, VTABLE & VPTR Internal Mechanics",
            "query": "How do virtual functions work internally using vtable and vptr in dynamic dispatch?",
            "expected_keywords": ["virtual", "vtable", "vptr", "pointer", "function"],
        },
        {
            "id": "HARD TEST 2",
            "concept": "Virtual Destructors & Memory Leak Prevention",
            "query": "Why should a base class destructor be declared virtual when deleting derived objects through base pointer?",
            "expected_keywords": ["destructor", "virtual", "delete", "base", "derived"],
        },
        {
            "id": "HARD TEST 3",
            "concept": "Diamond Problem in Multiple Inheritance & Virtual Base Class",
            "query": "What is the diamond problem in multiple inheritance and how is it resolved using virtual base class?",
            "expected_keywords": ["inheritance", "multiple", "ambiguity", "virtual", "base"],
        },
    ]

    results_summary = []

    for test_case in hard_test_cases:
        t_id = test_case["id"]
        concept = test_case["concept"]
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]

        print(f"\n----------------------------------------------------------------------")
        print(f" >>> {t_id}: {concept}")
        print(f" >>> Query: \"{query}\"")
        print(f"----------------------------------------------------------------------")

        t0 = time.time()
        # 1. Sparse BM25 Search
        bm25_hits = store.bm25_search(query, top_k=15)
        t_bm25 = time.time() - t0

        t1 = time.time()
        # 2. Dense Vector Search
        vector_hits = store.vector_search(query, top_k=15)
        t_vector = time.time() - t1

        # 3. Hybrid RRF Fusion
        t2 = time.time()
        fused_candidates = reciprocal_rank_fusion([bm25_hits, vector_hits], k=60)
        t_rrf = time.time() - t2

        # 4. Cross-Encoder Reranking
        t3 = time.time()
        top_reranked = reranker.rerank(query=query, documents=fused_candidates[:15], top_k=3)
        t_rerank = time.time() - t3

        total_time = (t_bm25 + t_vector + t_rrf + t_rerank) * 1000

        print(f"  [Timing] BM25: {t_bm25*1000:.1f}ms | Vector: {t_vector*1000:.1f}ms | RRF: {t_rrf*1000:.1f}ms | Rerank: {t_rerank*1000:.1f}ms | Total: {total_time:.1f}ms")
        print(f"  [Candidates] BM25 hits: {len(bm25_hits)} | Vector hits: {len(vector_hits)} | Fused: {len(fused_candidates)}")

        assert len(top_reranked) > 0, f"Reranking produced no hits for {t_id}!"
        top_hit = top_reranked[0]
        top_text = top_hit.get("text", "")
        top_score = top_hit.get("rerank_score", 0.0)
        page_num = top_hit.get("page_number", "N/A")

        # Check keyword matches in top hit
        matches = [kw for kw in expected_keywords if kw.lower() in top_text.lower()]
        accuracy_pct = (len(matches) / len(expected_keywords)) * 100

        print(f"\n  ★ TOP RERANKED ANSWER (Page {page_num}, Score: {top_score:+.4f}, Keyword Coverage: {accuracy_pct:.0f}%):")
        print(f"    \"{top_text.strip()[:240]}...\"\n")

        print("  Comparison of Top Hit Rankings across Stages:")
        print(f"    - BM25 Top-1 Page      : {bm25_hits[0].get('page_number')} (score: {bm25_hits[0].get('score'):.2f})")
        print(f"    - Vector Top-1 Page    : {vector_hits[0].get('page_number')} (score: {vector_hits[0].get('score'):.4f})")
        print(f"    - RRF Fused Top-1 Page : {fused_candidates[0].get('page_number')} (RRF score: {fused_candidates[0].get('rrf_score'):.5f})")
        print(f"    - Final Reranked Page  : {page_num} (Cross-Encoder score: {top_score:+.4f})")

        results_summary.append({
            "test_id": t_id,
            "concept": concept,
            "page": page_num,
            "score": top_score,
            "keywords_matched": f"{len(matches)}/{len(expected_keywords)}",
            "latency_ms": f"{total_time:.1f}ms",
        })

    print("\n" + "=" * 70)
    print(" SUMMARY OF 3 HARD OOP TEST RUNS")
    print("=" * 70)
    for res in results_summary:
        print(f"  {res['test_id']}: {res['concept']}")
        print(f"    -> Page: {res['page']} | Cross-Encoder Score: {res['score']:+.4f} | Coverage: {res['keywords_matched']} | Latency: {res['latency_ms']}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    pdf_file = Path("data/raw/oops_notes.pdf")
    if not pdf_file.exists():
        print(f"Error: {pdf_file} not found!")
        sys.exit(1)

    print("\n" + "#" * 70)
    print("  RAGFORGE HARD PIPELINE TEST ON OOPS_NOTES.PDF")
    print("  S3 -> SQS -> OpenSearch -> BM25 + Vector -> RRF Hybrid -> CrossEncoder")
    print("#" * 70)

    # 1. S3 Test
    test_s3_pipeline(pdf_file)

    # 2. SQS Test
    test_sqs_pipeline()

    # 3. OpenSearch Test
    os_store = test_opensearch_cluster()

    # 4. CrossEncoder initialization & 3 Hard Queries
    print("\n  -> Initializing Cross-Encoder Reranker ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
    reranker = CrossEncoderReranker()
    # Trigger model warmup
    _ = reranker._get_model()
    print("  -> Cross-Encoder ready.")

    run_hard_queries(os_store, reranker)
