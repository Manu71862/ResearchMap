"""
Phase 5B Controlled Evaluation Comparison:
A. Existing BM25 Configuration C
B. Existing Hybrid RRF (Phase 5A Unoptimized Baseline)
C. Optimized Hybrid RRF (Phase 5B Sub-500ms CPU Target)

Evaluated under identical conditions:
- Expanded 10,835-document corpus (data/experimental/corpus_expanded_10k.jsonl)
- Fine-tuned NLI checkpoint (models/finetuned_nli)
- 24 diagnostic claims (experiments/new_claims.jsonl)
- 161 validation claims (data/processed/val_internal.jsonl)
- Decision thresholds (ENTAIL_THRESHOLD=0.50, CONTRADICT_THRESHOLD=0.50)
"""

import os
import sys
import json
import time
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from typing import Dict, List, Any, Tuple, Set
import numpy as np

from backend.config import Config
from backend.data_loader import (
    load_corpus,
    load_claims,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.pipeline import ClaimVerificationPipeline

from scripts.experimental.dense_retriever import DenseRetriever
from scripts.experimental.hybrid_retriever import HybridRetriever
from scripts.experimental.optimized_hybrid_retriever import OptimizedHybridRetriever

EXPANDED_CORPUS_PATH = BASE_DIR / "data" / "experimental" / "corpus_expanded_10k.jsonl"
TEST_CLAIMS_PATH = BASE_DIR / "experiments" / "new_claims.jsonl"
VAL_CLAIMS_PATH = BASE_DIR / "data" / "processed" / "val_internal.jsonl"
CACHE_DIR = BASE_DIR / "experiments" / "cache"
EMB_PATH = CACHE_DIR / "doc_embeddings_minilm_10k.npy"
IDS_PATH = CACHE_DIR / "doc_ids_10k.json"
RESULTS_JSON_PATH = BASE_DIR / "experiments" / "phase5b_optimization_results.json"

TOP_K_DOCS = 10
TOP_K_SENTENCES = 4
DOC_SCORE_WEIGHT = 0.35
MAX_PER_DOC = 2

def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = CANONICAL_LABELS) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / n

    per_class = {}
    f1_list = []
    cm = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    cm_matrix = [[cm[l1][l2] for l2 in labels] for l1 in labels]

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm_matrix,
        "total": n,
        "correct": correct
    }

def compute_retrieval_metrics(eval_claims: List[Dict[str, Any]], retriever) -> Dict[str, Any]:
    doc_r5_hits = 0
    doc_r10_hits = 0
    doc_r20_hits = 0
    sent_hits = 0
    reciprocal_ranks_10 = []
    reciprocal_ranks_20 = []
    ndcg_10_scores = []

    answerable_count = 0

    for item in eval_claims:
        gold_doc_ids: Set[int] = item.get("gold_doc_ids", set())
        gold_sentences: Set[Tuple[int, int]] = item.get("gold_sentences", set())

        if not gold_doc_ids:
            continue
        answerable_count += 1

        claim_text = item["claim"]
        docs_20 = retriever.retrieve_documents(claim_text, top_k=20)
        retrieved_ids = [d["doc_id"] for d in docs_20]

        hit5 = any(did in gold_doc_ids for did in retrieved_ids[:5])
        hit10 = any(did in gold_doc_ids for did in retrieved_ids[:10])
        hit20 = any(did in gold_doc_ids for did in retrieved_ids[:20])

        if hit5:
            doc_r5_hits += 1
        if hit10:
            doc_r10_hits += 1
        if hit20:
            doc_r20_hits += 1

        rr10 = 0.0
        for rank, did in enumerate(retrieved_ids[:10], start=1):
            if did in gold_doc_ids:
                rr10 = 1.0 / rank
                break
        reciprocal_ranks_10.append(rr10)

        rr20 = 0.0
        for rank, did in enumerate(retrieved_ids[:20], start=1):
            if did in gold_doc_ids:
                rr20 = 1.0 / rank
                break
        reciprocal_ranks_20.append(rr20)

        dcg10 = sum(
            1.0 / math.log2(rank + 1)
            for rank, did in enumerate(retrieved_ids[:10], start=1)
            if did in gold_doc_ids
        )
        ideal_count = min(len(gold_doc_ids), 10)
        idcg10 = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1)) if ideal_count > 0 else 1.0
        ndcg_10_scores.append(dcg10 / idcg10 if idcg10 > 0 else 0.0)

        docs_10 = docs_20[:TOP_K_DOCS]
        sentences_top = retriever.retrieve_sentences(
            claim_text,
            docs_10,
            top_k_sentences=TOP_K_SENTENCES,
            doc_score_weight=DOC_SCORE_WEIGHT,
            max_per_doc=MAX_PER_DOC
        )
        retrieved_sents = {(s["doc_id"], s["sentence_idx"]) for s in sentences_top}
        if gold_sentences and (retrieved_sents & gold_sentences):
            sent_hits += 1

    if answerable_count == 0:
        return {}

    return {
        "answerable_count": answerable_count,
        "doc_recall@5": round(doc_r5_hits / answerable_count, 4),
        "doc_recall@10": round(doc_r10_hits / answerable_count, 4),
        "doc_recall@20": round(doc_r20_hits / answerable_count, 4),
        "sentence_evidence_recall": round(sent_hits / answerable_count, 4),
        "doc_mrr@10": round(sum(reciprocal_ranks_10) / answerable_count, 4),
        "doc_mrr@20": round(sum(reciprocal_ranks_20) / answerable_count, 4),
        "doc_ndcg@10": round(sum(ndcg_10_scores) / answerable_count, 4),
    }

def evaluate_approach(
    name: str,
    retriever,
    nli_model: NLIEntailmentModel,
    test_claims: List[Dict[str, Any]],
    val_claims: List[Dict[str, Any]],
    index_time: float,
    ram_mb: float,
    disk_mb: float,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating Approach: {name}")
    print(f"=======================================================")

    pipeline = ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=nli_model,
        top_k_docs=TOP_K_DOCS,
        top_k_sentences=TOP_K_SENTENCES,
        entail_threshold=Config.ENTAIL_THRESHOLD,
        contradict_threshold=Config.CONTRADICT_THRESHOLD,
    )

    # 1. Held-out test evaluation (N=24)
    test_preds = []
    latencies_test = []
    y_true_test = []
    y_pred_test = []

    for item in test_claims:
        cid = item["id"]
        claim_text = item["claim"]
        gold_label = item["gold_label"]
        y_true_test.append(gold_label)

        t_start = time.time()
        res = pipeline.verify_claim(claim_text, claim_id=cid)
        dt_ms = (time.time() - t_start) * 1000.0
        latencies_test.append(dt_ms)

        pred_label = res["predicted_label"]
        y_pred_test.append(pred_label)

        test_preds.append({
            "claim_id": cid,
            "claim": claim_text,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "confidence": round(float(res.get("confidence", 0.0)), 4),
            "is_correct": pred_label == gold_label,
            "retrieved_candidate_doc_ids": res.get("retrieved_candidate_doc_ids", []),
            "predicted_evidence_sentences": res.get("predicted_evidence_sentences", []),
            "latency_ms": round(dt_ms, 2)
        })

    test_metrics = compute_metrics(y_true_test, y_pred_test)
    test_retrieval = compute_retrieval_metrics(test_claims, retriever)
    mean_test_latency = round(float(np.mean(latencies_test)), 2)
    p95_test_latency = round(float(np.percentile(latencies_test, 95)), 2)

    # 2. Validation evaluation (N=161)
    val_preds = []
    latencies_val = []
    y_true_val = []
    y_pred_val = []

    for item in val_claims:
        cid = item["id"]
        claim_text = item["claim"]
        gold_label = item["gold_label"]
        y_true_val.append(gold_label)

        t_start = time.time()
        res = pipeline.verify_claim(claim_text, claim_id=cid)
        dt_ms = (time.time() - t_start) * 1000.0
        latencies_val.append(dt_ms)

        pred_label = res["predicted_label"]
        y_pred_val.append(pred_label)

    val_metrics = compute_metrics(y_true_val, y_pred_val)
    val_retrieval = compute_retrieval_metrics(val_claims, retriever)
    mean_val_latency = round(float(np.mean(latencies_val)), 2)
    p95_val_latency = round(float(np.percentile(latencies_val, 95)), 2)

    print(f"  Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test Doc R@10: {test_retrieval.get('doc_recall@10', 0.0)*100:.2f}% | Sent Recall: {test_retrieval.get('sentence_evidence_recall', 0.0)*100:.2f}%")
    print(f"  Test Latency:  Mean={mean_test_latency} ms | P95={p95_test_latency} ms")
    print(f"  Val Accuracy:  {val_metrics['accuracy']*100:.2f}% | Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Val Doc R@10:  {val_retrieval.get('doc_recall@10', 0.0)*100:.2f}% | Sent Recall: {val_retrieval.get('sentence_evidence_recall', 0.0)*100:.2f}%")
    print(f"  Val Latency:   Mean={mean_val_latency} ms | P95={p95_val_latency} ms")

    claim_24_result = next((p for p in test_preds if p["claim_id"] == 24), {})

    return {
        "approach_name": name,
        "index_build_time_seconds": round(index_time, 2),
        "ram_footprint_mb": round(ram_mb, 2),
        "disk_footprint_mb": round(disk_mb, 2),
        "mean_test_latency_ms": mean_test_latency,
        "p95_test_latency_ms": p95_test_latency,
        "mean_val_latency_ms": mean_val_latency,
        "p95_val_latency_ms": p95_val_latency,
        "test_metrics_n24": {
            "overall_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "retrieval": test_retrieval,
        },
        "val_metrics_n161": {
            "overall_accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "per_class": val_metrics["per_class"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "retrieval": val_retrieval,
        },
        "claim_24_diagnostic": claim_24_result,
        "test_predictions": test_preds,
    }

def main():
    print("=" * 70)
    print("ResearchMap Phase 5B: Controlled Optimization Benchmark")
    print("=" * 70)

    # 1. Load corpus
    t0 = time.time()
    corpus = load_corpus(EXPANDED_CORPUS_PATH)
    print(f"Loaded {len(corpus)} documents in {time.time()-t0:.2f}s.")

    # 2. Load NLI model
    model_path = Config.MODELS_DIR / "finetuned_nli"
    nli_model = NLIEntailmentModel(model_name=str(model_path), device="cpu")

    # 3. Load datasets
    with open(TEST_CLAIMS_PATH, "r", encoding="utf-8") as f:
        raw_test = [json.loads(line) for line in f if line.strip()]

    test_claims = []
    for c in raw_test:
        gold_dids = set()
        if c.get("evidence_doc_id") is not None:
            gold_dids.add(int(c["evidence_doc_id"]))
        gold_sents = set()
        did = c.get("evidence_doc_id")
        if did is not None:
            for sidx in c.get("gold_sentence_indices", []):
                gold_sents.add((int(did), int(sidx)))

        test_claims.append({
            "id": c["id"],
            "claim": c["claim"],
            "gold_label": c["gold_label"],
            "evidence_doc_id": c.get("evidence_doc_id"),
            "gold_doc_ids": gold_dids,
            "gold_sentences": gold_sents,
            "answerable_from_corpus": c.get("answerable_from_corpus", False)
        })

    raw_val = load_claims(VAL_CLAIMS_PATH)
    val_claims = []
    for c in raw_val:
        val_claims.append({
            "id": c["id"],
            "claim": c["claim"],
            "gold_label": c["gold_label"],
            "gold_doc_ids": c.get("gold_doc_evidence", set()),
            "gold_sentences": c.get("gold_sentence_evidence", set())
        })

    print(f"Loaded {len(test_claims)} test claims and {len(val_claims)} validation claims.")

    # 4. Initialize Retrievers
    # A. BM25 Baseline
    t0 = time.time()
    bm25 = BM25Retriever(corpus)
    bm25_time = time.time() - t0
    bm25_ram = 35.0
    bm25_disk = 0.0

    # B. Existing Dense & Unoptimized Hybrid RRF
    t0 = time.time()
    dense = DenseRetriever(corpus, EMB_PATH, IDS_PATH, device="cpu")
    dense_time = time.time() - t0
    hybrid_unopt = HybridRetriever(bm25, dense, rrf_k=60)
    hybrid_time = bm25_time + dense_time
    hybrid_ram = 114.6
    hybrid_disk = 15.87

    # C. Optimized Hybrid RRF
    opt_hybrid = OptimizedHybridRetriever(
        bm25, dense, rrf_k=60, candidate_pool=50, sentence_doc_gating=5
    )
    opt_time = hybrid_time
    opt_ram = 114.6
    opt_disk = 15.87

    # Run controlled 3-way evaluation
    res_a = evaluate_approach(
        "A. Existing BM25 Configuration C",
        bm25, nli_model, test_claims, val_claims,
        bm25_time, bm25_ram, bm25_disk
    )
    res_b = evaluate_approach(
        "B. Existing Hybrid RRF (Phase 5A Unoptimized)",
        hybrid_unopt, nli_model, test_claims, val_claims,
        hybrid_time, hybrid_ram, hybrid_disk
    )
    res_c = evaluate_approach(
        "C. Optimized Hybrid RRF (Phase 5B)",
        opt_hybrid, nli_model, test_claims, val_claims,
        opt_time, opt_ram, opt_disk
    )

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_document_count": len(corpus),
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2 (22.7M parameters)",
        "nli_model": str(model_path),
        "target_latency_ms": 500.0,
        "approaches": {
            "A_BM25": res_a,
            "B_Hybrid_RRF_Unoptimized": res_b,
            "C_Hybrid_RRF_Optimized": res_c,
        }
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f_out:
        json.dump(results_data, f_out, indent=2)
    print(f"\nSaved Phase 5B optimization results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
