"""
Phase 4B: Comparative Evaluation of Original 5k Corpus vs Expanded 10k Corpus.

Evaluates both:
Configuration A: Original 5,183-abstract corpus + Retrieval Configuration C
Configuration B: Expanded ~10,000-abstract corpus + Retrieval Configuration C

Metrics Measured:
- Index build time (seconds)
- Mean query latency (ms)
- Overall accuracy, Answerable accuracy, Macro-F1
- Per-class Precision, Recall, F1 (SUPPORTS, CONTRADICTS, NEI)
- Document Recall@5, @10, @20
- Sentence evidence recall
- Document MRR@10, MRR@20, nDCG@10
- Answerable vs Unanswerable subset breakdown
- Per-claim change tracking and newly surfaced evidence analysis

Outputs:
- experiments/corpus_expansion_comparison.json
"""

import os
import sys
import json
import time
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from typing import Dict, List, Any, Tuple, Set
from collections import Counter

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

BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_CORPUS_PATH = BASE_DIR / "data" / "raw" / "corpus.jsonl"
EXPANDED_CORPUS_PATH = BASE_DIR / "data" / "experimental" / "corpus_expanded_10k.jsonl"
TEST_CLAIMS_PATH = BASE_DIR / "experiments" / "new_claims.jsonl"
VAL_CLAIMS_PATH = BASE_DIR / "data" / "processed" / "val_internal.jsonl"
OUTPUT_COMPARISON_PATH = BASE_DIR / "experiments" / "corpus_expansion_comparison.json"

# Fixed Configuration C parameters
TOP_K_DOCS = 10
TOP_K_SENTENCES = 4
DOC_SCORE_WEIGHT = 0.35
MAX_PER_DOC = 2

def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = CANONICAL_LABELS
) -> Dict[str, Any]:
    """Compute overall accuracy, macro F1, and per-class precision/recall/F1."""
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

def compute_retrieval_metrics(
    eval_claims: List[Dict[str, Any]],
    pipeline: ClaimVerificationPipeline,
) -> Dict[str, Any]:
    """
    Compute document recall@5, @10, @20, sentence recall, MRR, nDCG on claims with gold evidence.
    """
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

        # Retrieve top 20 candidate docs for evaluation
        docs_20 = pipeline.retriever.retrieve_documents(claim_text, top_k=20)
        retrieved_ids = [d["doc_id"] for d in docs_20]

        # Doc Recall@5, @10, @20
        hit5 = any(did in gold_doc_ids for did in retrieved_ids[:5])
        hit10 = any(did in gold_doc_ids for did in retrieved_ids[:10])
        hit20 = any(did in gold_doc_ids for did in retrieved_ids[:20])

        if hit5:
            doc_r5_hits += 1
        if hit10:
            doc_r10_hits += 1
        if hit20:
            doc_r20_hits += 1

        # MRR@10 and MRR@20
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

        # nDCG@10 (binary relevance)
        dcg10 = sum(
            1.0 / math.log2(rank + 1)
            for rank, did in enumerate(retrieved_ids[:10], start=1)
            if did in gold_doc_ids
        )
        ideal_count = min(len(gold_doc_ids), 10)
        idcg10 = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1)) if ideal_count > 0 else 1.0
        ndcg_10_scores.append(dcg10 / idcg10 if idcg10 > 0 else 0.0)

        # Sentence recall using top-10 docs & top-4 sentences
        docs_10 = docs_20[:TOP_K_DOCS]
        sentences_top = pipeline.retriever.retrieve_sentences(
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

def run_evaluation_on_corpus(
    corpus_name: str,
    corpus_path: Path,
    nli_model: NLIEntailmentModel,
    test_claims: List[Dict[str, Any]],
    val_claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build retriever on corpus and run complete evaluation."""
    print(f"\n=======================================================")
    print(f"Evaluating: {corpus_name} ({corpus_path})")
    print(f"=======================================================")

    t0 = time.time()
    corpus = load_corpus(corpus_path)
    t_load = time.time() - t0
    print(f"Loaded {len(corpus)} documents in {t_load:.2f}s.")

    t0 = time.time()
    retriever = BM25Retriever(corpus)
    t_index = time.time() - t0
    print(f"Built BM25 lexical index in {t_index:.2f}s.")

    pipeline = ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=nli_model,
        top_k_docs=TOP_K_DOCS,
        top_k_sentences=TOP_K_SENTENCES,
        entail_threshold=Config.ENTAIL_THRESHOLD,
        contradict_threshold=Config.CONTRADICT_THRESHOLD,
    )

    # 1. Evaluate 24 Test Claims
    print("--- Evaluating Held-Out Diagnostic Test Set (N=24) ---")
    test_preds = []
    latencies = []
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
        latencies.append(dt_ms)

        pred_label = res["predicted_label"]
        y_pred_test.append(pred_label)

        # Check candidate docs
        cand_ids = res.get("retrieved_candidate_doc_ids", [])
        evidence_sents = res.get("predicted_evidence_sentences", [])

        # Check if gold document exists in this corpus
        gold_did = item.get("evidence_doc_id")
        gold_in_corpus = gold_did in corpus if gold_did is not None else False

        test_preds.append({
            "claim_id": cid,
            "claim": claim_text,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "confidence": round(float(res.get("confidence", 0.0)), 4),
            "is_correct": pred_label == gold_label,
            "evidence_doc_id": gold_did,
            "gold_doc_present_in_corpus": gold_in_corpus,
            "retrieved_candidate_doc_ids": cand_ids,
            "predicted_evidence_sentences": evidence_sents,
            "latency_ms": round(dt_ms, 2)
        })

    test_metrics = compute_metrics(y_true_test, y_pred_test)
    test_retrieval = compute_retrieval_metrics(test_claims, pipeline)

    # Answerable subset metrics on test set
    ans_true = [t for t, item in zip(y_true_test, test_claims) if item.get("gold_doc_ids")]
    ans_pred = [p for p, item in zip(y_pred_test, test_claims) if item.get("gold_doc_ids")]
    ans_metrics = compute_metrics(ans_true, ans_pred) if ans_true else {}

    # Unanswerable / absent subset metrics
    unans_true = [t for t, item in zip(y_true_test, test_claims) if not item.get("gold_doc_ids")]
    unans_pred = [p for p, item in zip(y_pred_test, test_claims) if not item.get("gold_doc_ids")]
    unans_metrics = compute_metrics(unans_true, unans_pred) if unans_true else {}

    mean_test_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}% ({test_metrics['correct']}/{test_metrics['total']})")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    if ans_metrics:
        print(f"Answerable Accuracy: {ans_metrics['accuracy']*100:.2f}% ({ans_metrics['correct']}/{ans_metrics['total']})")
    print(f"Mean Latency: {mean_test_latency} ms")

    # 2. Evaluate 161 Validation Claims
    print("\n--- Evaluating Internal Validation Set (N=161) ---")
    val_preds = []
    y_true_val = []
    y_pred_val = []
    val_latencies = []

    for item in val_claims:
        cid = item["id"]
        claim_text = item["claim"]
        gold_label = item["gold_label"]
        y_true_val.append(gold_label)

        t_start = time.time()
        res = pipeline.verify_claim(claim_text, claim_id=cid)
        dt_ms = (time.time() - t_start) * 1000.0
        val_latencies.append(dt_ms)

        pred_label = res["predicted_label"]
        y_pred_val.append(pred_label)

        val_preds.append({
            "claim_id": cid,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "is_correct": pred_label == gold_label,
            "latency_ms": round(dt_ms, 2)
        })

    val_metrics = compute_metrics(y_true_val, y_pred_val)
    val_retrieval = compute_retrieval_metrics(val_claims, pipeline)
    mean_val_latency = round(sum(val_latencies) / len(val_latencies), 2) if val_latencies else 0.0

    print(f"Val Accuracy: {val_metrics['accuracy']*100:.2f}% ({val_metrics['correct']}/{val_metrics['total']})")
    print(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"Val Doc Recall@10: {val_retrieval.get('doc_recall@10', 0.0)*100:.2f}%")

    return {
        "corpus_name": corpus_name,
        "corpus_path": str(corpus_path),
        "corpus_document_count": len(corpus),
        "index_build_time_seconds": round(t_index, 2),
        "mean_query_latency_ms": mean_test_latency,
        "mean_val_latency_ms": mean_val_latency,
        "test_metrics_n24": {
            "overall_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "correct": test_metrics["correct"],
            "total": test_metrics["total"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "answerable_subset": ans_metrics,
            "unanswerable_subset": unans_metrics,
            "retrieval": test_retrieval,
        },
        "val_metrics_n161": {
            "overall_accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "correct": val_metrics["correct"],
            "total": val_metrics["total"],
            "per_class": val_metrics["per_class"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "retrieval": val_retrieval,
        },
        "test_predictions": test_preds,
    }

def main():
    print("=" * 70)
    print("Phase 4B: Corpus Expansion Comparison Experiment")
    print("=" * 70)

    # 1. Load fine-tuned checkpoint
    model_path = Config.MODELS_DIR / "finetuned_nli"
    if not model_path.exists():
        print(f"[Error] Checkpoint not found at {model_path}")
        sys.exit(1)

    print(f"Loading fine-tuned NLI model from: {model_path}")
    nli_model = NLIEntailmentModel(
        model_name=str(model_path),
        device=Config.DEVICE
    )

    # 2. Load evaluation test sets
    with open(TEST_CLAIMS_PATH, "r", encoding="utf-8") as f:
        raw_test = [json.loads(line) for line in f if line.strip()]

    # Format test claims
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

    # Format val claims
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

    # 3. Run Configuration A (Original 5k corpus)
    res_a = run_evaluation_on_corpus(
        corpus_name="Original Corpus (5,183 abstracts)",
        corpus_path=ORIGINAL_CORPUS_PATH,
        nli_model=nli_model,
        test_claims=test_claims,
        val_claims=val_claims,
    )

    # 4. Run Configuration B (Expanded 10k corpus)
    if not EXPANDED_CORPUS_PATH.exists():
        print(f"[Error] Expanded corpus not found at {EXPANDED_CORPUS_PATH}")
        sys.exit(1)

    res_b = run_evaluation_on_corpus(
        corpus_name="Expanded Corpus (~10,000 abstracts)",
        corpus_path=EXPANDED_CORPUS_PATH,
        nli_model=nli_model,
        test_claims=test_claims,
        val_claims=val_claims,
    )

    # 5. Compare per-claim differences
    print("\n=======================================================")
    print("Per-Claim Diagnostic Analysis (Test Set N=24)")
    print("=======================================================")
    changes = []
    for pa, pb in zip(res_a["test_predictions"], res_b["test_predictions"]):
        cid = pa["claim_id"]
        gold = pa["gold_label"]
        la = pa["predicted_label"]
        lb = pb["predicted_label"]
        c_flag = "CHANGED" if la != lb else "SAME"
        if la != lb:
            changes.append({
                "claim_id": cid,
                "claim": pa["claim"],
                "gold_label": gold,
                "pred_a": la,
                "pred_b": lb,
                "became_correct": lb == gold and la != gold,
                "became_incorrect": la == gold and lb != gold,
                "gold_doc_in_b": pb.get("gold_doc_present_in_corpus", False)
            })
        print(f"Claim {cid:2d} | Gold: {gold:<12} | A: {la:<12} | B: {lb:<12} | {c_flag}")

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_configuration": {
            "name": "Configuration C",
            "top_k_docs": TOP_K_DOCS,
            "top_k_sentences": TOP_K_SENTENCES,
            "doc_score_weight": DOC_SCORE_WEIGHT,
            "max_per_doc": MAX_PER_DOC,
        },
        "model_checkpoint": str(model_path),
        "configuration_a_original": res_a,
        "configuration_b_expanded": res_b,
        "per_claim_changes": changes,
    }

    OUTPUT_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_COMPARISON_PATH, "w", encoding="utf-8") as f_out:
        json.dump(output_data, f_out, indent=2)
    print(f"\nSaved detailed comparative results to: {OUTPUT_COMPARISON_PATH}")

if __name__ == "__main__":
    main()
