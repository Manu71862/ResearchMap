"""
Phase 1: Evidence Aggregation Improvement Benchmark Runner
Evaluates four aggregation strategies:
1. Baseline: Existing Max-Probability Single-Sentence Thresholding
2. Confidence-Weighted Voting
3. Relevance-Weighted Voting
4. Consensus-Based Aggregation

Evaluates on:
- Validation Set (data/processed/val_internal.jsonl, N=161)
- Diagnostic Test Set (experiments/new_claims.jsonl, N=24)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import load_corpus, load_claims, LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI, CANONICAL_LABELS
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from scripts.experimental.aggregation_strategies import (
    BaselineAggregation,
    ConfidenceWeightedVotingAggregation,
    RelevanceWeightedVotingAggregation,
    ConsensusBasedAggregation,
)

CORPUS_PATH = Config.RAW_DATA_DIR / "corpus.jsonl"
VAL_CLAIMS_PATH = Config.PROCESSED_DATA_DIR / "val_internal.jsonl"
TEST_CLAIMS_PATH = BASE_DIR / "experiments" / "new_claims.jsonl"
OUTPUT_JSON_PATH = BASE_DIR / "experiments" / "phase1_aggregation_results.json"

def compute_detailed_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / n

    cm = {l1: {l2: 0 for l2 in CANONICAL_LABELS} for l1 in CANONICAL_LABELS}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    f1_list = []
    for l in CANONICAL_LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[l] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == l)
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list)

    # False contradiction: Actual != CONTRADICTS but predicted CONTRADICTS
    false_contradictions = sum(1 for t, p in zip(y_true, y_pred) if t != LABEL_CONTRADICTS and p == LABEL_CONTRADICTS)
    # False support: Actual != SUPPORTS but predicted SUPPORTS
    false_supports = sum(1 for t, p in zip(y_true, y_pred) if t != LABEL_SUPPORTS and p == LABEL_SUPPORTS)

    return {
        "total_claims": n,
        "correct": correct,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": [[cm[l1][l2] for l2 in CANONICAL_LABELS] for l1 in CANONICAL_LABELS],
        "false_contradictions": false_contradictions,
        "false_supports": false_supports
    }

def main():
    print("=" * 70)
    print("ResearchMap Phase 1: Evidence Aggregation Improvement Benchmark")
    print("=" * 70)

    # Load resources
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus):,} corpus abstracts.")
    model_path = Config.MODELS_DIR / "finetuned_nli"
    nli_model = NLIEntailmentModel(model_name=str(model_path), device="cpu")
    bm25 = BM25Retriever(corpus)

    # Load datasets
    val_claims = load_claims(VAL_CLAIMS_PATH)
    with open(TEST_CLAIMS_PATH, "r", encoding="utf-8") as f:
        test_claims = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(val_claims)} validation claims and {len(test_claims)} diagnostic test claims.")

    # Instantiate strategies
    strategies = {
        "1_Baseline": BaselineAggregation(entail_thresh=0.50, contradict_thresh=0.50),
        "2_ConfidenceWeighted": ConfidenceWeightedVotingAggregation(entail_thresh=0.45, contradict_thresh=0.45, conflict_margin=0.15),
        "3_RelevanceWeighted": RelevanceWeightedVotingAggregation(entail_thresh=0.45, contradict_thresh=0.45, conflict_margin=0.15),
        "4_ConsensusBased": ConsensusBasedAggregation(
            base_thresh=0.50,
            high_conf_single_thresh=0.75,
            neutral_consensus_thresh=0.70,
            conflict_tolerance=0.15
        )
    }

    # Precompute retrieval and NLI inference pairs for perfect consistency
    print("\nPrecomputing retrieval and candidate sentence NLI distributions...")

    def evaluate_dataset(claims_data, is_test: bool = False):
        all_eval_details = []
        for c in claims_data:
            claim_text = c["claim"]
            gold_label = c["gold_label"]
            cid = c["id"]

            t0 = time.time()
            cand_docs = bm25.retrieve_documents(claim_text, top_k=Config.BM25_TOP_K_DOCS)
            cand_sents = bm25.retrieve_sentences(claim_text, cand_docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES)

            pairs = [(s["text"], claim_text) for s in cand_sents]
            nli_outs = nli_model.predict_batch(pairs) if pairs else []

            ev_details = []
            for s, out in zip(cand_sents, nli_outs):
                ev_details.append({
                    "doc_id": s["doc_id"],
                    "sentence_idx": s["sentence_idx"],
                    "text": s["text"],
                    "bm25_score": s.get("bm25_score", 1.0),
                    "confidence": out["confidence"],
                    "probabilities": out["probabilities"],
                    "entailment_label": out["prediction"]
                })
            base_lat = (time.time() - t0) * 1000.0

            all_eval_details.append({
                "id": cid,
                "claim": claim_text,
                "gold_label": gold_label,
                "evidence_details": ev_details,
                "base_latency_ms": base_lat
            })
        return all_eval_details

    val_eval_data = evaluate_dataset(val_claims, is_test=False)
    test_eval_data = evaluate_dataset(test_claims, is_test=True)

    strategy_results = {}

    for strat_key, strat_impl in strategies.items():
        print(f"\nEvaluating: {strat_key} ({strat_impl.__class__.__name__})...")

        # Validation
        val_preds = []
        val_latencies = []
        val_y_true = []
        val_y_pred = []

        for item in val_eval_data:
            gold = item["gold_label"]
            val_y_true.append(gold)
            t_start = time.time()
            pred_lbl, conf, sents = strat_impl.aggregate(item["evidence_details"], item["claim"])
            dt = (time.time() - t_start) * 1000.0 + item["base_latency_ms"]
            val_latencies.append(dt)
            val_y_pred.append(pred_lbl)

            val_preds.append({
                "id": item["id"],
                "claim": item["claim"],
                "gold": gold,
                "pred": pred_lbl,
                "confidence": round(conf, 4),
                "is_correct": pred_lbl == gold
            })

        val_metrics = compute_detailed_metrics(val_y_true, val_y_pred)
        val_metrics["mean_latency_ms"] = round(float(np.mean(val_latencies)), 2)
        val_metrics["p95_latency_ms"] = round(float(np.percentile(val_latencies, 95)), 2)

        # Test
        test_preds = []
        test_latencies = []
        test_y_true = []
        test_y_pred = []

        for item in test_eval_data:
            gold = item["gold_label"]
            test_y_true.append(gold)
            t_start = time.time()
            pred_lbl, conf, sents = strat_impl.aggregate(item["evidence_details"], item["claim"])
            dt = (time.time() - t_start) * 1000.0 + item["base_latency_ms"]
            test_latencies.append(dt)
            test_y_pred.append(pred_lbl)

            test_preds.append({
                "id": item["id"],
                "claim": item["claim"],
                "gold": gold,
                "pred": pred_lbl,
                "confidence": round(conf, 4),
                "is_correct": pred_lbl == gold
            })

        test_metrics = compute_detailed_metrics(test_y_true, test_y_pred)
        test_metrics["mean_latency_ms"] = round(float(np.mean(test_latencies)), 2)
        test_metrics["p95_latency_ms"] = round(float(np.percentile(test_latencies, 95)), 2)

        print(f"  Val Accuracy:  {val_metrics['accuracy']*100:.2f}% | Macro-F1: {val_metrics['macro_f1']:.4f} | False CON: {val_metrics['false_contradictions']} | False SUP: {val_metrics['false_supports']}")
        print(f"  Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']:.4f} | False CON: {test_metrics['false_contradictions']} | False SUP: {test_metrics['false_supports']}")

        strategy_results[strat_key] = {
            "strategy_name": strat_impl.__class__.__name__,
            "validation_metrics": val_metrics,
            "diagnostic_test_metrics": test_metrics,
            "validation_predictions": val_preds,
            "test_predictions": test_preds
        }

    # Find representative claims with disagreement
    disagreements = []
    base_preds = strategy_results["1_Baseline"]["validation_predictions"]
    conf_preds = strategy_results["2_ConfidenceWeighted"]["validation_predictions"]
    rel_preds = strategy_results["3_RelevanceWeighted"]["validation_predictions"]
    cons_preds = strategy_results["4_ConsensusBased"]["validation_predictions"]

    for b, c, r, cs in zip(base_preds, conf_preds, rel_preds, cons_preds):
        preds_set = {b["pred"], c["pred"], r["pred"], cs["pred"]}
        if len(preds_set) > 1:
            # Look up evidence details from val_eval_data
            eval_item = next(it for it in val_eval_data if it["id"] == b["id"])
            disagreements.append({
                "claim_id": b["id"],
                "claim": b["claim"],
                "gold_label": b["gold"],
                "baseline_pred": b["pred"],
                "confidence_weighted_pred": c["pred"],
                "relevance_weighted_pred": r["pred"],
                "consensus_pred": cs["pred"],
                "evidence_candidates": [
                    {
                        "doc_id": d["doc_id"],
                        "sentence_idx": d["sentence_idx"],
                        "probabilities": d["probabilities"],
                        "text": d["text"][:120]
                    }
                    for d in eval_item["evidence_details"]
                ]
            })

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_evaluated": str(CORPUS_PATH),
        "nli_model": str(model_path),
        "total_validation_claims": len(val_claims),
        "total_diagnostic_claims": len(test_claims),
        "strategies": strategy_results,
        "disagreement_cases_count": len(disagreements),
        "disagreement_cases": disagreements[:15]
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved complete Phase 1 aggregation results to: {OUTPUT_JSON_PATH}")
    print(f"Total disagreement cases found: {len(disagreements)}")

if __name__ == "__main__":
    main()
