"""
Phase 2: Scientific Entity Grounding Filter Benchmark
Reproducible evaluation on SciFact validation (N=161) and diagnostic test (N=24).
Compares:
1. Baseline: Existing production pipeline (BM25 Config C + Fine-tuned DeBERTa)
2. Grounded Direct: Entity Grounding Filter applied to candidate evidence sentences
3. Grounded Pool: Entity Grounding Filter applied to candidate pool (top-8) selecting top-4 grounded sentences
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from backend.data_loader import (
    load_corpus,
    derive_gold_claim_label,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.config import Config
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from scripts.experimental.entity_grounding import ScientificEntityGroundingFilter


def load_datasets():
    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    corpus = load_corpus(corpus_path)

    val_path = Config.PROCESSED_DATA_DIR / "val_internal.jsonl"
    val_claims = [json.loads(line) for line in open(val_path, encoding="utf-8") if line.strip()]

    test_path = Config.BASE_DIR / "experiments" / "new_claims.jsonl"
    test_claims = [json.loads(line) for line in open(test_path, encoding="utf-8") if line.strip()]

    return corpus, val_claims, test_claims


def run_benchmark():
    print("=" * 70)
    print("ResearchMap Phase 2: Scientific Entity Grounding Filter Benchmark")
    print("=" * 70)

    corpus, val_claims, test_claims = load_datasets()
    print(f"Loaded corpus: {len(corpus):,} abstracts")
    print(f"Loaded validation claims: {len(val_claims)}")
    print(f"Loaded diagnostic claims: {len(test_claims)}")

    retriever = BM25Retriever(corpus)
    print("Initialized BM25Retriever (Config C: top_k_docs=10, top_k_sents=4, doc_weight=0.35)")

    model_path = Config.MODELS_DIR / "finetuned_nli"
    if not model_path.exists():
        model_path_str = Config.NLI_MODEL_NAME
        print(f"Finetuned model not found at {model_path}, using base model: {model_path_str}")
    else:
        model_path_str = str(model_path)
        print(f"Loaded fine-tuned model checkpoint from: {model_path_str}")

    nli_model = NLIEntailmentModel(model_name=model_path_str, device="cpu")
    grounding_filter = ScientificEntityGroundingFilter()

    def run_split(claims: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
        print(f"\nEvaluating {split_name} ({len(claims)} claims)...")
        golds = []
        baseline_preds = []
        grounded_direct_preds = []
        grounded_pool_preds = []

        baseline_latencies = []
        grounded_latencies = []

        total_candidates_direct = 0
        rejected_candidates_direct = 0
        total_candidates_pool = 0
        rejected_candidates_pool = 0

        claim_eval_records = []
        rejection_audit_log = []

        for idx, c in enumerate(claims):
            t_claim_start = time.perf_counter()
            cid = c.get("id")
            claim_text = c["claim"]

            if "gold_label" in c:
                gold = c["gold_label"]
            elif "evidence" in c:
                gold = derive_gold_claim_label(c["evidence"])
            else:
                gold = LABEL_NEI
            golds.append(gold)

            # Step 1: Document retrieval
            candidate_docs = retriever.retrieve_documents(claim_text, top_k=Config.BM25_TOP_K_DOCS)

            # Baseline candidate sentences (top 4)
            candidate_sentences_base = retriever.retrieve_sentences(
                claim_text, candidate_docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES
            )

            # Pool candidate sentences (top 8)
            candidate_sentences_pool = retriever.retrieve_sentences(
                claim_text, candidate_docs, top_k_sentences=8
            )

            # Gather unique sentences to avoid redundant NLI inference
            unique_sents = {}
            for s in candidate_sentences_base + candidate_sentences_pool:
                k = (s["doc_id"], s["sentence_idx"])
                if k not in unique_sents:
                    unique_sents[k] = s

            # NLI cross-encoder inference
            pairs = [(s["text"], claim_text) for s in unique_sents.values()]
            t_nli_start = time.perf_counter()
            nli_outputs = nli_model.predict_batch(pairs)
            nli_map = {k: out for k, out in zip(unique_sents.keys(), nli_outputs)}

            t_baseline_end = time.perf_counter()
            baseline_latencies.append((t_baseline_end - t_claim_start) * 1000.0)

            def aggregate_sentences(sents: List[Dict[str, Any]]) -> Tuple[str, float, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
                best_support = None
                best_contradict = None
                details = []

                for s in sents:
                    k = (s["doc_id"], s["sentence_idx"])
                    nli_out = nli_map[k]
                    doc_data = corpus.get(s["doc_id"], {})
                    det = {
                        "doc_id": s["doc_id"],
                        "sentence_idx": s["sentence_idx"],
                        "text": s["text"],
                        "title": doc_data.get("title", ""),
                        "bm25_score": s.get("bm25_score", 0.0),
                        "probabilities": nli_out["probabilities"],
                    }
                    if "grounding" in s:
                        det["grounding"] = s["grounding"]
                    details.append(det)

                    p_sup = nli_out["probabilities"].get(LABEL_SUPPORTS, 0.0)
                    p_con = nli_out["probabilities"].get(LABEL_CONTRADICTS, 0.0)

                    if p_sup >= Config.ENTAIL_THRESHOLD:
                        if best_support is None or p_sup > best_support["probabilities"][LABEL_SUPPORTS]:
                            best_support = det

                    if p_con >= Config.CONTRADICT_THRESHOLD:
                        if best_contradict is None or p_con > best_contradict["probabilities"][LABEL_CONTRADICTS]:
                            best_contradict = det

                if best_support and best_contradict:
                    if best_support["probabilities"][LABEL_SUPPORTS] >= best_contradict["probabilities"][LABEL_CONTRADICTS]:
                        return LABEL_SUPPORTS, best_support["probabilities"][LABEL_SUPPORTS], best_support, details
                    else:
                        return LABEL_CONTRADICTS, best_contradict["probabilities"][LABEL_CONTRADICTS], best_contradict, details
                elif best_support:
                    return LABEL_SUPPORTS, best_support["probabilities"][LABEL_SUPPORTS], best_support, details
                elif best_contradict:
                    return LABEL_CONTRADICTS, best_contradict["probabilities"][LABEL_CONTRADICTS], best_contradict, details
                else:
                    return LABEL_NEI, 0.0, None, details

            # 1. Baseline Decision
            base_label, base_conf, base_best_ev, base_details = aggregate_sentences(candidate_sentences_base)
            baseline_preds.append(base_label)

            # 2. Grounded Direct
            t_ground_start = time.perf_counter()
            accepted_direct, rejected_direct = grounding_filter.filter_candidate_sentences(
                claim_text, candidate_sentences_base, corpus
            )
            total_candidates_direct += len(candidate_sentences_base)
            rejected_candidates_direct += len(rejected_direct)

            for rej in rejected_direct:
                rejection_audit_log.append({
                    "split": split_name,
                    "claim_id": cid,
                    "claim": claim_text,
                    "doc_id": rej["doc_id"],
                    "title": corpus.get(rej["doc_id"], {}).get("title", ""),
                    "text": rej["text"],
                    "grounding": rej["grounding"],
                })

            grounded_dir_label, grounded_dir_conf, grounded_dir_best_ev, grounded_dir_details = aggregate_sentences(
                accepted_direct
            )
            grounded_direct_preds.append(grounded_dir_label)

            # 3. Grounded Pool
            accepted_pool, rejected_pool = grounding_filter.filter_candidate_sentences(
                claim_text, candidate_sentences_pool, corpus
            )
            total_candidates_pool += len(candidate_sentences_pool)
            rejected_candidates_pool += len(rejected_pool)

            top_accepted_pool = accepted_pool[:Config.BM25_TOP_K_SENTENCES]
            grounded_pool_label, grounded_pool_conf, grounded_pool_best_ev, grounded_pool_details = aggregate_sentences(
                top_accepted_pool
            )
            grounded_pool_preds.append(grounded_pool_label)

            t_ground_end = time.perf_counter()
            grounded_latencies.append((t_ground_end - t_claim_start) * 1000.0)

            # Record per-claim comparison
            claim_eval_records.append({
                "claim_id": cid,
                "claim": claim_text,
                "gold_label": gold,
                "baseline": {
                    "prediction": base_label,
                    "confidence": float(base_conf),
                    "best_evidence": base_best_ev,
                },
                "grounded_direct": {
                    "prediction": grounded_dir_label,
                    "confidence": float(grounded_dir_conf),
                    "best_evidence": grounded_dir_best_ev,
                    "accepted_count": len(accepted_direct),
                    "rejected_count": len(rejected_direct),
                },
                "grounded_pool": {
                    "prediction": grounded_pool_label,
                    "confidence": float(grounded_pool_conf),
                    "best_evidence": grounded_pool_best_ev,
                    "accepted_count": len(top_accepted_pool),
                },
                "disagreement": (base_label != grounded_dir_label) or (base_label != grounded_pool_label),
            })

            if (idx + 1) % 40 == 0 or (idx + 1) == len(claims):
                print(f"Processed {idx + 1}/{len(claims)} claims...")

        def calculate_metrics(preds: List[str], label_name: str) -> Dict[str, Any]:
            acc = float(accuracy_score(golds, preds))
            macro_f1 = float(f1_score(golds, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0))
            p, r, f1, _ = precision_recall_fscore_support(golds, preds, labels=CANONICAL_LABELS, zero_division=0)
            cm = confusion_matrix(golds, preds, labels=CANONICAL_LABELS).tolist()

            false_con = sum(1 for g, p in zip(golds, preds) if g != LABEL_CONTRADICTS and p == LABEL_CONTRADICTS)
            false_sup = sum(1 for g, p in zip(golds, preds) if g != LABEL_SUPPORTS and p == LABEL_SUPPORTS)

            return {
                "system": label_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "per_class": {
                    c: {
                        "precision": float(p[i]),
                        "recall": float(r[i]),
                        "f1": float(f1[i]),
                    }
                    for i, c in enumerate(CANONICAL_LABELS)
                },
                "confusion_matrix": cm,
                "false_contradictions": false_con,
                "false_supports": false_sup,
            }

        base_metrics = calculate_metrics(baseline_preds, "Baseline (Production Config C)")
        direct_metrics = calculate_metrics(grounded_direct_preds, "Grounded Direct Filter")
        pool_metrics = calculate_metrics(grounded_pool_preds, "Grounded Candidate Pool Filter")

        dir_rejection_rate = (rejected_candidates_direct / total_candidates_direct) if total_candidates_direct > 0 else 0.0
        pool_rejection_rate = (rejected_candidates_pool / total_candidates_pool) if total_candidates_pool > 0 else 0.0

        latency_summary = {
            "baseline": {
                "mean_ms": float(np.mean(baseline_latencies)),
                "median_ms": float(np.median(baseline_latencies)),
                "p95_ms": float(np.percentile(baseline_latencies, 95)),
            },
            "grounded": {
                "mean_ms": float(np.mean(grounded_latencies)),
                "median_ms": float(np.median(grounded_latencies)),
                "p95_ms": float(np.percentile(grounded_latencies, 95)),
            },
        }

        # Identify corrected vs worsened cases
        corrected_cases = []
        worsened_cases = []
        for rec in claim_eval_records:
            gold = rec["gold_label"]
            b_pred = rec["baseline"]["prediction"]
            g_pred = rec["grounded_direct"]["prediction"]

            if b_pred != gold and g_pred == gold:
                corrected_cases.append({
                    "claim_id": rec["claim_id"],
                    "claim": rec["claim"],
                    "gold": gold,
                    "baseline_pred": b_pred,
                    "grounded_pred": g_pred,
                    "baseline_ev": rec["baseline"]["best_evidence"],
                    "grounded_ev": rec["grounded_direct"]["best_evidence"],
                })
            elif b_pred == gold and g_pred != gold:
                worsened_cases.append({
                    "claim_id": rec["claim_id"],
                    "claim": rec["claim"],
                    "gold": gold,
                    "baseline_pred": b_pred,
                    "grounded_pred": g_pred,
                    "baseline_ev": rec["baseline"]["best_evidence"],
                    "grounded_ev": rec["grounded_direct"]["best_evidence"],
                })

        print(f"\n--- {split_name} Benchmark Summary ---")
        print(f"Direct Rejection Rate: {rejected_candidates_direct}/{total_candidates_direct} ({dir_rejection_rate*100:.2f}%)")
        print(f"Corrected Cases: {len(corrected_cases)} | Worsened Cases: {len(worsened_cases)}")
        for m in [base_metrics, direct_metrics, pool_metrics]:
            print(f"{m['system']:<35} | Acc: {m['accuracy']*100:.2f}% | F1: {m['macro_f1']:.4f} | False CON: {m['false_contradictions']} | False SUP: {m['false_supports']}")

        return {
            "split_name": split_name,
            "total_claims": len(claims),
            "metrics": {
                "baseline": base_metrics,
                "grounded_direct": direct_metrics,
                "grounded_pool": pool_metrics,
            },
            "rejection_statistics": {
                "direct": {
                    "total_candidates": total_candidates_direct,
                    "rejected_candidates": rejected_candidates_direct,
                    "rejection_rate": float(dir_rejection_rate),
                },
                "pool": {
                    "total_candidates": total_candidates_pool,
                    "rejected_candidates": rejected_candidates_pool,
                    "rejection_rate": float(pool_rejection_rate),
                },
            },
            "latency": latency_summary,
            "corrected_cases": corrected_cases,
            "worsened_cases": worsened_cases,
            "claim_records": claim_eval_records,
            "rejection_audit_sample": rejection_audit_log[:50],
        }

    val_benchmark = run_split(val_claims, "Validation Internal (N=161)")
    test_benchmark = run_split(test_claims, "Diagnostic New Claims (N=24)")

    full_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_used": model_path_str,
        "corpus_evaluated": str(Config.RAW_DATA_DIR / "corpus.jsonl"),
        "validation_benchmark": val_benchmark,
        "diagnostic_benchmark": test_benchmark,
    }

    output_path = Config.BASE_DIR / "experiments" / "phase2_entity_grounding_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nSaved full benchmark results to: {output_path}")
    return full_results


if __name__ == "__main__":
    run_benchmark()
