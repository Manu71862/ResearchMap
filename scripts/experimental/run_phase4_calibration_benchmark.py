"""
ResearchMap Phase 4: Systematic NLI Calibration & Benchmark Harness
Evaluates:
  1. Baseline NLI (models/finetuned_nli)
  2. Augmented NLI (models/experimental_augmented_nli)
  3. Calibrated Balanced NLI (models/experimental_calibrated_nli)
Across:
  - Validation Set (N=161)
  - Diagnostic Set (N=24)
  - Challenge Set (N=60)

Threshold search rule: Strictly optimized on Validation Set (val_internal.jsonl) ONLY.
Evaluates both default thresholds (0.50, 0.50) and validation-tuned thresholds.
Computes multi-class Brier scores, confusion matrices, per-class metrics,
latency, and detailed regression audits.
Outputs results to 'experiments/phase4_calibration_results.json'.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import (
    load_corpus,
    load_claims,
    CANONICAL_LABELS,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
)
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel


def multiclass_brier_score(golds_idx: List[int], prob_matrix: np.ndarray) -> float:
    """Mean squared difference between predicted probability vector and 1-hot ground truth."""
    n_samples = len(golds_idx)
    num_classes = prob_matrix.shape[1]
    one_hot = np.zeros((n_samples, num_classes))
    for i, idx in enumerate(golds_idx):
        one_hot[i, idx] = 1.0
    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def evaluate_claim_evidence_selection(
    candidate_sentences: List[Dict[str, Any]],
    nli_outputs: List[Dict[str, Any]],
    tau_sup: float,
    tau_con: float,
) -> Tuple[str, Dict[str, Any], List[float]]:
    """
    Simulates the claim verification pipeline decision logic given thresholds tau_sup and tau_con.
    Returns: (predicted_label, best_evidence_dict, claim_probability_vector [CON, SUP, NEI])
    """
    best_sup = None
    best_con = None

    for s, out in zip(candidate_sentences, nli_outputs):
        p_sup = out["probabilities"].get(LABEL_SUPPORTS, 0.0)
        p_con = out["probabilities"].get(LABEL_CONTRADICTS, 0.0)

        if p_sup >= tau_sup:
            if best_sup is None or p_sup > best_sup["p"]:
                best_sup = {
                    "p": p_sup,
                    "doc_id": s["doc_id"],
                    "sentence_idx": s.get("sentence_idx"),
                    "text": s["text"],
                    "probs": out["probabilities"],
                }

        if p_con >= tau_con:
            if best_con is None or p_con > best_con["p"]:
                best_con = {
                    "p": p_con,
                    "doc_id": s["doc_id"],
                    "sentence_idx": s.get("sentence_idx"),
                    "text": s["text"],
                    "probs": out["probabilities"],
                }

    if best_sup and best_con:
        if best_sup["p"] >= best_con["p"]:
            pred = LABEL_SUPPORTS
            best_ev = best_sup
        else:
            pred = LABEL_CONTRADICTS
            best_ev = best_con
    elif best_sup:
        pred = LABEL_SUPPORTS
        best_ev = best_sup
    elif best_con:
        pred = LABEL_CONTRADICTS
        best_ev = best_con
    else:
        pred = LABEL_NEI
        # Best neutral evidence is sentence with highest NEI probability
        best_ev = max(
            [
                {
                    "p": o["probabilities"].get(LABEL_NEI, 0.0),
                    "doc_id": s["doc_id"],
                    "sentence_idx": s.get("sentence_idx"),
                    "text": s["text"],
                    "probs": o["probabilities"],
                }
                for s, o in zip(candidate_sentences, nli_outputs)
            ],
            key=lambda x: x["p"],
            default={
                "p": 1.0,
                "doc_id": None,
                "sentence_idx": None,
                "text": "",
                "probs": {LABEL_CONTRADICTS: 0.0, LABEL_SUPPORTS: 0.0, LABEL_NEI: 1.0},
            },
        )

    probs = [
        best_ev["probs"].get(LABEL_CONTRADICTS, 0.0),
        best_ev["probs"].get(LABEL_SUPPORTS, 0.0),
        best_ev["probs"].get(LABEL_NEI, 0.0),
    ]

    return pred, best_ev, probs


def run_comprehensive_benchmark():
    corpus = load_corpus(Config.RAW_DATA_DIR / "corpus.jsonl")
    retriever = BM25Retriever(corpus)

    val_claims = load_claims(Config.PROCESSED_DATA_DIR / "val_internal.jsonl")
    test_claims = [json.loads(l) for l in open(Config.BASE_DIR / "experiments" / "new_claims.jsonl", encoding="utf-8") if l.strip()]
    challenge_pairs = [json.loads(l) for l in open(Config.BASE_DIR / "data" / "experimental" / "challenge_set.jsonl", encoding="utf-8") if l.strip()]

    label2idx = {LABEL_CONTRADICTS: 0, LABEL_SUPPORTS: 1, LABEL_NEI: 2}
    val_golds = [c["gold_label"] for c in val_claims]
    val_golds_idx = [label2idx[g] for g in val_golds]

    diag_golds = [c["gold_label"] for c in test_claims]
    diag_golds_idx = [label2idx[g] for g in diag_golds]

    ch_pairs_text = [(p["evidence_text"], p["claim"]) for p in challenge_pairs]
    ch_golds_idx = [p["gold_label_idx"] for p in challenge_pairs]
    ch_golds_str = [p["gold_label"] for p in challenge_pairs]
    ch_categories = [p["category"] for p in challenge_pairs]

    # Pre-retrieve sentences for validation and diagnostic claims to ensure identical candidate sentences
    print("Pre-retrieving candidate sentences for Validation and Diagnostic claims...")
    val_candidates = []
    for c in val_claims:
        docs = retriever.retrieve_documents(c["claim"], top_k=Config.BM25_TOP_K_DOCS)
        sents = retriever.retrieve_sentences(c["claim"], docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES)
        val_candidates.append(sents)

    diag_candidates = []
    for c in test_claims:
        docs = retriever.retrieve_documents(c["claim"], top_k=Config.BM25_TOP_K_DOCS)
        sents = retriever.retrieve_sentences(c["claim"], docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES)
        diag_candidates.append(sents)

    models_to_test = [
        {"key": "baseline", "name": "Model A (Baseline finetuned_nli)", "path": Config.MODELS_DIR / "finetuned_nli"},
        {"key": "augmented", "name": "Model B (Phase 3 Augmented)", "path": Config.MODELS_DIR / "experimental_augmented_nli"},
        {"key": "calibrated", "name": "Model C (Phase 4 Calibrated Balanced)", "path": Config.MODELS_DIR / "experimental_calibrated_nli"},
    ]

    benchmark_results = {}
    diagnostic_case_audits = {}

    for model_meta in models_to_test:
        m_key = model_meta["key"]
        m_name = model_meta["name"]
        m_path = model_meta["path"]

        if not m_path.exists():
            print(f"Skipping {m_name} (path not found: {m_path})")
            continue

        print(f"\n=======================================================")
        print(f"Evaluating {m_name}")
        print(f"Path: {m_path}")
        print(f"=======================================================")

        t0_load = time.time()
        model = NLIEntailmentModel(model_name=str(m_path), device="cpu")
        load_time = time.time() - t0_load

        # --- 1. Validation Set Inference & Threshold Tuning ---
        print("Scoring Validation Set (N=161)...")
        val_outputs = []
        t0_val = time.time()
        for c, sents in zip(val_claims, val_candidates):
            pairs = [(s["text"], c["claim"]) for s in sents]
            val_outputs.append(model.predict_batch(pairs))
        val_latency_ms = ((time.time() - t0_val) / len(val_claims)) * 1000.0

        # Grid search strictly on Validation Set
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        best_val_f1 = -1.0
        best_tau_sup = 0.50
        best_tau_con = 0.50
        best_val_acc = 0.0

        for t_sup in threshold_grid:
            for t_con in threshold_grid:
                preds = []
                for sents, outs in zip(val_candidates, val_outputs):
                    pred, _, _ = evaluate_claim_evidence_selection(sents, outs, t_sup, t_con)
                    preds.append(pred)
                f1 = f1_score(val_golds, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0)
                acc = accuracy_score(val_golds, preds)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_val_acc = acc
                    best_tau_sup = t_sup
                    best_tau_con = t_con

        print(f"Validation Tuning Complete: Best tau_sup={best_tau_sup}, tau_con={best_tau_con} (Macro-F1: {best_val_f1:.4f}, Acc: {best_val_acc*100:.2f}%)")

        def compute_val_metrics(t_sup, t_con):
            preds = []
            probs_matrix = []
            for sents, outs in zip(val_candidates, val_outputs):
                pred, _, probs = evaluate_claim_evidence_selection(sents, outs, t_sup, t_con)
                preds.append(pred)
                probs_matrix.append(probs)

            acc = accuracy_score(val_golds, preds)
            f1 = f1_score(val_golds, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0)
            cm = confusion_matrix(val_golds, preds, labels=CANONICAL_LABELS).tolist()
            brier = multiclass_brier_score(val_golds_idx, np.array(probs_matrix))
            prec, rec, f1_per, supp = precision_recall_fscore_support(val_golds, preds, labels=CANONICAL_LABELS, zero_division=0)
            per_class = {
                lbl: {"precision": float(prec[i]), "recall": float(rec[i]), "f1_score": float(f1_per[i]), "support": int(supp[i])}
                for i, lbl in enumerate(CANONICAL_LABELS)
            }
            return {
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "per_class": per_class,
                "confusion_matrix": cm,
                "brier_score": float(brier),
                "tau_sup": t_sup,
                "tau_con": t_con,
            }

        val_default_metrics = compute_val_metrics(0.50, 0.50)
        val_tuned_metrics = compute_val_metrics(best_tau_sup, best_tau_con)

        # --- 2. Diagnostic Test Inference & Evaluation ---
        print("Scoring Diagnostic Test Set (N=24)...")
        diag_outputs = []
        t0_diag = time.time()
        for c, sents in zip(test_claims, diag_candidates):
            pairs = [(s["text"], c["claim"]) for s in sents]
            diag_outputs.append(model.predict_batch(pairs))
        diag_latency_ms = ((time.time() - t0_diag) / len(test_claims)) * 1000.0

        def compute_diag_metrics(t_sup, t_con):
            preds = []
            probs_matrix = []
            claim_records = []
            for c, sents, outs in zip(test_claims, diag_candidates, diag_outputs):
                pred, ev, probs = evaluate_claim_evidence_selection(sents, outs, t_sup, t_con)
                preds.append(pred)
                probs_matrix.append(probs)
                claim_records.append({
                    "claim_id": c["id"],
                    "claim": c["claim"],
                    "gold": c["gold_label"],
                    "prediction": pred,
                    "confidence": ev["p"],
                    "probabilities": {LABEL_CONTRADICTS: probs[0], LABEL_SUPPORTS: probs[1], LABEL_NEI: probs[2]},
                    "evidence_doc": ev["doc_id"],
                    "evidence_sentence_idx": ev["sentence_idx"],
                    "evidence_text": ev["text"],
                })

            acc = accuracy_score(diag_golds, preds)
            f1 = f1_score(diag_golds, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0)
            cm = confusion_matrix(diag_golds, preds, labels=CANONICAL_LABELS).tolist()
            brier = multiclass_brier_score(diag_golds_idx, np.array(probs_matrix))
            prec, rec, f1_per, supp = precision_recall_fscore_support(diag_golds, preds, labels=CANONICAL_LABELS, zero_division=0)
            per_class = {
                lbl: {"precision": float(prec[i]), "recall": float(rec[i]), "f1_score": float(f1_per[i]), "support": int(supp[i])}
                for i, lbl in enumerate(CANONICAL_LABELS)
            }
            return {
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "per_class": per_class,
                "confusion_matrix": cm,
                "brier_score": float(brier),
                "tau_sup": t_sup,
                "tau_con": t_con,
                "claims": claim_records,
            }

        diag_default_metrics = compute_diag_metrics(0.50, 0.50)
        diag_tuned_metrics = compute_diag_metrics(best_tau_sup, best_tau_con)

        # Store diagnostic audit records for comparison
        diagnostic_case_audits[m_key] = {
            "default": diag_default_metrics["claims"],
            "tuned": diag_tuned_metrics["claims"],
        }

        # --- 3. Challenge Set Evaluation (N=60) ---
        print("Scoring Challenge Set (N=60)...")
        ch_outputs = model.predict_batch(ch_pairs_text)

        def compute_ch_metrics(t_sup, t_con):
            preds = []
            probs_matrix = []
            for out in ch_outputs:
                p_sup = out["probabilities"].get(LABEL_SUPPORTS, 0.0)
                p_con = out["probabilities"].get(LABEL_CONTRADICTS, 0.0)
                if p_sup >= t_sup and p_con >= t_con:
                    pred = LABEL_SUPPORTS if p_sup >= p_con else LABEL_CONTRADICTS
                elif p_sup >= t_sup:
                    pred = LABEL_SUPPORTS
                elif p_con >= t_con:
                    pred = LABEL_CONTRADICTS
                else:
                    pred = LABEL_NEI
                preds.append(pred)
                probs_matrix.append([
                    out["probabilities"].get(LABEL_CONTRADICTS, 0.0),
                    out["probabilities"].get(LABEL_SUPPORTS, 0.0),
                    out["probabilities"].get(LABEL_NEI, 0.0),
                ])

            acc = accuracy_score(ch_golds_str, preds)
            f1 = f1_score(ch_golds_str, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0)
            brier = multiclass_brier_score(ch_golds_idx, np.array(probs_matrix))

            # Breakdown by category
            cat_metrics = {}
            for cat in sorted(list(set(ch_categories))):
                indices = [i for i, c in enumerate(ch_categories) if c == cat]
                cat_gold = [ch_golds_str[i] for i in indices]
                cat_pred = [preds[i] for i in indices]
                cat_acc = accuracy_score(cat_gold, cat_pred)
                cat_metrics[cat] = {"accuracy": float(cat_acc), "count": len(indices)}

            return {
                "accuracy": float(acc),
                "macro_f1": float(f1),
                "brier_score": float(brier),
                "category_breakdown": cat_metrics,
            }

        ch_default_metrics = compute_ch_metrics(0.50, 0.50)
        ch_tuned_metrics = compute_ch_metrics(best_tau_sup, best_tau_con)

        # Store all results for this model
        benchmark_results[m_key] = {
            "name": m_name,
            "path": str(m_path),
            "optimal_val_thresholds": {"tau_sup": best_tau_sup, "tau_con": best_tau_con},
            "validation": {
                "default_thresholds": val_default_metrics,
                "validation_tuned_thresholds": val_tuned_metrics,
                "latency_ms_per_claim": float(val_latency_ms),
            },
            "diagnostic": {
                "default_thresholds": {k: v for k, v in diag_default_metrics.items() if k != "claims"},
                "validation_tuned_thresholds": {k: v for k, v in diag_tuned_metrics.items() if k != "claims"},
                "latency_ms_per_claim": float(diag_latency_ms),
            },
            "challenge": {
                "default_thresholds": ch_default_metrics,
                "validation_tuned_thresholds": ch_tuned_metrics,
            },
        }

    # Save outputs
    output_json_path = Config.BASE_DIR / "experiments" / "phase4_calibration_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_results": benchmark_results,
            "diagnostic_case_audits": diagnostic_case_audits,
        }, f, indent=2)

    print(f"\n=======================================================")
    print(f"Benchmark Complete! Results saved to {output_json_path}")
    print(f"=======================================================")

    return benchmark_results, diagnostic_case_audits


if __name__ == "__main__":
    run_comprehensive_benchmark()
