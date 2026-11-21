"""
Comprehensive NLI Error Analysis & Reliability Audit Script
Evaluates the fine-tuned NLI model and pipeline across:
1. Internal Validation Set (val_internal.jsonl, N=161)
2. Diagnostic New Claims Set (new_claims.jsonl, N=24)
Categorizes errors into:
- Negation Polarity Failures
- Causality vs. Correlation/Association Mismatch
- Directional / Semantic Reversals
- Numerical & Magnitude Comparison Errors
- Temporal & Population Scope Mismatches
- Out-of-Context Semantic Decoupling (Retrieval-Induced)
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import load_corpus, load_claims, LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI, CANONICAL_LABELS
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.pipeline import ClaimVerificationPipeline

def detect_error_category(claim: str, evidence_text: str, gold: str, pred: str, title: str = "") -> str:
    """Classify the primary failure mode of an erroneous prediction."""
    claim_lower = claim.lower()
    ev_lower = evidence_text.lower()
    title_lower = title.lower()

    # 1. Numerical Comparison
    has_num_claim = bool(re.search(r"\b\d+(?:\.\d+)?%?|\b(one|two|three|fold|billion|million)\b", claim_lower))
    has_num_ev = bool(re.search(r"\b\d+(?:\.\d+)?%?|\b(one|two|three|fold|billion|million)\b", ev_lower))
    if has_num_claim and has_num_ev and (gold != pred):
        # Check if numbers mismatch
        nums_c = re.findall(r"\d+(?:\.\d+)?%?", claim_lower)
        nums_e = re.findall(r"\d+(?:\.\d+)?%?", ev_lower)
        if nums_c and nums_e and set(nums_c) != set(nums_e):
            return "numerical_magnitude_comparison"

    # 2. Negation Failure
    negation_words = {"not", "no", "never", "cannot", "fails to", "failed", "absence", "without", "lacks", "neither", "nor"}
    claim_has_neg = any(w in claim_lower.split() for w in negation_words)
    ev_has_neg = any(w in ev_lower.split() for w in negation_words)
    if (claim_has_neg != ev_has_neg) and (gold == "CONTRADICTS" or pred == "CONTRADICTS"):
        return "negation_polarity_reversal"

    # 3. Causality vs. Association
    causal_words = {"causes", "caused", "causing", "leads to", "induces", "induced", "triggers", "triggered", "prevents", "eradicates", "eliminates"}
    assoc_words = {"associated with", "correlated with", "linked to", "observed in", "relationship between"}
    if any(w in claim_lower for w in causal_words) and any(w in ev_lower for w in assoc_words):
        return "causality_vs_association_mismatch"

    # 4. Directional Reversal (e.g. increase vs decrease)
    increase_words = {"increases", "increased", "increasing", "higher", "elevated", "promotes", "enhances"}
    decrease_words = {"decreases", "decreased", "decreasing", "lower", "reduced", "inhibits", "suppresses", "downregulates"}
    c_inc = any(w in claim_lower for w in increase_words)
    c_dec = any(w in claim_lower for w in decrease_words)
    e_inc = any(w in ev_lower for w in increase_words)
    e_dec = any(w in ev_lower for w in decrease_words)
    if (c_inc and e_dec) or (c_dec and e_inc):
        return "directional_reversal"

    # 5. Temporal & Scope Mismatch
    temporal_words = {"first trimester", "during pregnancy", "early-life", "long-term", "elderly", "in vitro", "in vivo", "in mice", "murine"}
    if any(w in claim_lower for w in temporal_words) and not any(w in ev_lower for w in temporal_words):
        return "temporal_population_scope_mismatch"

    # 6. Semantic Decoupling / Retrieval Drift (Claim topic missing from evidence paper)
    if title:
        key_nouns = [w for w in claim_lower.split() if len(w) > 4 and w not in {"which", "their", "during", "other", "about"}]
        # Check if fewer than 1 key noun is in title or evidence
        overlap = sum(1 for w in key_nouns if w in ev_lower or w in title_lower)
        if overlap <= 1 and gold == "NOT ENOUGH INFORMATION":
            return "semantic_decoupling_retrieval_drift"

    return "general_nli_semantic_reasoning_error"

def main():
    print("=" * 70)
    print("Running NLI Reliability & Error Audit")
    print("=" * 70)

    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus):,} corpus abstracts.")

    model_path = Config.MODELS_DIR / "finetuned_nli"
    nli_model = NLIEntailmentModel(model_name=str(model_path), device="cpu")
    bm25 = BM25Retriever(corpus)

    pipeline = ClaimVerificationPipeline(
        retriever=bm25,
        entailment_model=nli_model,
        top_k_docs=Config.BM25_TOP_K_DOCS,
        top_k_sentences=Config.BM25_TOP_K_SENTENCES,
        entail_threshold=Config.ENTAIL_THRESHOLD,
        contradict_threshold=Config.CONTRADICT_THRESHOLD,
    )

    # 1. Evaluate on val_internal.jsonl (N=161)
    val_claims = load_claims(Config.PROCESSED_DATA_DIR / "val_internal.jsonl")
    val_errors = []
    val_y_true = []
    val_y_pred = []

    for c in val_claims:
        cid = c["id"]
        claim_text = c["claim"]
        gold_label = c["gold_label"]
        val_y_true.append(gold_label)

        res = pipeline.verify_claim(claim_text, claim_id=cid)
        pred_label = res["predicted_label"]
        val_y_pred.append(pred_label)

        if pred_label != gold_label:
            top_ev = res.get("evidence_details", [])[0] if res.get("evidence_details") else {}
            ev_text = top_ev.get("text", "")
            doc_id = top_ev.get("doc_id", None)
            doc_title = corpus.get(doc_id, {}).get("title", "") if doc_id else ""
            category = detect_error_category(claim_text, ev_text, gold_label, pred_label, doc_title)

            val_errors.append({
                "claim_id": cid,
                "claim": claim_text,
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "confidence": res.get("confidence", 0.0),
                "error_category": category,
                "retrieved_doc_id": doc_id,
                "doc_title": doc_title,
                "evidence_text": ev_text,
                "probabilities": top_ev.get("probabilities", {}),
            })

    # 2. Evaluate on new_claims.jsonl (N=24)
    with open(BASE_DIR / "experiments" / "new_claims.jsonl", "r", encoding="utf-8") as f:
        raw_test = [json.loads(line) for line in f if line.strip()]

    test_errors = []
    test_y_true = []
    test_y_pred = []

    for c in raw_test:
        cid = c["id"]
        claim_text = c["claim"]
        gold_label = c["gold_label"]
        test_y_true.append(gold_label)

        res = pipeline.verify_claim(claim_text, claim_id=cid)
        pred_label = res["predicted_label"]
        test_y_pred.append(pred_label)

        if pred_label != gold_label:
            top_ev = res.get("evidence_details", [])[0] if res.get("evidence_details") else {}
            ev_text = top_ev.get("text", "")
            doc_id = top_ev.get("doc_id", None)
            doc_title = corpus.get(doc_id, {}).get("title", "") if doc_id else ""
            category = detect_error_category(claim_text, ev_text, gold_label, pred_label, doc_title)

            test_errors.append({
                "claim_id": cid,
                "claim": claim_text,
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "confidence": res.get("confidence", 0.0),
                "error_category": category,
                "retrieved_doc_id": doc_id,
                "doc_title": doc_title,
                "evidence_text": ev_text,
                "probabilities": top_ev.get("probabilities", {}),
            })

    # Metrics computation
    def compute_summary(y_true, y_pred, errors):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        acc = correct / total if total > 0 else 0.0
        cm = {l1: {l2: 0 for l2 in CANONICAL_LABELS} for l1 in CANONICAL_LABELS}
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        per_class = {}
        for l in CANONICAL_LABELS:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class[l] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": sum(1 for t in y_true if t == l)}

        cat_counts = {}
        for e in errors:
            cat = e["error_category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "total_claims": total,
            "correct": correct,
            "error_count": len(errors),
            "accuracy": round(acc, 4),
            "macro_f1": round(sum(v["f1"] for v in per_class.values()) / 3, 4),
            "per_class": per_class,
            "confusion_matrix": [[cm[l1][l2] for l2 in CANONICAL_LABELS] for l1 in CANONICAL_LABELS],
            "error_category_distribution": cat_counts
        }

    val_summary = compute_summary(val_y_true, val_y_pred, val_errors)
    test_summary = compute_summary(test_y_true, test_y_pred, test_errors)

    out_data = {
        "timestamp": "2026-11-21",
        "model": str(model_path),
        "validation_benchmark_n161": {
            "summary": val_summary,
            "errors": val_errors
        },
        "diagnostic_benchmark_n24": {
            "summary": test_summary,
            "errors": test_errors
        }
    }

    out_file = BASE_DIR / "experiments" / "nli_error_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nSaved detailed NLI error analysis to: {out_file}")
    print(f"Val Accuracy: {val_summary['accuracy']*100:.2f}%, Errors: {val_summary['error_count']}/161")
    print(f"Val Error Categories: {val_summary['error_category_distribution']}")
    print(f"Test Accuracy: {test_summary['accuracy']*100:.2f}%, Errors: {test_summary['error_count']}/24")
    print(f"Test Error Categories: {test_summary['error_category_distribution']}")

if __name__ == "__main__":
    main()
