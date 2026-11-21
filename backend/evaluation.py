import json
import platform
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)

def evaluate_retrieval(
    claims: List[Dict[str, Any]],
    retrieved_docs_per_claim: List[List[int]],
    k_list: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Calculate Document Evidence Recall@K across all claims with gold evidence.
    A hit occurs if at least one gold evidence doc_id is among the top-k retrieved documents.
    """
    eligible_claims = 0
    hits_at_k = {k: 0 for k in k_list}
    all_hits_at_k = {k: 0 for k in k_list}

    for claim, retrieved in zip(claims, retrieved_docs_per_claim):
        gold_docs = claim.get("gold_doc_evidence", set())
        if not gold_docs:
            continue
        eligible_claims += 1

        for k in k_list:
            top_k_retrieved = set(retrieved[:k])
            # Any gold doc retrieved
            if bool(top_k_retrieved.intersection(gold_docs)):
                hits_at_k[k] += 1
            # All gold docs retrieved
            if gold_docs.issubset(top_k_retrieved):
                all_hits_at_k[k] += 1

    results = {
        "eligible_claims_with_evidence": eligible_claims,
    }
    for k in k_list:
        results[f"doc_recall@{k}"] = (hits_at_k[k] / eligible_claims) if eligible_claims > 0 else 0.0
        results[f"doc_strict_recall@{k}"] = (all_hits_at_k[k] / eligible_claims) if eligible_claims > 0 else 0.0

    return results

def evaluate_sentence_evidence(
    claims: List[Dict[str, Any]],
    predicted_sentences_per_claim: List[List[Tuple[int, int]]]
) -> Dict[str, float]:
    """
    Calculate sentence-level evidence Precision, Recall, and F1 against annotated gold evidence.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    claims_with_gold_sentences = 0

    for claim, pred_sentences in zip(claims, predicted_sentences_per_claim):
        gold_sentences = claim.get("gold_sentence_evidence", set())
        if gold_sentences:
            claims_with_gold_sentences += 1

        pred_set = set(pred_sentences)
        tp = len(pred_set.intersection(gold_sentences))
        fp = len(pred_set - gold_sentences)
        fn = len(gold_sentences - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "claims_with_gold_evidence_sentences": claims_with_gold_sentences,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "sentence_evidence_precision": precision,
        "sentence_evidence_recall": recall,
        "sentence_evidence_f1": f1,
    }

def evaluate_claim_verification(
    gold_labels: List[str],
    pred_labels: List[str]
) -> Dict[str, Any]:
    """
    Calculate Macro-F1 and per-class Precision, Recall, and F1 for claim verification,
    along with confusion matrix.
    """
    # Classification metrics
    report = classification_report(
        gold_labels,
        pred_labels,
        labels=CANONICAL_LABELS,
        target_names=CANONICAL_LABELS,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        gold_labels,
        pred_labels,
        labels=CANONICAL_LABELS
    ).tolist()

    macro_f1 = f1_score(gold_labels, pred_labels, labels=CANONICAL_LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(gold_labels, pred_labels, labels=CANONICAL_LABELS, average="weighted", zero_division=0)

    per_class = {}
    for lbl in CANONICAL_LABELS:
        per_class[lbl] = {
            "precision": report[lbl]["precision"],
            "recall": report[lbl]["recall"],
            "f1_score": report[lbl]["f1-score"],
            "support": report[lbl]["support"]
        }

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": CANONICAL_LABELS,
            "matrix": cm
        }
    }

def generate_evaluation_report(
    eval_summary: Dict[str, Any],
    inspectable_examples: List[Dict[str, Any]],
    output_md_path: Path
):
    """
    Generate a comprehensive, human-readable Markdown evaluation report.
    """
    md = []
    md.append("# ResearchMap — Baseline NLP Evaluation Report\n")
    md.append(f"**Date & Timestamp**: {eval_summary.get('timestamp')}")
    md.append(f"**Dataset**: SciFact (EMNLP 2020)")
    md.append(f"**Split Evaluated**: `{eval_summary.get('split_name')}` ({eval_summary.get('total_claims')} claims)")
    md.append(f"**Lexical Retrieval**: BM25Okapi (Top-{eval_summary.get('bm25_top_k_docs')} Docs, Top-{eval_summary.get('bm25_top_k_sentences')} Sentences)")
    md.append(f"**NLI Model**: `{eval_summary.get('nli_model_name')}`")
    md.append(f"**Hardware / Device**: `{eval_summary.get('device')}` ({eval_summary.get('platform')})\n")

    md.append("## 1. Executive Summary & Headline Metrics\n")
    md.append("| Metric | Baseline Value | Description |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Macro-F1 (Claim Verification)** | **{eval_summary['claim_verification']['macro_f1']:.4f}** | Unweighted mean of F1 scores across 3 classes |")
    md.append(f"| **Weighted-F1** | {eval_summary['claim_verification']['weighted_f1']:.4f} | Class-prevalence weighted F1 score |")
    md.append(f"| **Evidence Retrieval Recall@5** | **{eval_summary['retrieval']['doc_recall@5']:.4f}** | % of claims where at least one gold doc is in Top-5 BM25 |")
    md.append(f"| **Evidence Retrieval Recall@10** | **{eval_summary['retrieval']['doc_recall@10']:.4f}** | % of claims where at least one gold doc is in Top-10 BM25 |")
    md.append(f"| **Evidence Sentence Precision** | {eval_summary['sentence_evidence']['sentence_evidence_precision']:.4f} | Precision of retrieved evidence sentences vs gold |")
    md.append(f"| **Evidence Sentence Recall** | {eval_summary['sentence_evidence']['sentence_evidence_recall']:.4f} | Recall of retrieved evidence sentences vs gold |")
    md.append(f"| **Evidence Sentence F1** | {eval_summary['sentence_evidence']['sentence_evidence_f1']:.4f} | Harmonic mean of sentence evidence P & R |\n")

    md.append("## 2. Per-Class Claim Verification Breakdown\n")
    md.append("| Class | Precision | Recall | F1-Score | Support (Gold Count) |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for lbl in CANONICAL_LABELS:
        m = eval_summary["claim_verification"]["per_class"][lbl]
        md.append(f"| **{lbl}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['support']} |")
    md.append("")

    md.append("## 3. Confusion Matrix\n")
    cm = eval_summary["claim_verification"]["confusion_matrix"]["matrix"]
    md.append("| True \\ Pred | SUPPORTS | CONTRADICTS | NOT ENOUGH INFO |")
    md.append("| :--- | :---: | :---: | :---: |")
    for idx, lbl in enumerate(CANONICAL_LABELS):
        row = cm[idx]
        md.append(f"| **{lbl}** | {row[0]} | {row[1]} | {row[2]} |")
    md.append("")

    md.append("## 4. Evidence Retrieval Performance\n")
    ret = eval_summary["retrieval"]
    md.append(f"- **Eligible Claims with Gold Evidence**: {ret['eligible_claims_with_evidence']}")
    md.append(f"- **Doc Recall@5 (Any Gold Doc)**: {ret['doc_recall@5'] * 100:.2f}%")
    md.append(f"- **Doc Recall@10 (Any Gold Doc)**: {ret['doc_recall@10'] * 100:.2f}%")
    md.append(f"- **Strict Doc Recall@5 (All Gold Docs)**: {ret.get('doc_strict_recall@5', 0.0) * 100:.2f}%")
    md.append(f"- **Strict Doc Recall@10 (All Gold Docs)**: {ret.get('doc_strict_recall@10', 0.0) * 100:.2f}%\n")

    md.append("## 5. Sentence Evidence Performance\n")
    sev = eval_summary["sentence_evidence"]
    md.append(f"- **Claims with Gold Sentences**: {sev['claims_with_gold_evidence_sentences']}")
    md.append(f"- **True Positive Sentences**: {sev['true_positives']}")
    md.append(f"- **False Positive Sentences**: {sev['false_positives']}")
    md.append(f"- **False Negative Sentences**: {sev['false_negatives']}")
    md.append(f"- **Sentence Evidence Precision**: {sev['sentence_evidence_precision']:.4f}")
    md.append(f"- **Sentence Evidence Recall**: {sev['sentence_evidence_recall']:.4f}")
    md.append(f"- **Sentence Evidence F1**: {sev['sentence_evidence_f1']:.4f}\n")

    md.append("## 6. Inspectable Prediction Examples\n")
    for i, ex in enumerate(inspectable_examples, 1):
        md.append(f"### Example {i}: Claim ID {ex['claim_id']}\n")
        md.append(f"- **Claim**: *\"{ex['claim']}\"*")
        md.append(f"- **Gold Label**: `{ex['gold_label']}`")
        status_icon = "CORRECT" if ex['gold_label'] == ex['predicted_label'] else "INCORRECT"
        md.append(f"- **Predicted Label**: `{ex['predicted_label']}` ({status_icon})")
        md.append(f"- **Confidence Score**: `{ex['confidence']:.4f}`")
        if ex.get("evidence_details"):
            best_ev = ex["evidence_details"][0]
            md.append(f"- **Top Retrieved Evidence (Doc {best_ev['doc_id']}, Sent {best_ev['sentence_idx']})**:")
            md.append(f"  > *\"{best_ev['text']}\"*")
            md.append(f"  - Document Title: *{best_ev['title']}*")
            md.append(f"  - BM25 Lexical Score: `{best_ev['bm25_score']:.2f}`")
            md.append(f"  - NLI Probabilities: SUPPORTS={best_ev['probabilities'][LABEL_SUPPORTS]:.3f}, CONTRADICTS={best_ev['probabilities'][LABEL_CONTRADICTS]:.3f}, NEI={best_ev['probabilities'][LABEL_NEI]:.3f}")
        else:
            md.append("- **Retrieved Evidence**: *(None - No matching candidates)*")
        md.append("")

    md.append("## 7. Environment & Reproducibility Details\n")
    md.append(f"- **Python Version**: `{eval_summary.get('python_version')}`")
    md.append(f"- **Random Seed**: `{eval_summary.get('random_seed')}`")
    md.append(f"- **Packages**: `torch=={eval_summary.get('torch_version')}`, `transformers=={eval_summary.get('transformers_version')}`, `rank_bm25==0.2.2`, `scikit-learn=={eval_summary.get('sklearn_version')}`")
    md.append(f"- **Dataset Source**: SciFact Official AI2 Release (`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`)")
    md.append(f"- **License**: Apache 2.0 (Allen Institute for AI)\n")

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Human-readable evaluation report saved to: {output_md_path.resolve()}")
