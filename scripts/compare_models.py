"""
Script to fairly compare the Baseline Zero-Shot model against the Fine-Tuned model
on IDENTICAL held-out evaluation claims (claims_dev.jsonl or val_internal.jsonl).
Generates side-by-side comparison tables, delta metrics, confusion matrices, and markdown report.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure workspace root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import load_corpus, load_claims, CANONICAL_LABELS
from scripts.evaluate_baseline import run_evaluation

def compare_models(
    split_type: str = "dev",
    baseline_model: str = "cross-encoder/nli-deberta-v3-small",
    finetuned_model_dir: str = "models/finetuned_nli",
    output_prefix: str = "model_comparison"
) -> Dict[str, Any]:
    print("============================================================")
    print("=== RESEARCHMAP PHASE 2: FAIR MODEL COMPARISON BENCHMARK ===")
    print("============================================================")
    print(f"Held-Out Split: {split_type}")
    print(f"Baseline Model: {baseline_model}")
    print(f"Fine-Tuned Model: {finetuned_model_dir}")

    # Step 1: Run evaluation for Baseline Model
    print("\n>>> EVALUATING BASELINE (ZERO-SHOT) MODEL >>>")
    baseline_summary = run_evaluation(
        split_type=split_type,
        model_name=baseline_model,
        output_prefix=f"{output_prefix}_baseline"
    )

    # Step 2: Run evaluation for Fine-Tuned Model
    print("\n>>> EVALUATING FINE-TUNED MODEL >>>")
    finetuned_summary = run_evaluation(
        split_type=split_type,
        model_name=finetuned_model_dir,
        output_prefix=f"{output_prefix}_finetuned"
    )

    # Step 3: Compute Deltas
    base_f1 = baseline_summary["claim_verification"]["macro_f1"]
    fine_f1 = finetuned_summary["claim_verification"]["macro_f1"]
    delta_macro_f1 = fine_f1 - base_f1

    base_weighted_f1 = baseline_summary["claim_verification"]["weighted_f1"]
    fine_weighted_f1 = finetuned_summary["claim_verification"]["weighted_f1"]
    delta_weighted_f1 = fine_weighted_f1 - base_weighted_f1

    base_sent_f1 = baseline_summary["sentence_evidence"]["sentence_evidence_f1"]
    fine_sent_f1 = finetuned_summary["sentence_evidence"]["sentence_evidence_f1"]
    delta_sent_f1 = fine_sent_f1 - base_sent_f1

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "split_evaluated": baseline_summary["split_name"],
        "total_claims": baseline_summary["total_claims"],
        "baseline_model": baseline_model,
        "finetuned_model": finetuned_model_dir,
        "metrics": {
            "baseline": baseline_summary,
            "finetuned": finetuned_summary,
            "deltas": {
                "macro_f1": delta_macro_f1,
                "weighted_f1": delta_weighted_f1,
                "sentence_evidence_f1": delta_sent_f1,
            }
        }
    }

    # Save machine-readable comparison
    reports_dir = Config.REPORTS_DIR
    json_path = reports_dir / f"{output_prefix}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Generate Markdown Comparison Report
    md_path = reports_dir / f"{output_prefix}_report.md"
    generate_comparison_markdown(baseline_summary, finetuned_summary, delta_macro_f1, delta_weighted_f1, md_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("=== FAIR COMPARISON SUMMARY (IDENTICAL HELD-OUT SPLIT) ===")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Baseline (Zero-Shot)':<20} | {'Fine-Tuned':<15} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Macro-F1 (Claim Verif)':<30} | {base_f1:<20.4f} | {fine_f1:<15.4f} | {delta_macro_f1:>+7.4f}")
    print(f"{'Weighted-F1':<30} | {base_weighted_f1:<20.4f} | {fine_weighted_f1:<15.4f} | {delta_weighted_f1:>+7.4f}")
    print(f"{'Sentence Evidence F1':<30} | {base_sent_f1:<20.4f} | {fine_sent_f1:<15.4f} | {delta_sent_f1:>+7.4f}")

    print("\nPer-Class F1 Breakdown:")
    for lbl in CANONICAL_LABELS:
        b_class = baseline_summary["claim_verification"]["per_class"][lbl]["f1_score"]
        f_class = finetuned_summary["claim_verification"]["per_class"][lbl]["f1_score"]
        d_class = f_class - b_class
        print(f"  {lbl:<24} | Base: {b_class:.4f} | Fine: {f_class:.4f} | Delta: {d_class:>+7.4f}")
    print("=" * 70)

    return comparison

def generate_comparison_markdown(base_s: Dict[str, Any], fine_s: Dict[str, Any], delta_f1: float, delta_wf1: float, out_path: Path):
    md = []
    md.append("# ResearchMap — Phase 2 Model Comparison Report\n")
    md.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Split Evaluated**: `{base_s['split_name']}` ({base_s['total_claims']} claims)")
    md.append(f"**Baseline Model**: `{base_s['nli_model_name']}` (Zero-Shot)")
    md.append(f"**Fine-Tuned Model**: `{fine_s['nli_model_name']}` (Trained on SciFact train partition)\n")

    md.append("## 1. Executive Comparison Table\n")
    md.append("| Metric | Zero-Shot Baseline | Fine-Tuned Model | Delta (Absolute) | Relative Change |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")

    b_f1 = base_s["claim_verification"]["macro_f1"]
    f_f1 = fine_s["claim_verification"]["macro_f1"]
    rel_f1 = ((f_f1 - b_f1) / b_f1 * 100) if b_f1 > 0 else 0.0
    md.append(f"| **Claim Verification Macro-F1** | **{b_f1:.4f}** | **{f_f1:.4f}** | **{f_f1 - b_f1:>+0.4f}** | **{rel_f1:>+0.2f}%** |")

    b_wf1 = base_s["claim_verification"]["weighted_f1"]
    f_wf1 = fine_s["claim_verification"]["weighted_f1"]
    rel_wf1 = ((f_wf1 - b_wf1) / b_wf1 * 100) if b_wf1 > 0 else 0.0
    md.append(f"| **Weighted-F1** | {b_wf1:.4f} | {f_wf1:.4f} | {f_wf1 - b_wf1:>+0.4f} | {rel_wf1:>+0.2f}% |")

    b_r5 = base_s["retrieval"]["doc_recall@5"]
    f_r5 = fine_s["retrieval"]["doc_recall@5"]
    md.append(f"| **Evidence Doc Recall@5 (BM25)** | {b_r5*100:.2f}% | {f_r5*100:.2f}% | {(f_r5 - b_r5)*100:>+0.2f}% | Shared Retriever |")

    b_r10 = base_s["retrieval"]["doc_recall@10"]
    f_r10 = fine_s["retrieval"]["doc_recall@10"]
    md.append(f"| **Evidence Doc Recall@10 (BM25)** | {b_r10*100:.2f}% | {f_r10*100:.2f}% | {(f_r10 - b_r10)*100:>+0.2f}% | Shared Retriever |")

    b_sp = base_s["sentence_evidence"]["sentence_evidence_precision"]
    f_sp = fine_s["sentence_evidence"]["sentence_evidence_precision"]
    md.append(f"| **Sentence Evidence Precision** | {b_sp:.4f} | {f_sp:.4f} | {f_sp - b_sp:>+0.4f} | - |")

    b_sr = base_s["sentence_evidence"]["sentence_evidence_recall"]
    f_sr = fine_s["sentence_evidence"]["sentence_evidence_recall"]
    md.append(f"| **Sentence Evidence Recall** | {b_sr:.4f} | {f_sr:.4f} | {f_sr - b_sr:>+0.4f} | - |")

    b_sf1 = base_s["sentence_evidence"]["sentence_evidence_f1"]
    f_sf1 = fine_s["sentence_evidence"]["sentence_evidence_f1"]
    md.append(f"| **Sentence Evidence F1** | {b_sf1:.4f} | {f_sf1:.4f} | {f_sf1 - b_sf1:>+0.4f} | - |\n")

    md.append("## 2. Per-Class Performance Breakdown\n")
    md.append("| Class | Baseline Precision | Baseline Recall | Baseline F1 | Fine-Tuned Precision | Fine-Tuned Recall | Fine-Tuned F1 | F1 Delta |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for lbl in CANONICAL_LABELS:
        b_p = base_s["claim_verification"]["per_class"][lbl]["precision"]
        b_r = base_s["claim_verification"]["per_class"][lbl]["recall"]
        b_c = base_s["claim_verification"]["per_class"][lbl]["f1_score"]

        f_p = fine_s["claim_verification"]["per_class"][lbl]["precision"]
        f_r = fine_s["claim_verification"]["per_class"][lbl]["recall"]
        f_c = fine_s["claim_verification"]["per_class"][lbl]["f1_score"]

        md.append(f"| **{lbl}** | {b_p:.3f} | {b_r:.3f} | {b_c:.3f} | {f_p:.3f} | {f_r:.3f} | **{f_c:.3f}** | **{f_c - b_c:>+0.3f}** |")
    md.append("")

    md.append("## 3. Confusion Matrices\n")
    md.append("### Baseline (Zero-Shot) Confusion Matrix:")
    md.append("| True \\ Pred | SUPPORTS | CONTRADICTS | NOT ENOUGH INFO |")
    md.append("| :--- | :---: | :---: | :---: |")
    for idx, lbl in enumerate(CANONICAL_LABELS):
        row = base_s["claim_verification"]["confusion_matrix"]["matrix"][idx]
        md.append(f"| **{lbl}** | {row[0]} | {row[1]} | {row[2]} |")
    md.append("")

    md.append("### Fine-Tuned Model Confusion Matrix:")
    md.append("| True \\ Pred | SUPPORTS | CONTRADICTS | NOT ENOUGH INFO |")
    md.append("| :--- | :---: | :---: | :---: |")
    for idx, lbl in enumerate(CANONICAL_LABELS):
        row = fine_s["claim_verification"]["confusion_matrix"]["matrix"][idx]
        md.append(f"| **{lbl}** | {row[0]} | {row[1]} | {row[2]} |")
    md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved comparison report to: {out_path.resolve()}")

if __name__ == "__main__":
    from typing import Dict, Any
    parser = argparse.ArgumentParser(description="Fair comparison between Baseline and Fine-Tuned models.")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "val"], help="Held-out split for comparison")
    parser.add_argument("--baseline", type=str, default=Config.NLI_MODEL_NAME, help="Baseline model")
    parser.add_argument("--finetuned", type=str, default="models/finetuned_nli", help="Fine-tuned checkpoint directory")
    parser.add_argument("--prefix", type=str, default="model_comparison", help="Output prefix")

    args = parser.parse_args()
    compare_models(
        split_type=args.split,
        baseline_model=args.baseline,
        finetuned_model_dir=args.finetuned,
        output_prefix=args.prefix
    )
