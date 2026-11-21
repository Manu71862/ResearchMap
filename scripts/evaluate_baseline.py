"""
Script to evaluate the Baseline NLP pipeline on SciFact.
Computes Macro-F1, per-class metrics, BM25 Recall@5/Recall@10, sentence evidence P/R/F1,
confusion matrix, and saves structured reports and inspectable examples.
"""

import os
import sys
import json
import argparse
import platform
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
import transformers
import sklearn

# Ensure workspace root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import (
    load_corpus,
    load_claims,
    get_split_statistics,
    CANONICAL_LABELS,
)
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.pipeline import ClaimVerificationPipeline
from backend.evaluation import (
    evaluate_retrieval,
    evaluate_sentence_evidence,
    evaluate_claim_verification,
    generate_evaluation_report,
)

def run_evaluation(
    split_type: str = "dev",
    limit: int = None,
    model_name: str = Config.NLI_MODEL_NAME,
    device: str = Config.DEVICE,
    top_k_docs: int = Config.BM25_TOP_K_DOCS,
    top_k_sentences: int = Config.BM25_TOP_K_SENTENCES,
    output_prefix: str = "baseline"
):
    Config.ensure_directories()
    raw_dir = Config.RAW_DATA_DIR
    processed_dir = Config.PROCESSED_DATA_DIR
    reports_dir = Config.REPORTS_DIR

    # Resolve split path
    if split_type == "dev":
        split_path = raw_dir / "claims_dev.jsonl"
        split_name = "SciFact Official Dev (claims_dev.jsonl)"
    elif split_type == "test":
        split_path = processed_dir / "test.jsonl"
        split_name = "SciFact Processed Stratified Test (test.jsonl)"
    elif split_type == "val":
        split_path = processed_dir / "val.jsonl"
        split_name = "SciFact Processed Stratified Val (val.jsonl)"
    else:
        split_path = Path(split_type)
        split_name = f"Custom Split ({split_path.name})"

    print(f"=== ResearchMap Baseline Evaluation ===")
    print(f"Evaluating split: {split_name}")
    print(f"Loading corpus from {raw_dir / 'corpus.jsonl'}...")
    corpus = load_corpus(raw_dir / "corpus.jsonl")
    print(f"Loaded {len(corpus):,} documents into corpus.")

    print(f"Loading claims from {split_path}...")
    claims = load_claims(split_path)
    if limit and limit > 0:
        claims = claims[:limit]
        print(f"Truncated claims to {len(claims)} records for quick run.")
    print(f"Loaded {len(claims)} claims for evaluation.")

    split_stats = get_split_statistics(claims)
    print("Class distribution:", split_stats["label_distribution"])

    # Initialize BM25 Retriever
    print("Building BM25 index over full corpus...")
    retriever = BM25Retriever(corpus)
    print("BM25 index built successfully.")

    # Measure retrieval performance for top-5 and top-10
    print("Evaluating BM25 candidate retrieval (Recall@5, Recall@10)...")
    retrieved_docs_top10 = []
    for c in tqdm(claims, desc="BM25 Retrieval"):
        top_docs = retriever.retrieve_documents(c["claim"], top_k=10)
        retrieved_docs_top10.append([d["doc_id"] for d in top_docs])

    retrieval_metrics = evaluate_retrieval(claims, retrieved_docs_top10, k_list=[5, 10])
    print(f"  Doc Recall@5:  {retrieval_metrics['doc_recall@5'] * 100:.2f}%")
    print(f"  Doc Recall@10: {retrieval_metrics['doc_recall@10'] * 100:.2f}%")

    # Initialize NLI model
    print(f"\nInitializing NLI Model '{model_name}' on '{device}'...")
    entailment_model = NLIEntailmentModel(model_name=model_name, device=device)

    # Initialize Pipeline
    pipeline = ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=entailment_model,
        top_k_docs=top_k_docs,
        top_k_sentences=top_k_sentences,
    )

    # Run full verification
    print(f"\nRunning end-to-end verification pipeline over {len(claims)} claims...")
    predictions = []
    pred_labels = []
    gold_labels = []
    pred_sentences_all = []

    for c in tqdm(claims, desc="Verifying Claims"):
        pred = pipeline.verify_claim(c["claim"], claim_id=c["id"])
        pred["gold_label"] = c["gold_label"]
        pred["gold_doc_evidence"] = list(c["gold_doc_evidence"])
        pred["gold_sentence_evidence"] = [list(pair) for pair in c["gold_sentence_evidence"]]

        predictions.append(pred)
        pred_labels.append(pred["predicted_label"])
        gold_labels.append(c["gold_label"])
        pred_sentences_all.append(pred["predicted_evidence_sentences"])

    # Compute claim verification classification metrics
    verification_metrics = evaluate_claim_verification(gold_labels, pred_labels)

    # Compute sentence-level evidence metrics
    sentence_evidence_metrics = evaluate_sentence_evidence(claims, pred_sentences_all)

    # Prepare summary dictionary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "split_name": split_name,
        "total_claims": len(claims),
        "split_statistics": split_stats,
        "bm25_top_k_docs": top_k_docs,
        "bm25_top_k_sentences": top_k_sentences,
        "nli_model_name": model_name,
        "device": str(entailment_model.device),
        "platform": f"{os.name} {platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "random_seed": Config.RANDOM_SEED,
        "claim_verification": verification_metrics,
        "retrieval": retrieval_metrics,
        "sentence_evidence": sentence_evidence_metrics,
    }

    # Select inspectable examples (mix of correct and error cases across classes)
    inspectable_examples = []
    # Grab 2 SUPPORTS, 2 CONTRADICTS, 2 NEI
    examples_by_label = {"SUPPORTS": 0, "CONTRADICTS": 0, "NOT ENOUGH INFORMATION": 0}
    for p in predictions:
        gl = p["gold_label"]
        if examples_by_label.get(gl, 0) < 2:
            inspectable_examples.append(p)
            examples_by_label[gl] += 1
        if len(inspectable_examples) >= 6:
            break

    # Save outputs
    preds_path = reports_dir / f"{output_prefix}_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved machine-readable predictions to: {preds_path.resolve()}")

    metrics_path = reports_dir / f"{output_prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved machine-readable metrics to: {metrics_path.resolve()}")

    report_md_path = reports_dir / f"{output_prefix}_evaluation_report.md"
    generate_evaluation_report(summary, inspectable_examples, report_md_path)

    # Print Headline Metrics
    print("\n" + "=" * 60)
    print("=== RESEARCHMAP BASELINE HEADLINE RESULTS ===")
    print("=" * 60)
    print(f"Split: {split_name} ({len(claims)} claims)")
    print(f"Macro-F1 (Claim Verification): {verification_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:                 {verification_metrics['weighted_f1']:.4f}")
    print(f"Evidence Doc Recall@5:        {retrieval_metrics['doc_recall@5'] * 100:.2f}%")
    print(f"Evidence Doc Recall@10:       {retrieval_metrics['doc_recall@10'] * 100:.2f}%")
    print(f"Evidence Sentence Precision:  {sentence_evidence_metrics['sentence_evidence_precision']:.4f}")
    print(f"Evidence Sentence Recall:     {sentence_evidence_metrics['sentence_evidence_recall']:.4f}")
    print(f"Evidence Sentence F1:         {sentence_evidence_metrics['sentence_evidence_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    for lbl in CANONICAL_LABELS:
        m = verification_metrics["per_class"][lbl]
        print(f"  {lbl:<24}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1_score']:.3f} (N={m['support']})")
    print("=" * 60)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResearchMap baseline on SciFact.")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of claims for testing")
    parser.add_argument("--model", type=str, default=Config.NLI_MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="auto, cpu, or cuda")
    parser.add_argument("--top-k-docs", type=int, default=Config.BM25_TOP_K_DOCS, help="BM25 top-k candidate documents")
    parser.add_argument("--top-k-sentences", type=int, default=Config.BM25_TOP_K_SENTENCES, help="BM25 top-k candidate sentences")
    parser.add_argument("--prefix", type=str, default="baseline", help="Output file prefix under reports/")

    args = parser.parse_args()
    run_evaluation(
        split_type=args.split,
        limit=args.limit,
        model_name=args.model,
        device=args.device,
        top_k_docs=args.top_k_docs,
        top_k_sentences=args.top_k_sentences,
        output_prefix=args.prefix,
    )
