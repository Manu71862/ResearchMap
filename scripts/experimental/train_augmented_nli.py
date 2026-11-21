"""
ResearchMap Phase 3: Augmented NLI Model Training & Benchmark Harness

1. Trains an augmented NLI model checkpoint on:
   - Original train partition pairs (1,439 pairs)
   - Experimental validated hard-negatives (1,500 pairs)
   Total: 2,939 training pairs.
2. Saves checkpoint safely to 'models/experimental_augmented_nli' (preserving 'models/finetuned_nli').
3. Benchmarks Model A (Baseline finetuned_nli) vs Model B (Augmented experimental_augmented_nli) across:
   - SciFact Validation Internal (N=161)
   - Diagnostic Test New Claims (N=24)
   - Independent Generalization Challenge Set (N=60) across 6 reasoning categories.
4. Records all training history, loss, confusion matrices, per-class metrics, and latencies.
5. Saves results to 'experiments/phase3_hard_negative_results.json'.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from backend.data_loader import (
    load_corpus,
    load_claims,
    derive_gold_claim_label,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.dataset_builder import build_claim_evidence_pairs, LABEL_TO_IDX, IDX_TO_LABEL
from backend.config import Config
from backend.trainer import ClaimVerificationTrainer
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel


def load_augmented_training_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus = load_corpus(Config.RAW_DATA_DIR / "corpus.jsonl")
    train_claims = load_claims(Config.PROCESSED_DATA_DIR / "train_partition.jsonl")
    val_claims = load_claims(Config.PROCESSED_DATA_DIR / "val_internal.jsonl")

    # Build baseline training pairs
    base_train_pairs = build_claim_evidence_pairs(
        train_claims, corpus, include_distractors=True, seed=Config.RANDOM_SEED
    )
    val_pairs = build_claim_evidence_pairs(
        val_claims, corpus, include_distractors=True, seed=Config.RANDOM_SEED
    )

    # Load experimental hard-negative pairs
    hard_neg_path = Config.BASE_DIR / "data" / "experimental" / "hard_negative_pairs.jsonl"
    if not hard_neg_path.exists():
        raise FileNotFoundError(f"Hard-negative dataset not found at {hard_neg_path}. Run generate_hard_negatives.py first.")

    hard_neg_pairs = []
    with open(hard_neg_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                hard_neg_pairs.append({
                    "claim_id": item.get("source_claim_id"),
                    "claim": item["claim"],
                    "evidence_text": item["evidence_text"],
                    "doc_id": item.get("doc_id"),
                    "sentence_indices": [],
                    "label": item["label"],
                    "label_idx": item["label_idx"],
                    "is_distractor": item.get("category") == "cross_document_distractor",
                })

    combined_train_pairs = list(base_train_pairs) + hard_neg_pairs
    rng = random = np.random.RandomState(Config.RANDOM_SEED)
    indices = rng.permutation(len(combined_train_pairs))
    shuffled_train_pairs = [combined_train_pairs[i] for i in indices]

    print(f"Original train pairs: {len(base_train_pairs)}")
    print(f"Hard-negative pairs added: {len(hard_neg_pairs)}")
    print(f"Total augmented train pairs: {len(shuffled_train_pairs)}")
    print(f"Validation pairs for evaluation: {len(val_pairs)}")

    return shuffled_train_pairs, val_pairs


def train_augmented_model(train_pairs: List[Dict[str, Any]], val_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    output_dir = Config.MODELS_DIR / "experimental_augmented_nli"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint = str(Config.MODELS_DIR / "finetuned_nli")
    print(f"\nInitializing trainer with base checkpoint: {base_checkpoint}")
    print(f"Output checkpoint directory: {output_dir}")

    batch_size = 16
    epochs = 1
    lr = 1.5e-5

    trainer = ClaimVerificationTrainer(
        model_name=base_checkpoint,
        output_dir=str(output_dir),
        device="cpu",
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
        max_length=256,
        seed=Config.RANDOM_SEED,
    )

    t0 = time.time()
    training_summary = trainer.train(train_pairs, val_pairs)
    train_duration = time.time() - t0

    # Ensure researchmap_model_config.json is saved in new checkpoint
    model_cfg = {
        "model_name": "cross-encoder/nli-deberta-v3-small",
        "phase": "phase3_augmented_hard_negative",
        "parent_checkpoint": "models/finetuned_nli",
        "max_length": 256,
        "label_mapping": {0: LABEL_CONTRADICTS, 1: LABEL_SUPPORTS, 2: LABEL_NEI},
        "training_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "random_seed": Config.RANDOM_SEED,
    }
    with open(output_dir / "researchmap_model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=2)

    return {
        "training_summary": training_summary,
        "duration_seconds": train_duration,
        "output_dir": str(output_dir),
        "config": model_cfg,
    }


def evaluate_end_to_end_claims(
    model: NLIEntailmentModel,
    retriever: BM25Retriever,
    claims: List[Dict[str, Any]],
    corpus: Dict[int, Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    golds = []
    preds = []
    latencies = []

    for c in claims:
        t0 = time.perf_counter()
        claim_text = c["claim"]
        if "gold_label" in c:
            gold = c["gold_label"]
        elif "evidence" in c:
            gold = derive_gold_claim_label(c["evidence"])
        else:
            gold = LABEL_NEI
        golds.append(gold)

        candidate_docs = retriever.retrieve_documents(claim_text, top_k=Config.BM25_TOP_K_DOCS)
        candidate_sentences = retriever.retrieve_sentences(
            claim_text, candidate_docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES
        )

        if not candidate_sentences:
            preds.append(LABEL_NEI)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            continue

        pairs = [(s["text"], claim_text) for s in candidate_sentences]
        nli_outputs = model.predict_batch(pairs)

        best_sup = None
        best_con = None
        for s, out in zip(candidate_sentences, nli_outputs):
            p_sup = out["probabilities"].get(LABEL_SUPPORTS, 0.0)
            p_con = out["probabilities"].get(LABEL_CONTRADICTS, 0.0)
            if p_sup >= Config.ENTAIL_THRESHOLD:
                if best_sup is None or p_sup > best_sup["p"]:
                    best_sup = {"p": p_sup, "doc_id": s["doc_id"]}
            if p_con >= Config.CONTRADICT_THRESHOLD:
                if best_con is None or p_con > best_con["p"]:
                    best_con = {"p": p_con, "doc_id": s["doc_id"]}

        if best_sup and best_con:
            pred = LABEL_SUPPORTS if best_sup["p"] >= best_con["p"] else LABEL_CONTRADICTS
        elif best_sup:
            pred = LABEL_SUPPORTS
        elif best_con:
            pred = LABEL_CONTRADICTS
        else:
            pred = LABEL_NEI

        preds.append(pred)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    acc = float(accuracy_score(golds, preds))
    macro_f1 = float(f1_score(golds, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0))
    p, r, f1, _ = precision_recall_fscore_support(golds, preds, labels=CANONICAL_LABELS, zero_division=0)
    cm = confusion_matrix(golds, preds, labels=CANONICAL_LABELS).tolist()

    false_con = sum(1 for g, pr in zip(golds, preds) if g != LABEL_CONTRADICTS and pr == LABEL_CONTRADICTS)
    false_sup = sum(1 for g, pr in zip(golds, preds) if g != LABEL_SUPPORTS and pr == LABEL_SUPPORTS)

    return {
        "split": split_name,
        "total_claims": len(claims),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
            for i, c in enumerate(CANONICAL_LABELS)
        },
        "confusion_matrix": cm,
        "false_contradictions": false_con,
        "false_supports": false_sup,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "predictions": preds,
        "golds": golds,
    }


def evaluate_challenge_set(model: NLIEntailmentModel) -> Dict[str, Any]:
    ch_path = Config.BASE_DIR / "data" / "experimental" / "challenge_set.jsonl"
    if not ch_path.exists():
        raise FileNotFoundError(f"Challenge set not found at {ch_path}")

    pairs_data = [json.loads(line) for line in open(ch_path, encoding="utf-8") if line.strip()]
    text_pairs = [(p["evidence_text"], p["claim"]) for p in pairs_data]

    t0 = time.perf_counter()
    outputs = model.predict_batch(text_pairs)
    total_time = (time.perf_counter() - t0) * 1000.0

    golds = [p["gold_label"] for p in pairs_data]
    preds = [o["prediction"] for o in outputs]

    acc = float(accuracy_score(golds, preds))
    macro_f1 = float(f1_score(golds, preds, labels=CANONICAL_LABELS, average="macro", zero_division=0))
    cm = confusion_matrix(golds, preds, labels=CANONICAL_LABELS).tolist()

    # Per-category evaluation
    cat_results = {}
    for p, pred in zip(pairs_data, preds):
        cat = p["category"]
        cat_results.setdefault(cat, {"golds": [], "preds": []})
        cat_results[cat]["golds"].append(p["gold_label"])
        cat_results[cat]["preds"].append(pred)

    category_breakdown = {}
    for cat, data in cat_results.items():
        c_acc = float(accuracy_score(data["golds"], data["preds"]))
        c_f1 = float(f1_score(data["golds"], data["preds"], labels=CANONICAL_LABELS, average="macro", zero_division=0))
        category_breakdown[cat] = {
            "count": len(data["golds"]),
            "accuracy": c_acc,
            "macro_f1": c_f1,
            "correct": sum(1 for g, pr in zip(data["golds"], data["preds"]) if g == pr),
        }

    return {
        "total_pairs": len(pairs_data),
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "category_breakdown": category_breakdown,
        "latency_ms": {
            "total": total_time,
            "per_pair": total_time / len(pairs_data) if pairs_data else 0.0,
        },
    }


def run_phase3_pipeline():
    print("=" * 70)
    print("ResearchMap Phase 3: Scientific Hard-Negative Model Training & Benchmark")
    print("=" * 70)

    corpus = load_corpus(Config.RAW_DATA_DIR / "corpus.jsonl")
    retriever = BM25Retriever(corpus)
    val_claims = load_claims(Config.PROCESSED_DATA_DIR / "val_internal.jsonl")
    test_claims = [json.loads(l) for l in open(Config.BASE_DIR / "experiments" / "new_claims.jsonl", encoding="utf-8") if l.strip()]

    # 1. Prepare Training Data
    train_pairs, val_pairs = load_augmented_training_data()

    # 2. Train Augmented Model
    train_result = train_augmented_model(train_pairs, val_pairs)

    # 3. Load Models for Comparative Evaluation
    print("\n--- Initializing Models for Comparative Evaluation ---")
    model_base = NLIEntailmentModel(model_name=str(Config.MODELS_DIR / "finetuned_nli"), device="cpu")
    model_aug = NLIEntailmentModel(model_name=str(Config.MODELS_DIR / "experimental_augmented_nli"), device="cpu")

    # 4. Benchmark on Validation Internal (N=161)
    print("\nEvaluating Model A (Baseline) on Validation Set (N=161)...")
    val_base = evaluate_end_to_end_claims(model_base, retriever, val_claims, corpus, "Validation (Baseline)")
    print(f"Baseline Val Acc: {val_base['accuracy']*100:.2f}% | Macro-F1: {val_base['macro_f1']:.4f} | False CON: {val_base['false_contradictions']}")

    print("\nEvaluating Model B (Augmented) on Validation Set (N=161)...")
    val_aug = evaluate_end_to_end_claims(model_aug, retriever, val_claims, corpus, "Validation (Augmented)")
    print(f"Augmented Val Acc: {val_aug['accuracy']*100:.2f}% | Macro-F1: {val_aug['macro_f1']:.4f} | False CON: {val_aug['false_contradictions']}")

    # 5. Benchmark on Diagnostic New Claims (N=24)
    print("\nEvaluating Model A (Baseline) on Diagnostic Test (N=24)...")
    test_base = evaluate_end_to_end_claims(model_base, retriever, test_claims, corpus, "Diagnostic (Baseline)")
    print(f"Baseline Diag Acc: {test_base['accuracy']*100:.2f}% | Macro-F1: {test_base['macro_f1']:.4f} | False CON: {test_base['false_contradictions']}")

    print("\nEvaluating Model B (Augmented) on Diagnostic Test (N=24)...")
    test_aug = evaluate_end_to_end_claims(model_aug, retriever, test_claims, corpus, "Diagnostic (Augmented)")
    print(f"Augmented Diag Acc: {test_aug['accuracy']*100:.2f}% | Macro-F1: {test_aug['macro_f1']:.4f} | False CON: {test_aug['false_contradictions']}")

    # 6. Benchmark on Challenge Set (N=60)
    print("\nEvaluating Model A (Baseline) on Challenge Set (N=60)...")
    ch_base = evaluate_challenge_set(model_base)
    print(f"Baseline Challenge Acc: {ch_base['overall_accuracy']*100:.2f}% | Macro-F1: {ch_base['macro_f1']:.4f}")

    print("\nEvaluating Model B (Augmented) on Challenge Set (N=60)...")
    ch_aug = evaluate_challenge_set(model_aug)
    print(f"Augmented Challenge Acc: {ch_aug['overall_accuracy']*100:.2f}% | Macro-F1: {ch_aug['macro_f1']:.4f}")

    # 7. Collect Full Results
    final_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_experiment": train_result,
        "dataset_composition": {
            "original_train_pairs": 1439,
            "hard_negative_pairs": 1500,
            "total_augmented_train_pairs": len(train_pairs),
            "label_distribution": {
                "SUPPORTS": sum(1 for p in train_pairs if p["label"] == LABEL_SUPPORTS),
                "CONTRADICTS": sum(1 for p in train_pairs if p["label"] == LABEL_CONTRADICTS),
                "NOT ENOUGH INFORMATION": sum(1 for p in train_pairs if p["label"] == LABEL_NEI),
            },
        },
        "validation_benchmark_n161": {
            "baseline": val_base,
            "augmented": val_aug,
        },
        "diagnostic_benchmark_n24": {
            "baseline": test_base,
            "augmented": test_aug,
        },
        "generalization_challenge_n60": {
            "baseline": ch_base,
            "augmented": ch_aug,
        },
    }

    out_json = Config.BASE_DIR / "experiments" / "phase3_hard_negative_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved complete Phase 3 benchmark results to: {out_json}")
    return final_output


if __name__ == "__main__":
    run_phase3_pipeline()
