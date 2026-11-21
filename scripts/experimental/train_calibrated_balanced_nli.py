"""
ResearchMap Phase 4: Calibrated & Class-Balanced NLI Training
Addresses the 61.3% NEI prior distortion identified in Phase 3.

Technique:
1. Uses the augmented dataset (original train partition + hard-negatives).
2. Computes inverse-frequency class weights:
     w_c = N / (3 * N_c)
3. Custom training loop using weighted CrossEntropyLoss to prevent NEI logit inflation.
4. Saves checkpoint safely to 'models/experimental_calibrated_nli' (preserving baseline and augmented).
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

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
from backend.dataset_builder import (
    build_claim_evidence_pairs,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
)
from backend.trainer import (
    ClaimVerificationTrainer,
    ClaimEvidenceDataset,
    collate_fn_factory,
)


def prepare_balanced_training_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], torch.Tensor]:
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
        raise FileNotFoundError(f"Hard-negative dataset not found at {hard_neg_path}.")

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
    rng = np.random.RandomState(Config.RANDOM_SEED)
    indices = rng.permutation(len(combined_train_pairs))
    shuffled_train_pairs = [combined_train_pairs[i] for i in indices]

    # Count class frequencies: 0: CONTRADICTS, 1: SUPPORTS, 2: NEI
    class_counts = [0, 0, 0]
    for p in shuffled_train_pairs:
        class_counts[p["label_idx"]] += 1

    total_samples = len(shuffled_train_pairs)
    print(f"Total training pairs: {total_samples}")
    print(f"Class distribution: CONTRADICTS={class_counts[0]} ({class_counts[0]/total_samples*100:.1f}%), "
          f"SUPPORTS={class_counts[1]} ({class_counts[1]/total_samples*100:.1f}%), "
          f"NEI={class_counts[2]} ({class_counts[2]/total_samples*100:.1f}%)")

    # Balanced class weights: w_c = N / (num_classes * count_c)
    weights = [total_samples / (3.0 * count) for count in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"Calculated Inverse-Frequency Class Weights: {weights} (CON={weights[0]:.3f}, SUP={weights[1]:.3f}, NEI={weights[2]:.3f})")

    return shuffled_train_pairs, val_pairs, class_weights_tensor


def train_calibrated_model(
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    class_weights: torch.Tensor,
) -> Dict[str, Any]:
    output_dir = Config.MODELS_DIR / "experimental_calibrated_nli"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint = str(Config.MODELS_DIR / "finetuned_nli")
    print(f"\nInitializing trainer from base checkpoint: {base_checkpoint}")
    print(f"Target output checkpoint: {output_dir}")

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

    collate_fn = collate_fn_factory(trainer.tokenizer, max_length=trainer.max_length)
    train_loader = DataLoader(
        ClaimEvidenceDataset(train_pairs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ClaimEvidenceDataset(val_pairs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * trainer.warmup_ratio)

    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=lr,
        weight_decay=trainer.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_weights = class_weights.to(trainer.device)

    pre_metrics = trainer.evaluate_pairs(val_loader)
    print(f"Initial Zero-Shot Val Macro-F1: {pre_metrics['macro_f1']:.4f}")

    start_time = time.time()
    trainer.model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        labels = batch["labels"].to(trainer.device)
        input_ids = batch["input_ids"].to(trainer.device)
        attention_mask = batch["attention_mask"].to(trainer.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(trainer.device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = trainer.model(**kwargs)
        logits = outputs.logits

        # Weighted Cross-Entropy Loss to eliminate NEI bias
        loss = F.cross_entropy(logits, labels, weight=loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        if step % 40 == 0 or step == len(train_loader):
            print(f"  Step {step}/{len(train_loader)} - Batch Loss: {loss.item():.4f}")

    train_duration = time.time() - start_time
    avg_train_loss = epoch_loss / len(train_loader)

    post_metrics = trainer.evaluate_pairs(val_loader)
    print(f"\nTraining Complete in {train_duration:.1f}s")
    print(f"Train Loss: {avg_train_loss:.4f} | Post Val Macro-F1: {post_metrics['macro_f1']:.4f}")

    # Save checkpoint
    print(f"Saving checkpoint to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    trainer.tokenizer.save_pretrained(str(output_dir))
    print("Checkpoint saved successfully.")

    return {
        "train_loss": avg_train_loss,
        "val_metrics": post_metrics,
        "duration_seconds": train_duration,
        "class_weights": class_weights.tolist(),
    }


if __name__ == "__main__":
    train_pairs, val_pairs, weights = prepare_balanced_training_data()
    summary = train_calibrated_model(train_pairs, val_pairs, weights)
    print("Summary:", summary)
