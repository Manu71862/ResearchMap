"""
Trainer module for Scientific Claim Verification fine-tuning.
Trains a sequence classification / cross-encoder model on claim-evidence pairs
with early stopping, validation tracking, dynamic padding, and checkpoint management.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from backend.dataset_builder import LABEL_TO_IDX, IDX_TO_LABEL
from backend.data_loader import CANONICAL_LABELS, LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI

class ClaimEvidenceDataset(Dataset):
    """PyTorch Dataset for (evidence, claim) text pairs and classification labels."""

    def __init__(self, pairs: List[Dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        return {
            "evidence": item["evidence_text"],
            "claim": item["claim"],
            "label_idx": item["label_idx"],
        }

def collate_fn_factory(tokenizer, max_length: int = 256):
    """Creates a collator with dynamic padding."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        premises = [b["evidence"] for b in batch]
        hypotheses = [b["claim"] for b in batch]
        labels = [b["label_idx"] for b in batch]

        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate_fn

class ClaimVerificationTrainer:
    """Handles end-to-end fine-tuning and checkpointing."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        output_dir: str = "models/finetuned_nli",
        device: str = "auto",
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_length: int = 256,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_length = max_length
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing Trainer for '{model_name}' on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: LABEL_CONTRADICTS, 1: LABEL_SUPPORTS, 2: LABEL_NEI},
            label2id={LABEL_CONTRADICTS: 0, LABEL_SUPPORTS: 1, LABEL_NEI: 2},
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

    def evaluate_pairs(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on a pair dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                outputs = self.model(**kwargs)
                loss = outputs.loss
                total_loss += loss.item() * len(labels)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        per_class_f1 = {}
        for idx, lbl in IDX_TO_LABEL.items():
            f1 = f1_score([1 if y == idx else 0 for y in all_labels],
                          [1 if p == idx else 0 for p in all_preds],
                          zero_division=0)
            per_class_f1[lbl] = float(f1)

        return {
            "loss": avg_loss,
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_f1": per_class_f1,
        }

    def train(
        self,
        train_pairs: List[Dict[str, Any]],
        val_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run fine-tuning with training history and best-model checkpointing."""
        collate_fn = collate_fn_factory(self.tokenizer, max_length=self.max_length)
        train_loader = DataLoader(
            ClaimEvidenceDataset(train_pairs),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            ClaimEvidenceDataset(val_pairs),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"Starting fine-tuning: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs.")
        print(f"Total steps: {total_steps} (Warmup: {warmup_steps}, Epochs: {self.epochs}, Batch size: {self.batch_size})")

        start_time = time.time()
        best_val_f1 = -1.0
        best_epoch = -1
        training_history = []

        # Zero-shot validation performance before training
        pre_val_metrics = self.evaluate_pairs(val_loader)
        print(f"Initial Zero-Shot Val Macro-F1: {pre_val_metrics['macro_f1']:.4f} (Loss: {pre_val_metrics['loss']:.4f})")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_t0 = time.time()

            for step, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                labels = batch["labels"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                outputs = self.model(**kwargs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            val_metrics = self.evaluate_pairs(val_loader)
            epoch_time = time.time() - epoch_t0

            print(
                f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Val Per-Class: {val_metrics['per_class_f1']}"
            )

            history_entry = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_per_class_f1": val_metrics["per_class_f1"],
                "epoch_time_seconds": epoch_time,
            }
            training_history.append(history_entry)

            # Save best checkpoint
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                self.save_checkpoint(self.output_dir, is_best=True)
                print(f"  -> Saved new best checkpoint (Val Macro-F1: {best_val_f1:.4f}) to {self.output_dir.resolve()}")

        total_duration = time.time() - start_time
        print(f"\nFine-tuning completed in {total_duration:.1f}s. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

        # Save training summary
        train_summary = {
            "model_name": self.model_name,
            "device": str(self.device),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "total_duration_seconds": total_duration,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "initial_zero_shot_val_macro_f1": pre_val_metrics["macro_f1"],
            "training_history": training_history,
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(train_summary, f, indent=2)

        return train_summary

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model weights, tokenizer, and label mappings."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        metadata = {
            "base_model": self.model_name,
            "num_labels": 3,
            "id2label": {0: LABEL_CONTRADICTS, 1: LABEL_SUPPORTS, 2: LABEL_NEI},
            "label2id": {LABEL_CONTRADICTS: 0, LABEL_SUPPORTS: 1, LABEL_NEI: 2},
            "saved_at": time.time(),
        }
        with open(path / "researchmap_model_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
