"""
Stand-alone training script for ResearchMap Phase 2.
Fine-tunes a scientific claim verification cross-encoder model
on leakage-safe training partitions without touching the official development claims.
"""

import sys
import argparse
import json
from pathlib import Path

# Ensure workspace root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import load_corpus, load_claims
from backend.dataset_builder import build_claim_evidence_pairs
from backend.trainer import ClaimVerificationTrainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResearchMap claim verification model.")
    parser.add_argument("--model", type=str, default=Config.NLI_MODEL_NAME, help="Base pretrained model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/finetuned_nli", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")

    args = parser.parse_args()

    print("=== ResearchMap Phase 2: Model Fine-Tuning ===")
    print(f"Base model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")

    # Load corpus
    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    print(f"Loading corpus from {corpus_path}...")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus):,} corpus abstracts.")

    # Load partitions
    train_split_path = Config.PROCESSED_DATA_DIR / "train_partition.jsonl"
    val_split_path = Config.PROCESSED_DATA_DIR / "val_internal.jsonl"

    if not train_split_path.exists() or not val_split_path.exists():
        print("Error: Processed partitions not found. Run scripts/prepare_splits.py first.")
        sys.exit(1)

    print(f"Loading train partition from {train_split_path}...")
    train_claims = load_claims(train_split_path)
    print(f"Loading val partition from {val_split_path}...")
    val_claims = load_claims(val_split_path)

    # Build claim-evidence pairs
    print("\nBuilding training pairs from SciFact annotations...")
    train_pairs = build_claim_evidence_pairs(train_claims, corpus, include_distractors=True, seed=args.seed)
    val_pairs = build_claim_evidence_pairs(val_claims, corpus, include_distractors=True, seed=args.seed)

    print(f"Constructed {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")

    # Initialize trainer
    trainer = ClaimVerificationTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Run training
    summary = trainer.train(train_pairs, val_pairs)
    print("\n=== Fine-Tuning Execution Summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
