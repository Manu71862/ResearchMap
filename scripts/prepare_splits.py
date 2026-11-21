"""
Script to create a leakage-safe, reproducible Train / Internal-Val partition
from ONLY the official labelled SciFact training data (claims_train.jsonl).

Leakage-Safety Guarantees:
1. Only official claims_train.jsonl is used for training and internal validation.
2. Official development claims (claims_dev.jsonl) are strictly EXCLUDED from all training pools.
3. Exact internal duplicate claims within claims_train are deduplicated.
4. Any claims overlapping with claims_dev.jsonl are eliminated from the training pool.
5. Zero claim overlap (ID or normalized text) across train_partition, val_internal, and dev_heldout.
6. Fixed random seed (42) and stratified class sampling.
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Ensure workspace root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import (
    load_claims,
    get_split_statistics,
    CANONICAL_LABELS,
)

def verify_partition_leakage(
    train_claims: List[Dict[str, Any]],
    val_claims: List[Dict[str, Any]],
    dev_claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Formally verify that there is zero overlap in IDs or normalized claim text across partitions.
    """
    train_ids = {c["id"] for c in train_claims}
    val_ids = {c["id"] for c in val_claims}
    dev_ids = {c["id"] for c in dev_claims}

    train_texts = {c["claim"].strip().lower() for c in train_claims}
    val_texts = {c["claim"].strip().lower() for c in val_claims}
    dev_texts = {c["claim"].strip().lower() for c in dev_claims}

    train_val_id_overlap = len(train_ids.intersection(val_ids))
    train_dev_id_overlap = len(train_ids.intersection(dev_ids))
    val_dev_id_overlap = len(val_ids.intersection(dev_ids))

    train_val_text_overlap = len(train_texts.intersection(val_texts))
    train_dev_text_overlap = len(train_texts.intersection(dev_texts))
    val_dev_text_overlap = len(val_texts.intersection(dev_texts))

    is_leakage_free = (
        train_val_id_overlap == 0 and
        train_dev_id_overlap == 0 and
        val_dev_id_overlap == 0 and
        train_val_text_overlap == 0 and
        train_dev_text_overlap == 0 and
        val_dev_text_overlap == 0
    )

    return {
        "is_leakage_free": is_leakage_free,
        "id_overlaps": {
            "train_val": train_val_id_overlap,
            "train_dev": train_dev_id_overlap,
            "val_dev": val_dev_id_overlap,
        },
        "text_overlaps": {
            "train_val": train_val_text_overlap,
            "train_dev": train_dev_text_overlap,
            "val_dev": val_dev_text_overlap,
        }
    }

def prepare_leakage_safe_splits(
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
) -> Dict[str, Any]:
    Config.ensure_directories()
    raw_dir = Config.RAW_DATA_DIR
    processed_dir = Config.PROCESSED_DATA_DIR

    train_raw_claims = load_claims(raw_dir / "claims_train.jsonl")
    dev_raw_claims = load_claims(raw_dir / "claims_dev.jsonl")

    print(f"=== Leakage-Safe Split Preparation ===")
    print(f"Raw official training claims: {len(train_raw_claims)}")
    print(f"Raw official development claims: {len(dev_raw_claims)}")

    # Index official dev claims by normalized text to prevent cross-split leakage
    dev_normalized_texts = {c["claim"].strip().lower() for c in dev_raw_claims}

    # Deduplicate within train and purge any overlap with dev
    seen_train_texts = set()
    eligible_train_claims = []
    internal_duplicates = 0
    cross_dev_overlaps = 0

    for c in train_raw_claims:
        norm_text = c["claim"].strip().lower()
        if norm_text in dev_normalized_texts:
            cross_dev_overlaps += 1
            continue
        if norm_text in seen_train_texts:
            internal_duplicates += 1
            continue
        seen_train_texts.add(norm_text)
        eligible_train_claims.append(c)

    print(f"Purged {internal_duplicates} internal duplicate claims from training pool.")
    print(f"Purged {cross_dev_overlaps} claims overlapping with official dev set.")
    print(f"Total eligible leakage-safe training claims: {len(eligible_train_claims)}")

    # Stratified split by label
    by_label = defaultdict(list)
    for c in eligible_train_claims:
        by_label[c["gold_label"]].append(c)

    rng = random.Random(seed)
    train_partition = []
    val_internal = []

    for lbl, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        train_partition.extend(group[:n_train])
        val_internal.extend(group[n_train:])

    rng.shuffle(train_partition)
    rng.shuffle(val_internal)

    # Formally verify leakage
    leakage_audit = verify_partition_leakage(train_partition, val_internal, dev_raw_claims)
    if not leakage_audit["is_leakage_free"]:
        raise ValueError(f"CRITICAL: Data leakage detected during partitioning! {leakage_audit}")
    print("Leakage audit: PASSED. Zero claim text or ID overlap across all partitions.")

    # Save to data/processed
    files_to_save = {
        "train_partition.jsonl": train_partition,
        "val_internal.jsonl": val_internal,
    }

    summary = {
        "random_seed": seed,
        "protocol": "leakage_safe_training_only",
        "raw_train_count": len(train_raw_claims),
        "raw_dev_count": len(dev_raw_claims),
        "internal_duplicates_removed": internal_duplicates,
        "cross_dev_overlaps_removed": cross_dev_overlaps,
        "total_eligible_training_claims": len(eligible_train_claims),
        "leakage_audit": leakage_audit,
        "splits": {}
    }

    for filename, split_data in files_to_save.items():
        out_path = processed_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                rec = {
                    "id": item["id"],
                    "claim": item["claim"],
                    "evidence": item["evidence_raw"],
                    "cited_doc_ids": item["cited_doc_ids"],
                }
                f.write(json.dumps(rec) + "\n")
        stats = get_split_statistics(split_data)
        summary["splits"][filename] = stats
        print(f"\nSaved '{filename}': {stats['total_claims']} claims")
        for lbl in CANONICAL_LABELS:
            cnt = stats["label_distribution"][lbl]
            pct = stats["label_percentages"][lbl]
            print(f"  {lbl:<25}: {cnt:>4} ({pct:.1f}%)")

    # Save dev_heldout reference copy for evaluation
    dev_out_path = processed_dir / "dev_heldout.jsonl"
    with open(dev_out_path, "w", encoding="utf-8") as f:
        for item in dev_raw_claims:
            rec = {
                "id": item["id"],
                "claim": item["claim"],
                "evidence": item["evidence_raw"],
                "cited_doc_ids": item["cited_doc_ids"],
            }
            f.write(json.dumps(rec) + "\n")
    summary["splits"]["dev_heldout.jsonl"] = get_split_statistics(dev_raw_claims)
    print(f"\nSaved 'dev_heldout.jsonl': {len(dev_raw_claims)} claims (Strictly Held-Out Benchmark)")

    # Save metadata
    meta_path = processed_dir / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit metadata saved to {meta_path.resolve()}")

    return summary

if __name__ == "__main__":
    prepare_leakage_safe_splits(seed=Config.RANDOM_SEED)
