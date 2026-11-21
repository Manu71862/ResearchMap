"""
Comprehensive Unit Tests for ResearchMap Phase 2.
Covers:
- Leakage checks and duplicate purging
- Claim-to-label mapping
- Evidence-example construction & distractor safety
- Training/inference label consistency
- Checkpoint save and reload
- Metric calculations
- Empty or missing evidence handling
- Reproducible evaluation
"""

import pytest
import json
import torch
from pathlib import Path
from unittest.mock import MagicMock

from backend.data_loader import (
    load_claims,
    load_corpus,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.dataset_builder import (
    build_claim_evidence_pairs,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
)
from backend.entailment import NLIEntailmentModel
from backend.retrieval import BM25Retriever
from backend.pipeline import ClaimVerificationPipeline
from backend.evaluation import evaluate_claim_verification
from scripts.prepare_splits import verify_partition_leakage

def test_leakage_checks_clean():
    """Verify that our partitions have zero ID or text leakage."""
    processed_dir = Path("data/processed")
    if not (processed_dir / "train_partition.jsonl").exists():
        pytest.skip("Processed partitions not found.")

    train_claims = load_claims(processed_dir / "train_partition.jsonl")
    val_claims = load_claims(processed_dir / "val_internal.jsonl")
    dev_claims = load_claims(processed_dir / "dev_heldout.jsonl")

    audit = verify_partition_leakage(train_claims, val_claims, dev_claims)
    assert audit["is_leakage_free"] is True
    assert audit["id_overlaps"]["train_val"] == 0
    assert audit["id_overlaps"]["train_dev"] == 0
    assert audit["id_overlaps"]["val_dev"] == 0
    assert audit["text_overlaps"]["train_val"] == 0
    assert audit["text_overlaps"]["train_dev"] == 0
    assert audit["text_overlaps"]["val_dev"] == 0

def test_leakage_checks_catches_overlap():
    """Verify that verify_partition_leakage detects deliberate leaks."""
    c1 = [{"id": 1, "claim": "Quantum dots fluoresce brightly."}]
    c2 = [{"id": 2, "claim": "quantum dots fluoresce brightly."}]  # Same text, different ID
    c3 = [{"id": 3, "claim": "Completely unrelated claim."}]

    audit = verify_partition_leakage(c1, c2, c3)
    assert audit["is_leakage_free"] is False
    assert audit["text_overlaps"]["train_val"] == 1

def test_evidence_example_construction_and_distractor_safety():
    """
    Verify claim-evidence construction:
    1. Multi-sentence evidence is concatenated in order.
    2. Distractor sentences are NEVER marked as CONTRADICTS; they must be NEI.
    """
    dummy_corpus = {
        500: {
            "doc_id": 500,
            "title": "Aspirin Mechanics",
            "abstract": [
                "Sentence zero is background.",
                "Aspirin inhibits COX-1 enzyme.",
                "Inhibition reduces platelet aggregation.",
                "Patients reported no severe side effects."
            ]
        }
    }
    dummy_claims = [
        {
            "id": 10,
            "claim": "Aspirin blocks COX-1 and stops platelet clotting.",
            "evidence_raw": {
                "500": [{"sentences": [2, 1], "label": "SUPPORT"}]
            },
            "cited_doc_ids": [500]
        },
        {
            "id": 20,
            "claim": "Aspirin causes immediate cardiac arrest.",
            "evidence_raw": {
                "500": [{"sentences": [3], "label": "CONTRADICT"}]
            },
            "cited_doc_ids": [500]
        },
        {
            "id": 30,
            "claim": "Aspirin cures stage 4 pancreatic cancer.",
            "evidence_raw": {},
            "cited_doc_ids": [500]
        }
    ]

    pairs = build_claim_evidence_pairs(dummy_claims, dummy_corpus, include_distractors=True, seed=42)
    assert len(pairs) >= 3

    # Check multi-sentence concatenation order (sentence 1 + sentence 2)
    sup_pair = next(p for p in pairs if p["claim_id"] == 10 and not p["is_distractor"])
    assert sup_pair["label"] == LABEL_SUPPORTS
    assert sup_pair["label_idx"] == 1
    assert "Aspirin inhibits COX-1 enzyme. Inhibition reduces platelet aggregation." in sup_pair["evidence_text"]

    # Check contradict pair
    con_pair = next(p for p in pairs if p["claim_id"] == 20 and not p["is_distractor"])
    assert con_pair["label"] == LABEL_CONTRADICTS
    assert con_pair["label_idx"] == 0

    # Check that ALL distractor pairs are NEI and NOT CONTRADICTS
    distractors = [p for p in pairs if p["is_distractor"]]
    assert len(distractors) > 0
    for d in distractors:
        assert d["label"] == LABEL_NEI
        assert d["label_idx"] == 2
        assert d["label"] != LABEL_CONTRADICTS

def test_training_inference_label_consistency():
    """Verify exact index and canonical label correspondence."""
    assert LABEL_TO_IDX[LABEL_CONTRADICTS] == 0
    assert LABEL_TO_IDX[LABEL_SUPPORTS] == 1
    assert LABEL_TO_IDX[LABEL_NEI] == 2

    assert IDX_TO_LABEL[0] == LABEL_CONTRADICTS
    assert IDX_TO_LABEL[1] == LABEL_SUPPORTS
    assert IDX_TO_LABEL[2] == LABEL_NEI

def test_checkpoint_save_and_reload(tmp_path: Path):
    """Verify that a model checkpoint can be saved and loaded by NLIEntailmentModel."""
    # Use small pretrained base model for fast test
    base_model = "cross-encoder/nli-deberta-v3-small"
    model = NLIEntailmentModel(model_name=base_model, device="cpu")

    # Save to tmp directory
    ckpt_dir = tmp_path / "test_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(ckpt_dir)
    model.tokenizer.save_pretrained(ckpt_dir)

    # Reload from local checkpoint directory
    reloaded_model = NLIEntailmentModel(model_name=str(ckpt_dir), device="cpu")
    assert reloaded_model.label_mapping[0] == LABEL_CONTRADICTS
    assert reloaded_model.label_mapping[1] == LABEL_SUPPORTS
    assert reloaded_model.label_mapping[2] == LABEL_NEI

    # Run inference test on reloaded model
    out = reloaded_model.predict_pair("Smoking causes lung cancer.", "Smoking increases lung cancer risk.")
    assert "prediction" in out
    assert "confidence" in out
    assert out["prediction"] in CANONICAL_LABELS

def test_empty_and_missing_evidence_pipeline():
    """Verify pipeline handles claims with zero evidence or empty documents."""
    corpus = {}
    retriever = BM25Retriever(corpus)
    mock_model = MagicMock()

    pipeline = ClaimVerificationPipeline(retriever=retriever, entailment_model=mock_model)
    res = pipeline.verify_claim("Completely unbacked claim with zero docs")
    assert res["predicted_label"] == LABEL_NEI
    assert res["confidence"] == 0.0
    assert res["evidence_details"] == []

def test_reproducible_evaluation():
    """Verify that evaluating the same predictions produces identical metrics."""
    gold = [LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI]
    pred = [LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI]

    res1 = evaluate_claim_verification(gold, pred)
    res2 = evaluate_claim_verification(gold, pred)

    assert res1["macro_f1"] == res2["macro_f1"] == 1.0
    assert res1["weighted_f1"] == res2["weighted_f1"] == 1.0
    assert res1["confusion_matrix"] == res2["confusion_matrix"]
