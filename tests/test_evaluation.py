import pytest
from backend.evaluation import (
    evaluate_retrieval,
    evaluate_sentence_evidence,
    evaluate_claim_verification,
)
from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)

def test_evaluate_retrieval():
    claims = [
        {"gold_doc_evidence": {101}},
        {"gold_doc_evidence": {202}},
        {"gold_doc_evidence": set()},  # NEI claim, should not count towards recall
    ]
    retrieved = [
        [101, 102, 103],  # Hit for 101
        [999, 888, 777],  # Miss for 202
        [555, 666, 777],
    ]

    metrics = evaluate_retrieval(claims, retrieved, k_list=[1, 3])
    assert metrics["eligible_claims_with_evidence"] == 2
    assert metrics["doc_recall@1"] == 0.5  # 1/2
    assert metrics["doc_recall@3"] == 0.5  # 1/2

def test_evaluate_sentence_evidence():
    claims = [
        {"gold_sentence_evidence": {(101, 0), (101, 1)}},
        {"gold_sentence_evidence": {(202, 3)}},
    ]
    # Prediction: claim 0 got (101, 0) right and added false positive (101, 2)
    # claim 1 got (202, 3) right
    predicted = [
        [(101, 0), (101, 2)],
        [(202, 3)],
    ]
    # TP = 2 ((101, 0) and (202, 3))
    # FP = 1 ((101, 2))
    # FN = 1 ((101, 1))
    metrics = evaluate_sentence_evidence(claims, predicted)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    # P = 2 / 3 = 0.6667, R = 2 / 3 = 0.6667, F1 = 0.6667
    assert pytest.approx(metrics["sentence_evidence_precision"], 0.001) == 2 / 3
    assert pytest.approx(metrics["sentence_evidence_recall"], 0.001) == 2 / 3
    assert pytest.approx(metrics["sentence_evidence_f1"], 0.001) == 2 / 3

def test_evaluate_claim_verification():
    gold = [LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI, LABEL_SUPPORTS]
    pred = [LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_SUPPORTS, LABEL_SUPPORTS]

    metrics = evaluate_claim_verification(gold, pred)
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics

    # Per class checks
    # CONTRADICTS: 1 gold, 1 pred -> P=1.0, R=1.0, F1=1.0
    assert metrics["per_class"][LABEL_CONTRADICTS]["f1_score"] == 1.0
    # NEI: 1 gold, 0 pred -> P=0.0, R=0.0, F1=0.0
    assert metrics["per_class"][LABEL_NEI]["f1_score"] == 0.0

    cm = metrics["confusion_matrix"]["matrix"]
    assert len(cm) == 3
    assert len(cm[0]) == 3
