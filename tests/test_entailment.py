import pytest
from unittest.mock import MagicMock
import torch

from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
)
from backend.pipeline import ClaimVerificationPipeline
from backend.retrieval import BM25Retriever

def test_pipeline_decision_logic():
    # Setup dummy corpus and retriever
    corpus = {
        1: {
            "doc_id": 1,
            "title": "Aspirin in Heart Disease",
            "abstract": [
                "Aspirin significantly reduces myocardial infarction risk.",
                "Adverse bleeding events were rare."
            ]
        }
    }
    retriever = BM25Retriever(corpus)

    # Mock entailment model
    mock_model = MagicMock()
    # Case 1: Entailment returns high support
    mock_model.predict_batch.return_value = [
        {
            "prediction": LABEL_SUPPORTS,
            "confidence": 0.92,
            "probabilities": {
                LABEL_SUPPORTS: 0.92,
                LABEL_CONTRADICTS: 0.03,
                LABEL_NEI: 0.05
            }
        }
    ]

    pipeline = ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=mock_model,
        top_k_docs=1,
        top_k_sentences=1,
        entail_threshold=0.5,
        contradict_threshold=0.5
    )

    res = pipeline.verify_claim("Aspirin reduces heart attack risk")
    assert res["predicted_label"] == LABEL_SUPPORTS
    assert res["confidence"] == 0.92
    assert res["predicted_evidence_doc_ids"] == [1]
    assert res["predicted_evidence_sentences"] == [(1, 0)]

    # Case 2: Neutral returns NEI
    mock_model.predict_batch.return_value = [
        {
            "prediction": LABEL_NEI,
            "confidence": 0.85,
            "probabilities": {
                LABEL_SUPPORTS: 0.05,
                LABEL_CONTRADICTS: 0.10,
                LABEL_NEI: 0.85
            }
        }
    ]

    res_nei = pipeline.verify_claim("Aspirin cures common cold")
    assert res_nei["predicted_label"] == LABEL_NEI
    assert res_nei["predicted_evidence_doc_ids"] == []
    assert res_nei["predicted_evidence_sentences"] == []

def test_pipeline_empty_retrieval():
    corpus = {}
    retriever = BM25Retriever(corpus)
    mock_model = MagicMock()

    pipeline = ClaimVerificationPipeline(retriever=retriever, entailment_model=mock_model)
    res = pipeline.verify_claim("Any random query with empty corpus")
    assert res["predicted_label"] == LABEL_NEI
    assert res["confidence"] == 0.0
    assert res["predicted_evidence_doc_ids"] == []
