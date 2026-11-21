"""
Unit and integration tests for Condition-Aware Reversibility Filtering in ResearchMap.

Tests:
1. is_exposure_claim classification precision (positives and negatives).
2. is_cessation_reversibility_sentence detection precision.
3. Target smoking claim end-to-end verification (SUPPORTS, Doc 26067999 Sent 40).
4. Evidence preservation: Sentence 57 visible with original CONTRADICTS score and rationale.
5. Preservation of legitimate contradictions (cessation, withdrawal, direct disproof).
6. Probability immutability (unmutated distributions summing to ~1.0).
"""

import pytest
from pathlib import Path

from backend.config import Config
from backend.data_loader import (
    load_corpus,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.pipeline import (
    ClaimVerificationPipeline,
    is_exposure_claim,
    is_cessation_reversibility_sentence,
)

@pytest.fixture(scope="module")
def pipeline():
    """Load production pipeline fixture."""
    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    corpus = load_corpus(corpus_path)
    retriever = BM25Retriever(corpus)
    finetuned_dir = Config.MODELS_DIR / "finetuned_nli"
    model_path = str(finetuned_dir) if finetuned_dir.exists() else Config.NLI_MODEL_NAME
    entailment_model = NLIEntailmentModel(model_name=model_path, device=Config.DEVICE)
    return ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=entailment_model,
        top_k_docs=Config.BM25_TOP_K_DOCS,
        top_k_sentences=Config.BM25_TOP_K_SENTENCES,
    )

def test_is_exposure_claim_classifier():
    # Positive exposure claims (E -> +Y)
    assert is_exposure_claim("Smoking increases the risk of developing lung cancer.") is True
    assert is_exposure_claim("Radon exposure elevates the incidence of pulmonary malignancies.") is True
    assert is_exposure_claim("High dietary sodium intake promotes arterial hypertension.") is True
    assert is_exposure_claim("Asbestos fiber inhalation causes malignant mesothelioma.") is True

    # Negative: Claims with cessation/withdrawal/discontinuation modifiers
    assert is_exposure_claim("Smoking cessation increases the risk of lung cancer.") is False
    assert is_exposure_claim("Quitting smoking worsens cardiovascular outcomes.") is False
    assert is_exposure_claim("Abrupt withdrawal of beta-blockers reduces arrhythmias.") is False
    assert is_exposure_claim("Treatment discontinuation prevents tumor recurrence.") is False
    assert is_exposure_claim("Alcohol abstinence accelerates hepatic fibrosis.") is False
    assert is_exposure_claim("Stopping statin therapy lowers cardiac mortality.") is False

    # Negative: Non-exposure biomedical statements
    assert is_exposure_claim("BRCA1 is a human tumor suppressor gene.") is False
    assert is_exposure_claim("Insulin binds to receptor tyrosine kinases.") is False
    assert is_exposure_claim("Penicillin inhibits peptidoglycan cross-linking.") is False
    assert is_exposure_claim("Polymerase chain reaction amplifies specific DNA sequences.") is False

def test_is_cessation_reversibility_sentence():
    # Positive reversibility sentences (Cessation(E) -> -Y)
    assert is_cessation_reversibility_sentence(
        "Smoking cessation substantially reduces a person's risk for developing and dying of lung cancer."
    ) is True
    assert is_cessation_reversibility_sentence(
        "Quitting smoking rapidly lowers the incidence of myocardial infarction."
    ) is True
    assert is_cessation_reversibility_sentence(
        "Alcohol abstinence halts the progression of hepatic cirrhosis."
    ) is True
    assert is_cessation_reversibility_sentence(
        "Stopping statin therapy decreases LDL clearance."
    ) is True

    # Negative: Exposure without cessation
    assert is_cessation_reversibility_sentence(
        "The risk for lung cancer increases with age and cumulative exposure to tobacco smoke."
    ) is False
    assert is_cessation_reversibility_sentence(
        "Tobacco smoking does not increase lung cancer incidence."
    ) is False

    # Negative: Cessation without reduction predicate
    assert is_cessation_reversibility_sentence(
        "Smoking cessation increases transient appetite and weight gain."
    ) is False

def test_target_smoking_claim_prediction(pipeline):
    """Verify that the target smoking claim correctly outputs SUPPORTS citing Doc 26067999 Sent 40."""
    claim = "Smoking increases the risk of developing lung cancer."
    result = pipeline.verify_claim(claim)

    assert result["predicted_label"] == LABEL_SUPPORTS
    assert result["confidence"] >= 0.85
    assert result["reversibility_detected"] is True

    # Check winning evidence
    assert 26067999 in result["predicted_evidence_doc_ids"]
    assert (26067999, 40) in result["predicted_evidence_sentences"]

    # Verify that the cessation sentence (Sentence 57) is preserved in evidence_details
    sent_57 = next((e for e in result["evidence_details"] if e["doc_id"] == 26067999 and e["sentence_idx"] == 57), None)
    assert sent_57 is not None, "Sentence 57 must be present in evidence_details"
    assert sent_57["veto_masked"] is True
    assert sent_57["reversibility_rationale"] is not None
    assert "Hill's Reversibility Criterion" in sent_57["reversibility_rationale"]
    # Verify raw neural probabilities are untouched
    assert sent_57["probabilities"][LABEL_CONTRADICTS] > 0.95

    # Verify winning evidence sentence details (Sentence 40)
    sent_40 = next((e for e in result["evidence_details"] if e["doc_id"] == 26067999 and e["sentence_idx"] == 40), None)
    assert sent_40 is not None
    assert sent_40["probabilities"][LABEL_SUPPORTS] >= 0.85
    # Sentence 40 contains both cumulative exposure increase and 'quitting' decrease, so veto_masked is True
    assert sent_40["veto_masked"] is True

    # Verify that a sentence without cessation/quitting has veto_masked is False
    non_rev_sent = next((e for e in result["evidence_details"] if "quitting" not in e["text"].lower() and "cessation" not in e["text"].lower()), None)
    if non_rev_sent:
        assert non_rev_sent["veto_masked"] is False

def test_legitimate_contradiction_preservation(pipeline):
    """
    Verify that legitimate contradictions are NEVER suppressed.
    When a claim asserts that cessation increases risk, evidence that cessation reduces risk
    must be classified as CONTRADICTS.
    """
    claim = "Smoking cessation increases the risk of lung cancer."
    evidence_sent = "Smoking cessation substantially reduces a person's risk for developing and dying of lung cancer."

    # Direct NLI check
    nli_out = pipeline.entailment_model.predict_pair(evidence_sent, claim)
    assert nli_out["prediction"] == LABEL_CONTRADICTS
    assert nli_out["probabilities"][LABEL_CONTRADICTS] > 0.90

    # Pipeline classifier check: claim is NOT an exposure claim
    assert is_exposure_claim(claim) is False

    # Pipeline aggregation check with candidate detail
    candidate = {
        "doc_id": 26067999,
        "sentence_idx": 57,
        "text": evidence_sent,
        "title": "Screening for Lung Cancer",
        "bm25_score": 10.0,
        "entailment_label": nli_out["prediction"],
        "confidence": nli_out["confidence"],
        "probabilities": dict(nli_out["probabilities"]),
    }

    # Verify veto is NOT masked when claim has cessation modifier
    claim_is_exp = is_exposure_claim(claim)
    is_rev = claim_is_exp and is_cessation_reversibility_sentence(evidence_sent)
    assert is_rev is False, "Veto must NOT be masked for legitimate contradiction"

def test_direct_exposure_disproof_preservation(pipeline):
    """
    Verify that an active exposure claim paired with direct contradictory evidence
    (without cessation terms) preserves CONTRADICTS.
    """
    claim = "Smoking increases the risk of developing lung cancer."
    direct_disproof = "Rigorous epidemiological studies demonstrate that tobacco smoking does not increase lung cancer incidence."

    # Direct disproof does not contain cessation terms
    assert is_cessation_reversibility_sentence(direct_disproof) is False

    # Therefore reversibility veto is NOT masked
    claim_is_exp = is_exposure_claim(claim)
    is_rev = claim_is_exp and is_cessation_reversibility_sentence(direct_disproof)
    assert is_rev is False

def test_probability_immutability(pipeline):
    """Verify that all probability distributions in evidence details sum to ~1.0."""
    claim = "Smoking increases the risk of developing lung cancer."
    result = pipeline.verify_claim(claim)

    for ev in result["evidence_details"]:
        probs = ev["probabilities"]
        p_sum = sum(probs.values())
        assert p_sum == pytest.approx(1.0, abs=1e-4), f"Probabilities for sentence {ev['sentence_idx']} must sum to 1.0"
        for label in CANONICAL_LABELS:
            assert label in probs
            assert 0.0 <= probs[label] <= 1.0
