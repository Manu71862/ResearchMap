"""
Phase 4A Regression Test Suite: Claim Verdicts, Label Mapping, and Evidence Selection.

Tests:
1. Model output indices map strictly to intended canonical labels:
   0: CONTRADICTS, 1: SUPPORTS, 2: NOT ENOUGH INFORMATION.
2. The API loads and uses the fine-tuned checkpoint (not baseline or unintended model).
3. Gold labels and predicted labels remain strictly distinct in API requests/responses.
4. Empty, irrelevant, or nonsense queries are handled honestly as NOT ENOUGH INFORMATION.
5. Benchmark sample verification reports gold labels and highlights divergences transparently.
6. Real-model verification on Claim 49 yields SUPPORTS with Doc 5953485 sentence 2.
7. Document-weighted sentence retrieval prioritizes sentences from top BM25 documents.
8. Pipeline and checkpoint continue to work with valid probability distributions.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.config import Config
from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.entailment import NLIEntailmentModel
from backend.server import app, app_state
from backend.retrieval import BM25Retriever


@pytest.fixture(scope="module")
def client():
    """Create test client using FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_label_mapping_integrity():
    """Verify that model output indices map strictly to the canonical 3-class schema."""
    # Check NLIEntailmentModel default mapping
    expected_mapping = {
        0: LABEL_CONTRADICTS,
        1: LABEL_SUPPORTS,
        2: LABEL_NEI,
    }
    assert NLIEntailmentModel.LABEL_MAPPING == expected_mapping, (
        f"NLIEntailmentModel.LABEL_MAPPING mismatch: {NLIEntailmentModel.LABEL_MAPPING}"
    )

    # Check checkpoint config.json if fine-tuned model exists
    finetuned_config_path = Config.MODELS_DIR / "finetuned_nli" / "config.json"
    if finetuned_config_path.exists():
        with open(finetuned_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        # JSON keys are strings
        for idx, label in expected_mapping.items():
            assert id2label.get(str(idx)) == label, (
                f"Checkpoint config id2label[{idx}] expected {label}, got {id2label.get(str(idx))}"
            )


@pytest.mark.skipif(
    not (Config.MODELS_DIR / "finetuned_nli" / "model.safetensors").exists(),
    reason="Production fine-tuned checkpoint (models/finetuned_nli/model.safetensors) not found on disk. Run 'python scripts/train.py --model cross-encoder/nli-deberta-v3-small --output-dir models/finetuned_nli' to generate it."
)
def test_api_loads_finetuned_checkpoint(client: TestClient):
    """Verify the API loads models/finetuned_nli rather than an unintended checkpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["model_loaded"] is True
    assert "finetuned_nli" in str(data["model_name"]).lower(), (
        f"API should load finetuned_nli checkpoint, but reported: {data['model_name']}"
    )


def test_gold_and_predicted_labels_remain_separate(client: TestClient):
    """Verify gold labels and predicted labels are returned as distinct fields and not overwritten."""
    payload = {
        "claim_id": 49,
        "claim": "ADAR1 binds to Dicer to cleave pre-miRNA.",
    }
    res = client.post("/api/verify", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "predicted_label" in data
    assert "gold_label" in data
    assert "label_matches_gold" in data
    assert data["gold_label"] == LABEL_SUPPORTS
    assert data["predicted_label"] in CANONICAL_LABELS
    # They are distinct fields
    assert isinstance(data["label_matches_gold"], bool)


@pytest.mark.skipif(
    not (Config.MODELS_DIR / "finetuned_nli" / "model.safetensors").exists(),
    reason="Honest NEI abstention on out-of-domain evidence requires the fine-tuned checkpoint (models/finetuned_nli). Zero-shot base model exhibits contradiction bias."
)
def test_empty_or_irrelevant_evidence_handled_honestly(client: TestClient):
    """Verify that absent or irrelevant evidence is classified as NOT ENOUGH INFORMATION without hallucination."""
    payload = {
        "claim": "Quantum electrodynamics superluminal neutrinos propagate in parallel dimensions without friction.",
        "top_k_docs": 3,
        "top_k_sentences": 2,
    }
    res = client.post("/api/verify", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["predicted_label"] == LABEL_NEI
    assert data["warning"] is not None
    assert "insufficient" in data["warning"].lower() or "not enough" in data["warning"].lower()


def test_supports_sample_claim_49_verified(client: TestClient):
    """Verify that Claim 49 (ADAR1 binds to Dicer...) correctly predicts SUPPORTS with Doc 5953485."""
    payload = {
        "claim_id": 49,
        "claim": "ADAR1 binds to Dicer to cleave pre-miRNA.",
        "top_k_docs": 5,
        "top_k_sentences": 3,
    }
    res = client.post("/api/verify", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["predicted_label"] == LABEL_SUPPORTS
    assert data["gold_label"] == LABEL_SUPPORTS
    assert data["label_matches_gold"] is True
    assert 5953485 in data["predicted_evidence_doc_ids"]
    assert [5953485, 2] in data["predicted_evidence_sentences"]
    assert data["confidence"] > 0.85


def test_divergence_reported_transparently_for_claim_5(client: TestClient):
    """Verify that when a model makes an error (e.g. Claim 5), mismatch is reported honestly, not masked."""
    payload = {
        "claim_id": 5,
        "claim": "1/2000 in UK have abnormal PrP positivity.",
        "top_k_docs": 5,
        "top_k_sentences": 3,
    }
    res = client.post("/api/verify", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["gold_label"] == LABEL_SUPPORTS
    if data["predicted_label"] != LABEL_SUPPORTS:
        assert data["label_matches_gold"] is False
        assert "Benchmark gold label is 'SUPPORTS'" in data["warning"]


def test_document_weighted_sentence_retrieval():
    """Verify that parent document relevance score informs sentence ranking in BM25Retriever."""
    corpus = {
        101: {
            "doc_id": 101,
            "title": "ADAR1 complex and pre-miRNA cleavage",
            "abstract": [
                "Introductory sentence here.",
                "ADAR1 forms a complex with Dicer to promote pre-miRNA cleavage.",
            ],
            "structured": False,
        },
        202: {
            "doc_id": 202,
            "title": "Unrelated cellular division overview",
            "abstract": [
                "ADAR1 protein is expressed broadly.",
                "Cell division proceeds normally.",
            ],
            "structured": False,
        },
    }
    retriever = BM25Retriever(corpus)
    # Both documents have sentences mentioning ADAR1, but Doc 101 has higher doc BM25 for the claim
    candidate_docs = [
        {"doc_id": 101, "bm25_score": 12.5, "title": corpus[101]["title"], "abstract": corpus[101]["abstract"]},
        {"doc_id": 202, "bm25_score": 1.5, "title": corpus[202]["title"], "abstract": corpus[202]["abstract"]},
    ]
    candidate_sentences = retriever.retrieve_sentences(
        query="ADAR1 binds to Dicer to cleave pre-miRNA.",
        candidate_docs=candidate_docs,
        top_k_sentences=2,
    )
    assert len(candidate_sentences) > 0
    top_match = candidate_sentences[0]
    assert top_match["doc_id"] == 101
    assert top_match["sentence_idx"] == 1


def test_class_probabilities_sum_to_one(client: TestClient):
    """Verify that calibrated_probabilities sum to approximately 1.0 for valid claims."""
    payload = {
        "claim": "ADAR1 binds to Dicer to cleave pre-miRNA.",
    }
    res = client.post("/api/verify", json=payload)
    assert res.status_code == 200
    probs = res.json()["calibrated_probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4
