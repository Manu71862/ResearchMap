"""
API unit tests for ResearchMap FastAPI backend.
Tests:
- GET /api/health
- POST /api/verify (success, invalid input, empty retrieval, low confidence warnings)
- GET /api/papers/{paper_id} (valid and 404 cases)
- GET /api/samples
- GET /api/metrics
- Model loading failure handling (503 Service Unavailable)
"""

import pytest
from fastapi.testclient import TestClient

from backend.server import app, app_state
from backend.config import Config
from backend.data_loader import LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI, CANONICAL_LABELS

@pytest.fixture(scope="module")
def client():
    """Create test client using FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_loaded" in data
    assert "corpus_papers_count" in data
    assert data["corpus_papers_count"] > 0

def test_verify_success(client: TestClient):
    payload = {
        "claim": "0-dimensional biomaterials show inductive properties.",
        "top_k_docs": 3,
        "top_k_sentences": 2,
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["claim"] == payload["claim"]
    assert data["predicted_label"] in CANONICAL_LABELS
    assert 0.0 <= data["confidence"] <= 1.0
    assert "calibrated_probabilities" in data
    assert "evidence_details" in data
    assert "retrieved_papers" in data
    assert "disclaimer" in data
    assert len(data["retrieved_papers"]) <= 3

def test_verify_invalid_input(client: TestClient):
    # Empty string (violates min_length=3)
    res_empty = client.post("/api/verify", json={"claim": ""})
    assert res_empty.status_code == 422

    # Whitespace only
    res_ws = client.post("/api/verify", json={"claim": "    "})
    assert res_ws.status_code in (400, 422)

    # Missing claim key
    res_missing = client.post("/api/verify", json={})
    assert res_missing.status_code == 422

@pytest.mark.skipif(
    not (Config.MODELS_DIR / "finetuned_nli" / "model.safetensors").exists(),
    reason="In-domain NEI abstention on gibberish/irrelevant queries requires the fine-tuned checkpoint (models/finetuned_nli). Zero-shot base model exhibits contradiction bias."
)
def test_verify_empty_or_gibberish_retrieval(client: TestClient):
    # Nonsense string that yields no relevant matches
    payload = {"claim": "xyzqwerty887711nonexistentwordforretrieval"}
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == LABEL_NEI
    assert data["warning"] is not None

def test_paper_endpoint(client: TestClient):
    # Test valid paper from corpus
    if app_state["corpus"]:
        sample_doc_id = next(iter(app_state["corpus"].keys()))
        response = client.get(f"/api/papers/{sample_doc_id}")
        assert response.status_code == 200
        paper = response.json()
        assert paper["doc_id"] == sample_doc_id
        assert "title" in paper
        assert "abstract" in paper

    # Test non-existent paper
    res_404 = client.get("/api/papers/999999999")
    assert res_404.status_code == 404

def test_samples_and_metrics_endpoints(client: TestClient):
    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    samples = res_samples.json().get("samples", [])
    assert len(samples) > 0
    assert "claim" in samples[0]

    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert "metrics" in metrics_data

    # Verify Phase 4A claim-verification metrics (~0.6262 Macro-F1, ~0.6385 Weighted-F1)
    finetuned = metrics_data["metrics"]["finetuned"]
    assert "claim_verification" in finetuned
    assert finetuned["claim_verification"]["macro_f1"] == pytest.approx(0.6262, abs=0.002)
    assert finetuned["claim_verification"]["weighted_f1"] == pytest.approx(0.6385, abs=0.002)

    # Verify sentence evidence extraction metrics are strictly separated (~0.2914 sentence F1)
    assert "sentence_evidence" in finetuned
    assert finetuned["sentence_evidence"]["sentence_evidence_f1"] == pytest.approx(0.2914, abs=0.002)

    # Verify baseline metrics and deltas are present and distinct
    baseline = metrics_data["metrics"]["baseline"]
    deltas = metrics_data["metrics"]["deltas"]
    assert baseline["claim_verification"]["macro_f1"] == pytest.approx(0.4756, abs=0.002)
    assert deltas["macro_f1"] == pytest.approx(0.1505, abs=0.002)

def test_metrics_handles_missing_or_malformed_report(client: TestClient, monkeypatch, tmp_path):
    from backend.config import Config
    from backend import server

    # Reset cache
    orig_cache = server.app_state.get("benchmark_metrics", {})
    orig_mtime = server.app_state.get("benchmark_metrics_mtime", 0.0)

    try:
        # 1. Non-existent report file: returns {} safely without crashing
        monkeypatch.setattr(Config, "REPORTS_DIR", tmp_path)
        server.app_state["benchmark_metrics"] = {}
        server.app_state["benchmark_metrics_mtime"] = 0.0

        res_missing = client.get("/api/metrics")
        assert res_missing.status_code == 200
        assert res_missing.json() == {}

        # 2. Corrupted JSON file: returns {} safely without 500 error or crash
        corrupted_path = tmp_path / "model_comparison_metrics.json"
        corrupted_path.write_text("NOT_VALID_JSON{{{", encoding="utf-8")
        server.app_state["benchmark_metrics"] = {}
        server.app_state["benchmark_metrics_mtime"] = 0.0

        res_corrupt = client.get("/api/metrics")
        assert res_corrupt.status_code == 200
        assert res_corrupt.json() == {}
    finally:
        server.app_state["benchmark_metrics"] = orig_cache
        server.app_state["benchmark_metrics_mtime"] = orig_mtime


def test_model_loading_failure_handling(client: TestClient):
    # Temporarily set model_loaded = False
    original_flag = app_state["model_loaded"]
    try:
        app_state["model_loaded"] = False
        res = client.post("/api/verify", json={"claim": "Valid claim text test."})
        assert res.status_code == 503
        assert "not initialized" in res.json()["detail"].lower()
    finally:
        app_state["model_loaded"] = original_flag
