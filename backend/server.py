"""
FastAPI Server for ResearchMap Scientific Claim Verification.
Exposes REST endpoints for claim analysis, paper details, benchmark metrics, and health status.
Loads the fine-tuned model and BM25 index once at startup.
"""

import os
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure workspace root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import Config
from backend.data_loader import (
    load_corpus,
    load_claims,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.pipeline import ClaimVerificationPipeline

# Global service state
app_state = {
    "corpus": {},
    "retriever": None,
    "entailment_model": None,
    "pipeline": None,
    "model_path": None,
    "model_loaded": False,
    "sample_claims": [],
    "benchmark_metrics": {},
}

def load_application_resources():
    """Load corpus, BM25 index, fine-tuned model checkpoint, and benchmark metrics."""
    # 1. Load Corpus
    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    corpus = load_corpus(corpus_path)
    app_state["corpus"] = corpus
    print(f"[Startup] Loaded {len(corpus):,} research paper abstracts.")

    # 2. Build BM25 Index
    retriever = BM25Retriever(corpus)
    app_state["retriever"] = retriever
    print("[Startup] Built BM25 lexical index.")

    # 3. Load Fine-Tuned Model Checkpoint (or base model fallback)
    finetuned_dir = Config.MODELS_DIR / "finetuned_nli"
    if finetuned_dir.exists() and (finetuned_dir / "model.safetensors").exists():
        model_name = str(finetuned_dir)
        print(f"[Startup] Using fine-tuned checkpoint: {model_name}")
    else:
        model_name = Config.NLI_MODEL_NAME
        print(f"[Startup] Fine-tuned checkpoint not found; falling back to base model: {model_name}")

    entailment_model = NLIEntailmentModel(model_name=model_name, device=Config.DEVICE)
    app_state["entailment_model"] = entailment_model
    app_state["model_path"] = model_name
    app_state["model_loaded"] = True

    # 4. Initialize Pipeline
    pipeline = ClaimVerificationPipeline(
        retriever=retriever,
        entailment_model=entailment_model,
        top_k_docs=Config.BM25_TOP_K_DOCS,
        top_k_sentences=Config.BM25_TOP_K_SENTENCES,
        entail_threshold=Config.ENTAIL_THRESHOLD,
        contradict_threshold=Config.CONTRADICT_THRESHOLD,
    )
    app_state["pipeline"] = pipeline

    # 5. Load Real Sample Claims from dev_heldout.jsonl or claims_dev.jsonl
    sample_path = Config.PROCESSED_DATA_DIR / "dev_heldout.jsonl"
    if not sample_path.exists():
        sample_path = Config.RAW_DATA_DIR / "claims_dev.jsonl"

    if sample_path.exists():
        dev_claims = load_claims(sample_path)
        app_state["dev_claims_by_id"] = {c["id"]: c for c in dev_claims}
        app_state["dev_claims_by_text"] = {c["claim"].strip().lower(): c for c in dev_claims}

        # Select verified representative examples across classes
        # Including working examples (49, 50, 51, 42, 1, 13) and documented model limitations (5, 48)
        target_sample_ids = [49, 50, 5, 51, 42, 48, 1, 13, 36]
        samples = []
        for tid in target_sample_ids:
            if tid in app_state["dev_claims_by_id"]:
                c = app_state["dev_claims_by_id"][tid]
                samples.append({
                    "id": c["id"],
                    "claim": c["claim"],
                    "gold_label": c["gold_label"],
                    "cited_doc_ids": c["cited_doc_ids"],
                })
        app_state["sample_claims"] = samples
        print(f"[Startup] Loaded {len(samples)} verified sample claims from SciFact.")

    # 6. Load Evaluated Benchmark Report Metrics
    get_or_reload_benchmark_metrics()

def get_or_reload_benchmark_metrics() -> Dict[str, Any]:
    """
    Retrieve benchmark metrics from memory cache, reloading if the on-disk file
    was modified after server startup. Handles missing or malformed reports safely.
    """
    report_json_path = Config.REPORTS_DIR / "model_comparison_metrics.json"
    if not report_json_path.exists():
        app_state["benchmark_metrics"] = {}
        app_state["benchmark_metrics_mtime"] = 0.0
        return {}

    try:
        current_mtime = report_json_path.stat().st_mtime
        if current_mtime != app_state.get("benchmark_metrics_mtime", 0.0) or not app_state.get("benchmark_metrics"):
            with open(report_json_path, "r", encoding="utf-8") as f:
                app_state["benchmark_metrics"] = json.load(f)
            app_state["benchmark_metrics_mtime"] = current_mtime
    except Exception as e:
        print(f"[Warning] Failed to read or parse benchmark metrics from {report_json_path}: {e}")
        if "benchmark_metrics" not in app_state:
            app_state["benchmark_metrics"] = {}

    return app_state.get("benchmark_metrics", {})

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_application_resources()
    except Exception as e:
        print(f"[Startup Error] Failed to initialize resources: {e}")
        app_state["model_loaded"] = False
    yield
    # Shutdown
    print("[Shutdown] Cleaning up application resources.")

app = FastAPI(
    title="ResearchMap API",
    description="Scientific Claim Verification & Evidence Mapping API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Request / Response Models ---

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1500, description="Scientific claim text to verify")
    top_k_docs: Optional[int] = Field(None, ge=1, le=20, description="Number of candidate papers to retrieve")
    top_k_sentences: Optional[int] = Field(None, ge=1, le=10, description="Number of candidate sentences to rank")
    claim_id: Optional[int] = Field(None, description="Optional claim ID if verifying a benchmark sample")

class EvidenceDetail(BaseModel):
    doc_id: int
    sentence_idx: int
    text: str
    title: str
    bm25_score: float
    entailment_label: str
    confidence: float
    probabilities: Dict[str, float]

class RetrievedPaper(BaseModel):
    doc_id: int
    title: str
    bm25_score: float
    abstract_preview: str

class VerifyResponse(BaseModel):
    claim_id: Optional[int] = None
    claim: str
    predicted_label: str
    confidence: float
    calibrated_probabilities: Dict[str, float]
    predicted_evidence_doc_ids: List[int]
    predicted_evidence_sentences: List[List[int]]
    evidence_details: List[EvidenceDetail]
    retrieved_candidate_doc_ids: List[int]
    retrieved_papers: List[RetrievedPaper]
    warning: Optional[str] = None
    explanation: Optional[str] = None
    disclaimer: str
    gold_label: Optional[str] = None
    label_matches_gold: Optional[bool] = None

class PaperResponse(BaseModel):
    doc_id: int
    title: str
    abstract: List[str]
    structured: bool

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_name: Optional[str]
    corpus_papers_count: int
    device: str
    version: str

# --- API Endpoints ---

@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint reporting service, model, and corpus status."""
    device = "cpu"
    if app_state["entailment_model"]:
        device = str(app_state["entailment_model"].device)

    return {
        "status": "healthy" if app_state["model_loaded"] else "degraded",
        "service": "ResearchMap NLP Verification Service",
        "model_loaded": app_state["model_loaded"],
        "model_name": app_state["model_path"],
        "corpus_papers_count": len(app_state["corpus"]),
        "device": device,
        "version": "1.0.0",
    }

@app.post("/api/verify", response_model=VerifyResponse)
def verify_scientific_claim(request: VerifyRequest):
    """
    Accepts a scientific claim, runs lexical BM25 retrieval across research abstracts,
    evaluates candidate evidence sentences via the fine-tuned NLI model,
    and returns a structured decision with grounded evidence and confidence scores.
    """
    if not app_state["model_loaded"] or app_state["pipeline"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification pipeline is not initialized or model failed to load."
        )

    claim_text = request.claim.strip()
    if not claim_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Claim text cannot be empty or solely whitespace."
        )

    # Run verification pipeline
    pipeline: ClaimVerificationPipeline = app_state["pipeline"]
    # Temporarily apply request top-k parameters if customized
    old_top_k_docs = pipeline.top_k_docs
    old_top_k_sentences = pipeline.top_k_sentences

    try:
        pipeline.top_k_docs = request.top_k_docs if request.top_k_docs is not None else Config.BM25_TOP_K_DOCS
        pipeline.top_k_sentences = request.top_k_sentences if request.top_k_sentences is not None else Config.BM25_TOP_K_SENTENCES
        raw_result = pipeline.verify_claim(claim_text)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification pipeline error: {exc}"
        ) from exc
    finally:
        pipeline.top_k_docs = old_top_k_docs
        pipeline.top_k_sentences = old_top_k_sentences

    # Build paper summaries
    retrieved_papers = []
    for doc_id in raw_result.get("retrieved_candidate_doc_ids", []):
        doc = app_state["corpus"].get(doc_id)
        if doc:
            abstract_text = " ".join(doc.get("abstract", []))
            preview = abstract_text[:280] + "..." if len(abstract_text) > 280 else abstract_text
            # Find BM25 score if present in evidence details
            score = 0.0
            for ev in raw_result.get("evidence_details", []):
                if ev["doc_id"] == doc_id:
                    score = ev["bm25_score"]
                    break
            retrieved_papers.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "bm25_score": score,
                "abstract_preview": preview,
            })

    # Class probabilities dictionary (uncalibrated softmax)
    calibrated_probs = {
        LABEL_SUPPORTS: 0.0,
        LABEL_CONTRADICTS: 0.0,
        LABEL_NEI: 0.0,
    }
    if raw_result.get("predicted_evidence_sentences") and raw_result.get("evidence_details"):
        best_doc_id, best_s_idx = raw_result["predicted_evidence_sentences"][0]
        matched_ev = next(
            (ev for ev in raw_result["evidence_details"] if ev["doc_id"] == best_doc_id and ev["sentence_idx"] == best_s_idx),
            None
        )
        if matched_ev:
            calibrated_probs = matched_ev.get("probabilities", calibrated_probs)
        else:
            calibrated_probs = raw_result["evidence_details"][0].get("probabilities", calibrated_probs)
    elif raw_result.get("evidence_details"):
        best_ev = raw_result["evidence_details"][0]
        calibrated_probs = best_ev.get("probabilities", calibrated_probs)
    else:
        calibrated_probs[LABEL_NEI] = 1.0

    # Determine warning for low confidence or insufficient evidence
    warning = None
    label = raw_result.get("predicted_label", LABEL_NEI)
    conf = raw_result.get("confidence", 0.0)

    # Check if claim matches a known SciFact dev claim (by claim_id or text)
    gold_label = None
    label_matches_gold = None
    matched_dev_claim = None

    if request.claim_id is not None and request.claim_id in app_state.get("dev_claims_by_id", {}):
        matched_dev_claim = app_state["dev_claims_by_id"][request.claim_id]
    elif claim_text.lower() in app_state.get("dev_claims_by_text", {}):
        matched_dev_claim = app_state["dev_claims_by_text"][claim_text.lower()]

    if matched_dev_claim:
        gold_label = matched_dev_claim["gold_label"]
        label_matches_gold = (label == gold_label)
        resolved_claim_id = matched_dev_claim["id"]
    else:
        resolved_claim_id = raw_result.get("claim_id") or request.claim_id

    if label == LABEL_NEI:
        warning = (
            "Insufficient conclusive evidence found in corpus. "
            "Neither high-confidence supporting nor contradicting evidence could be validated."
        )
    elif conf < 0.65:
        warning = (
            f"Verdict determined with borderline confidence ({conf:.1%}). "
            "Evidence should be inspected manually."
        )

    # If gold label is known and mismatches model prediction, report it transparently rather than disguising it
    if gold_label is not None and not label_matches_gold:
        mismatch_msg = f"Benchmark gold label is '{gold_label}', but model predicted '{label}'."
        if warning:
            warning = f"{warning} ({mismatch_msg})"
        else:
            warning = mismatch_msg

    # Format predicted sentences as list of [doc_id, sentence_idx]
    pred_sents = [[doc_id, s_idx] for doc_id, s_idx in raw_result.get("predicted_evidence_sentences", [])]

    return {
        "claim_id": resolved_claim_id,
        "claim": claim_text,
        "predicted_label": label,
        "confidence": float(conf),
        "calibrated_probabilities": calibrated_probs,
        "predicted_evidence_doc_ids": raw_result.get("predicted_evidence_doc_ids", []),
        "predicted_evidence_sentences": pred_sents,
        "evidence_details": raw_result.get("evidence_details", []),
        "retrieved_candidate_doc_ids": raw_result.get("retrieved_candidate_doc_ids", []),
        "retrieved_papers": retrieved_papers,
        "warning": warning,
        "explanation": raw_result.get("explanation"),
        "disclaimer": (
            "ResearchMap is an academic AI claim verification system. "
            "Predictions are probabilistic semantic entailment assessments based solely "
            "on the available research corpus, not an infallible statement of scientific fact."
        ),
        "gold_label": gold_label,
        "label_matches_gold": label_matches_gold,
    }

@app.get("/api/papers/{paper_id}", response_model=PaperResponse)
def get_paper_by_id(paper_id: int):
    """Retrieve full research paper details from the scientific corpus."""
    paper = app_state["corpus"].get(paper_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper with doc_id {paper_id} was not found in the indexed corpus."
        )
    return {
        "doc_id": paper["doc_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "structured": paper.get("structured", False),
    }

@app.get("/api/samples")
def get_sample_claims():
    """Return real curated claims from the SciFact dataset for UI demonstrations."""
    return {
        "samples": app_state.get("sample_claims", [])
    }

@app.get("/api/metrics")
def get_benchmark_metrics():
    """Return recorded Phase 1 Baseline and Phase 2/4A Fine-Tuned benchmark evaluation metrics."""
    return get_or_reload_benchmark_metrics()
