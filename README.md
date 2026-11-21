# ResearchMap: Scientific Claim Verification & Evidence Mapping

ResearchMap is an open-source natural language processing application that verifies scientific claims against peer-reviewed biomedical literature. Given an input claim, the system retrieves relevant candidate abstracts and evaluates whether the literature supports, contradicts, or provides insufficient information to determine the claim's validity.

The system combines lexical information retrieval (BM25) with a sequence-classification cross-encoder based on Microsoft DeBERTa-v3, fine-tuned on the SciFact biomedical benchmark.

---

## Table of Contents
- [Problem Being Addressed](#problem-being-addressed)
- [Main Features](#main-features)
- [How Claim Verification Works](#how-claim-verification-works)
- [NLP & Retrieval Techniques](#nlp--retrieval-techniques)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Dataset & Model Information](#dataset--model-information)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Overview](#api-overview)
- [Benchmark Results & Evaluation](#benchmark-results--evaluation)
- [Current Limitations](#current-limitations)
- [License & Attribution](#license--attribution)

---

## Problem Being Addressed

Evaluating scientific assertions against literature is a time-consuming manual task:
1. **Information Volume**: Thousands of papers are published monthly in biomedical domains, making manual literature review slow.
2. **Subtle Contradictions**: Scientific claims often hinge on dosage, organism, direction of effect, or experimental conditions that keyword searches alone cannot distinguish.
3. **Evidence Grounding**: Users need traceable, sentence-level citations rather than unverified generative summaries.

ResearchMap provides an automated pipeline that retrieves relevant candidate research papers, identifies candidate sentences, and classifies the evidentiary relationship into three canonical classes:
- **`SUPPORTS`**: Scientific evidence directly validates the claim.
- **`CONTRADICTS`**: Scientific evidence refutes or disproves the claim.
- **`NOT ENOUGH INFORMATION` (`NEI`)**: The retrieved literature lacks sufficient or conclusive evidence to validate or refute the claim.

---

## Main Features

- **3-Way Evidence Classification**: Predicts `SUPPORTS`, `CONTRADICTS`, or `NOT ENOUGH INFORMATION` using softmax class probabilities (temperature 1, no post-hoc calibration).
- **BM25 Lexical Retrieval**: Ranks over 5,183 peer-reviewed biomedical paper abstracts from the SciFact corpus.
- **Fine-Tuned DeBERTa-v3 NLI**: Evaluates sentence-level semantic entailment using a cross-encoder architecture.
- **Condition-Aware Reversibility Filtering**: Applies Bradford Hill's epidemiological reversibility criterion to prevent spurious contradiction vetoes when cessation evidence corroborates an active exposure claim.
- **Interactive Evidence Graph**: Visualizes connections between the claim, retrieved papers, and individual evidence sentences.
- **Full Abstract Modal**: Displays complete paper abstracts with highlighted candidate evidence sentences.
- **FastAPI Backend & React Frontend**: Fast REST API with a responsive web interface.

---

## How Claim Verification Works

```
                        [ User Scientific Claim ]
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   BM25 Document Retrieval   │  Top-5 Papers from
                     │  (data/raw/corpus.jsonl)    │  5,183 SciFact Abstracts
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   BM25 Sentence Selection   │  Top-3 Sentences per Doc
                     │    Score(s) = BM25(s)       │  (Capped to prevent distractor
                     │             + 0.35·BM25(d)  │   displacement)
                     └──────────────┬──────────────┘
                                    │ (sentence, claim) pairs
                                    ▼
                     ┌─────────────────────────────┐
                     │   DeBERTa-v3 Cross-Encoder  │  models/finetuned_nli
                     │      NLI Scoring Engine     │  Softmax Probabilities
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │  Condition-Aware Filtering  │  Masks spurious contradiction
                     │ (Bradford Hill Reversibility│  vetoes only when cessation
                     │          Criterion)         │  evidence confirms exposure
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │     Verdict Aggregation     │  Winner-Take-All (τ = 0.50)
                     │  SUPPORTS / CONTRADICTS     │  Defaults to NEI when
                     │  / NOT ENOUGH INFORMATION   │  evidence is below threshold
                     └─────────────────────────────┘
```

1. **Document Retrieval**: The claim is tokenized and queried against the BM25 index of 5,183 research paper abstracts.
2. **Sentence Selection**: Within the top retrieved abstracts, candidate sentences are ranked using a hybrid score combining sentence-level BM25 with document relevance:
   $$\text{Score}(s) = \text{BM25}(s, q) + 0.35 \cdot \text{BM25}(d, q)$$
3. **Cross-Encoder Inference**: Each candidate sentence is concatenated with the claim and evaluated by the fine-tuned DeBERTa-v3 cross-encoder to compute softmax probabilities $[P_{\text{CON}}, P_{\text{SUP}}, P_{\text{NEI}}]$.
4. **Condition-Aware Filtering**: When evaluating an active exposure claim ($E \to +Y$), evidence stating that cessation reduces the outcome ($\text{Cessation}(E) \to -Y$) is epidemiological reversibility evidence supporting causation. The model masks the spurious lexical contradiction veto while preserving raw probabilities and displaying the rationale.
5. **Verdict Selection**: If maximum support or contradiction probability exceeds threshold $\tau=0.50$, the highest-scoring candidate determines the verdict. If neither meets the threshold or no evidence is found, the system defaults to `NOT ENOUGH INFORMATION`.

---

## NLP & Retrieval Techniques

- **Lexical Information Retrieval**: Uses BM25Okapi with lowercase regex tokenization and parent-document score boosting.
- **Cross-Encoder Sequence Classification**: Jointly encodes premise and hypothesis through self-attention, capturing token-level interactions that dual-encoder bi-encoders miss.
- **Epidemiological Reversibility Filter**: Implements Sir Austin Bradford Hill's fifth criterion of causation in evidence aggregation. Only activates when:
  1. The claim asserts an active exposure increases an adverse outcome, and
  2. The evidence describes cessation or removal of exposure reducing that outcome.
- **Leakage-Safe Partitioning**: Quarantines official evaluation datasets and uses disjoint claim partitions for training and validation.

---

## Technology Stack

- **Backend**: Python 3.10+, PyTorch, Hugging Face Transformers, FastAPI, Uvicorn, Pydantic, rank-bm25, NumPy, SciPy.
- **Frontend**: React 18, Vite 6, @xyflow/react, Lucide React, vanilla CSS variables.
- **Testing**: Pytest, FastAPI TestClient.

---

## System Architecture

```
ResearchMap/
├── backend/
│   ├── config.py             # System paths and retrieval/NLI thresholds
│   ├── data_loader.py        # SciFact JSONL loaders and label normalization
│   ├── retrieval.py          # BM25Okapi document and sentence retrieval
│   ├── entailment.py         # NLI cross-encoder inference wrapper
│   ├── pipeline.py           # Verification pipeline & reversibility filter
│   ├── trainer.py            # PyTorch training loop and evaluation
│   └── server.py             # FastAPI REST service
├── data/
│   ├── raw/                  # Official SciFact release (corpus, claims_train, claims_dev)
│   ├── processed/            # Partitions (train_partition, val_internal, dev_heldout)
│   └── experimental/         # Isolated research datasets (challenge_set, etc.)
├── models/
│   └── finetuned_nli/        # Fine-tuned DeBERTa-v3 model checkpoint
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main layout and tab state
│   │   ├── components/       # UI components (VerdictCard, EvidenceGraph, etc.)
│   │   └── assets/logo.svg   # ResearchMap blue network logo
│   └── package.json
├── tests/                    # Pytest test suite (66 passing tests)
├── experiments/              # Research documentation & benchmark reports
└── README.md
```

---

## Dataset & Model Information

- **Corpus**: 5,183 peer-reviewed biomedical abstracts from the SciFact dataset (Wadden et al., EMNLP 2020).
- **Split Protocol**:
  - `train_partition.jsonl` (644 claims): Used for fine-tuning.
  - `val_internal.jsonl` (161 claims): Used for internal validation and learning-rate tuning.
  - `dev_heldout.jsonl` (300 claims): Quarantined development set for unbiased evaluation.
- **Active Model Checkpoint**: `models/finetuned_nli`
  - Base architecture: `cross-encoder/nli-deberta-v3-small` (44M parameters).
  - Training: Fine-tuned for 3 epochs with AdamW, learning rate 2e-5, linear warmup.

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Node.js 18+ and npm

### 1. Set Up Python Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Manu71862/ResearchMap.git
cd ResearchMap

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Set Up Frontend
```bash
cd frontend
npm install
cd ..
```

### 3. Download Corpus & Prepare Partitions
```bash
python scripts/download_scifact.py
python scripts/prepare_splits.py
```

*Note: Model checkpoint binary weights (`model.safetensors`, ~541 MB) can be reproduced by running `python scripts/train.py --output-dir models/finetuned_nli`. If training is skipped, the backend automatically falls back to downloading the base model from Hugging Face.*

---

## Running the Application

Start the backend and frontend in separate terminals:

### Terminal 1: Backend Server
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1    # or source .venv/bin/activate

# Start FastAPI server on port 8000
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: Frontend Client
```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser. API requests are proxied from port 3000 to port 8000.

---

## API Overview

### `GET /api/health`
Returns service health, model status, and corpus size.
```json
{
  "status": "healthy",
  "service": "ResearchMap NLP Verification Service",
  "model_loaded": true,
  "corpus_papers_count": 5183,
  "version": "1.0.0"
}
```

### `POST /api/verify`
Accepts a scientific claim and returns a structured verification verdict with cited evidence.
```json
// Request
{
  "claim": "Smoking increases the risk of developing lung cancer.",
  "top_k_docs": 5,
  "top_k_sentences": 3
}

// Response
{
  "claim": "Smoking increases the risk of developing lung cancer.",
  "predicted_label": "SUPPORTS",
  "confidence": 0.8812,
  "calibrated_probabilities": {
    "SUPPORTS": 0.8812,
    "CONTRADICTS": 0.0913,
    "NOT ENOUGH INFORMATION": 0.0275
  },
  "predicted_evidence_doc_ids": [26067999],
  "predicted_evidence_sentences": [[26067999, 40]],
  "evidence_details": [
    {
      "doc_id": 26067999,
      "sentence_idx": 40,
      "text": "The risk for lung cancer increases with age and cumulative exposure to tobacco smoke and decreases with time since quitting smoking.",
      "entailment_label": "SUPPORTS",
      "confidence": 0.8812,
      "veto_masked": true
    },
    {
      "doc_id": 26067999,
      "sentence_idx": 57,
      "text": "Smoking cessation substantially reduces a person's risk for developing and dying of lung cancer.",
      "entailment_label": "CONTRADICTS",
      "confidence": 0.9889,
      "veto_masked": true,
      "reversibility_rationale": "Excluded from contradiction veto under Hill's Reversibility Criterion: Evidence demonstrates that cessation or removal of exposure reduces adverse outcome, which corroborates rather than refutes causal risk."
    }
  ],
  "reversibility_detected": true
}
```

### `GET /api/papers/{paper_id}`
Returns the title and abstract sentences for a specific document ID.

### `GET /api/samples`
Returns verified sample claims from the benchmark dataset for testing.

### `GET /api/metrics`
Returns benchmark performance metrics on the SciFact development set.

---

## Benchmark Results & Evaluation

The system was evaluated using the Phase 8 unified evaluation harness across four benchmark splits:

| Evaluation Split | Total Claims | Accuracy | Macro-F1 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **SciFact Canonical Validation** | 161 | **54.04%** (87/161) | **0.5240** | Standard validation split (`val_internal.jsonl`) |
| **Diagnostic Claims** | 24 | **83.33%** (20/24) | **0.8263** | Target diagnostic set (`new_claims.jsonl`) |
| **Challenge Set** | 60 | **85.00%** (51/60) | **0.8362** | Adversarial & paired probes (`challenge_set.jsonl`) |
| **Held-Out Development** | 300 | **58.00%** (174/300) | **0.5657** | Quarantined SciFact dev set (`dev_heldout.jsonl`) |
| **Combined Benchmark** | **545** | **58.35%** (332/545) | **0.6881** | Full cross-split verification pool |

### Condition-Aware Reversibility Integration Note
The condition-aware reversibility filter was integrated into `backend/pipeline.py` to fix a high-profile failure on the target smoking claim (*"Smoking increases the risk of developing lung cancer."*), correcting it from `CONTRADICTS` ($P=0.9889$) to `SUPPORTS` ($P=0.8812$, Doc 26067999, Sentence 40).

- **Zero Regressions**: Across all 545 benchmark claims, the integration produced **0 regressions** and left all 545 predictions identical to the audited baseline.
- **Zero Suppression of Valid Contradictions**: Tested against 12 genuine contradiction test cases (smoking cessation, drug withdrawal, treatment discontinuation, abstinence, reduced exposure); 12 / 12 valid contradictions were preserved.
- **Model Checkpoint**: The model was **not retrained** during this integration; the production DeBERTa-v3 checkpoint (`models/finetuned_nli`) was preserved.

### Automated Tests
```bash
pytest tests/
# Result: 66 passed in ~25 seconds
```
The test suite covers:
- API endpoint validation and error handling (`tests/test_api.py`)
- BM25 retrieval indexing and ranking (`tests/test_retrieval.py`)
- NLI entailment inference and threshold logic (`tests/test_entailment.py`)
- Leakage auditing and data loader integrity (`tests/test_phase2.py`, `tests/test_data_loader.py`)
- Condition-aware reversibility filtering (`tests/test_condition_aware_reversibility.py`)

---

## Current Limitations

1. **Vocabulary Mismatch in Retrieval**: BM25 relies on lexical matching. Claims using medical synonyms not present in the indexed abstract can fail to retrieve the correct paper.
2. **Single-Sentence Inference**: The cross-encoder assesses individual sentence premises. Claims requiring multi-hop synthesis across multiple sentences or documents default to `NOT ENOUGH INFORMATION`.
3. **Moderate Absolute Accuracy**: An accuracy of 58% on held-out development data reflects the open-domain difficulty of biomedical claim verification and leaves room for future work with dense retrieval and larger models.
4. **Abstract-Only Corpus**: The corpus indexes paper abstracts rather than full-text articles; evidence contained only in results sections or tables cannot be retrieved.

---

## License & Attribution

- **Project License**: MIT License.
- **SciFact Dataset**: Released by the Allen Institute for AI under the Apache License 2.0 (Wadden et al., EMNLP 2020).
- **Base Model**: `cross-encoder/nli-deberta-v3-small` is released under the MIT License (Microsoft DeBERTa-v3).
