# ResearchMap — Scientific Claim Verification & Evidence Mapping

**ResearchMap** is a Natural Language Processing (NLP) capstone project developed for third-year Computer Science & Engineering. The system automatically verifies complex scientific claims against peer-reviewed scientific literature and classifies evidence relationships into three canonical categories:
- **`SUPPORTS`**: Grounded scientific evidence directly validates the claim.
- **`CONTRADICTS`**: Grounded scientific evidence refutes or falsifies the claim.
- **`NOT ENOUGH INFORMATION` (`NEI`)**: The available corpus lacks sufficient or conclusive evidence to substantiate or refute the claim.

---

## Phase 2: Leakage-Safe Fine-Tuning & Fair Evaluation

In Phase 2, we conducted an audit of the evaluation protocol, established an airtight leakage-safe partitioning scheme, and fine-tuned a compact scientific sequence classifier (`cross-encoder/nli-deberta-v3-small`, 44M parameters) on SciFact claim-evidence pairs. We evaluated the fine-tuned checkpoint against the zero-shot baseline on the **exact same 300 held-out claims** of the official SciFact development set.

### Architecture Overview
```
                         [ Scientific Claim ]
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ BM25 Lexical Retrieval  │ (Abstract-level Top-5 &
                     │     (rank-bm25)         │  Sentence-level Top-3)
                     └────────────┬────────────┘
                                  │ Candidate Evidence Sentences
                                  ▼
                     ┌─────────────────────────┐
                     │  Fine-Tuned Scientific  │ (cross-encoder/nli-deberta-v3-small
                     │  NLI Cross-Encoder      │  Fine-Tuned on SciFact Pairs)
                     └────────────┬────────────┘
                                  │ Calibrated In-Domain Probabilities
                                  ▼
                   ┌──────────────────────────────┐
                   │ Evidence-Grounded Decision:  │
                   │ SUPPORTS / CONTRADICTS / NEI │
                   │ + Traceable Cited Evidence   │
                   └──────────────────────────────┘
```

---

## Project Structure

```
ResearchMap/
├── backend/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Configurable relative paths and thresholds
│   ├── data_loader.py        # SciFact JSONL loaders, label normalization, split statistics
│   ├── dataset_builder.py    # Semantic pair construction (multi-sentence & distractor safety)
│   ├── retrieval.py          # BM25Okapi document and sentence lexical retrieval
│   ├── entailment.py         # NLI Cross-Encoder inference wrapper (supports local checkpoints)
│   ├── trainer.py            # PyTorch training loop, dynamic padding, checkpointing
│   ├── pipeline.py           # End-to-end claim verification pipeline
│   └── evaluation.py         # Macro-F1, Recall@K, Sentence Evidence P/R/F1, confusion matrix
├── scripts/
│   ├── download_scifact.py   # Downloads & extracts official SciFact release
│   ├── prepare_splits.py     # Leakage-safe partitioning (training claims only)
│   ├── train.py              # Model fine-tuning CLI
│   ├── evaluate_baseline.py  # Repeatable benchmark evaluation runner
│   ├── compare_models.py     # Fair side-by-side model comparison runner
│   └── run_baseline.py       # Single-claim ad-hoc verification via CLI
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py   # Tests label parsing, evidence extraction, split stats
│   ├── test_retrieval.py     # Tests BM25 indexing, empty queries, sentence ranking
│   ├── test_entailment.py    # Tests mock NLI decision logic and thresholds
│   ├── test_evaluation.py    # Tests F1, recall, precision, and matrix calculations
│   └── test_phase2.py        # Tests leakage audit, distractor safety, checkpoint save/reload
├── data/
│   ├── raw/                  # Official SciFact release files (corpus, claims_train, claims_dev, claims_test)
│   └── processed/            # Leakage-safe partitions & metadata (train_partition, val_internal, dev_heldout)
├── models/
│   └── finetuned_nli/        # Fine-tuned model checkpoint (safetensors, config, tokenizer)
├── reports/                  # Benchmark reports, metrics JSON, and predictions JSONL
│   ├── model_comparison_report.md
│   ├── model_comparison_metrics.json
│   ├── baseline_dev_evaluation_report.md
│   └── baseline_dev_metrics.json
├── frontend/                 # Scaffolding for future React + Vite website
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
├── requirements.txt          # Python dependencies with pinned versions
├── .env.example              # Template environment configuration
├── .env                      # Local environment configuration
├── .gitignore                # Git exclusions (caches, virtualenvs, weights)
└── README.md                 # Complete project documentation
```

---

## Leakage-Safe Data Protocol

### Audit Findings
In standard SciFact releases, the official test set (`claims_test.jsonl`) labels are withheld on the AI2 leaderboard. To evaluate locally without data leakage:
1. **Zero Dev Overlap**: The official development set (`claims_dev.jsonl`, 300 claims) was strictly **quarantined from training**.
2. **Purged Training Overlap**: Across the 809 raw training claims, 2 internal duplicates and 2 claims textually overlapping with the official dev set were purged.
3. **Partitioning**: The remaining 805 eligible training claims were split with random seed `42`:
   - **`train_partition.jsonl`** (644 claims, 80%): Used exclusively for model parameter fine-tuning.
   - **`val_internal.jsonl`** (161 claims, 20%): Used for internal validation, learning rate scheduling, and early stopping.
   - **`dev_heldout.jsonl`** (300 claims): Preserved as the independent benchmark for fair model comparison.
4. **Leakage Verification**: Formally audited by `scripts/prepare_splits.py` and unit tests in `tests/test_phase2.py` — confirming **0 overlapping claim IDs and 0 overlapping claim texts** across all partitions.

---

## Environment Setup & Reproduction (PowerShell)

### 1. Setup Virtual Environment & Dependencies
```powershell
# From the ResearchMap root directory:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download Dataset & Prepare Leakage-Safe Partitions
```powershell
python scripts\download_scifact.py
python scripts\prepare_splits.py
```

### 3. Run All Automated Unit Tests (20 Tests)
```powershell
pytest -v
```

### 4. Fine-Tune the Scientific NLI Model
```powershell
python scripts\train.py --epochs 3 --batch-size 8 --lr 2e-5 --output-dir models/finetuned_nli
```

### 5. Run Fair Side-by-Side Model Comparison Benchmark
```powershell
python scripts\compare_models.py --split dev
```

### 6. Verify an Ad-Hoc Scientific Claim (CLI)
```powershell
python scripts\run_baseline.py --claim "0-dimensional biomaterials show inductive properties."
```

---

## Measured Benchmark Results: Baseline vs. Fine-Tuned

Evaluated on the **exact same 300 held-out claims** of the official SciFact development set (`claims_dev.jsonl`) against the full 5,183-abstract corpus:

### Executive Comparison
| Evaluation Dimension | Metric | Zero-Shot Baseline | Fine-Tuned Model | Absolute Delta | Relative Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Claim Verification** | **Macro-F1** | **0.4625** | **0.6088** | **+0.1463** | **+31.64%** |
| | **Weighted-F1** | 0.4661 | **0.6183** | **+0.1521** | **+32.63%** |
| | **SUPPORTS F1** | 0.4678 | **0.6778** | **+0.2100** | **+44.89%** |
| | **CONTRADICTS F1** | 0.4410 | **0.5672** | **+0.1261** | **+28.60%** |
| | **NOT ENOUGH INFO F1** | 0.4786 | **0.5815** | **+0.1029** | **+21.50%** |
| **Document Retrieval** | **Doc Recall@5** | 87.77% | 87.77% | +0.00% | Shared BM25 |
| | **Doc Recall@10** | 92.55% | 92.55% | +0.00% | Shared BM25 |
| **Evidence Extraction** | **Precision** | 0.2865 | **0.3784** | **+0.0919** | **+32.08%** |
| | **Recall** | 0.1393 | **0.1913** | **+0.0519** | **+37.26%** |
| | **F1-Score** | 0.1875 | **0.2541** | **+0.0666** | **+35.52%** |

### Per-Class Precision & Recall Breakdown
| Class | Baseline Precision | Baseline Recall | Baseline F1 | Fine-Tuned Precision | Fine-Tuned Recall | Fine-Tuned F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SUPPORTS`** | 0.851 | 0.323 | 0.468 | 0.704 | **0.653** | **0.678** | **+0.210** |
| **`CONTRADICTS`** | 0.328 | 0.672 | 0.441 | **0.543** | 0.594 | **0.567** | **+0.126** |
| **`NOT ENOUGH INFORMATION`** | 0.459 | 0.500 | 0.479 | **0.574** | **0.589** | **0.581** | **+0.103** |

### Confusion Matrix Comparison
#### Baseline (Zero-Shot) Model:
| True \ Predicted | SUPPORTS | CONTRADICTS | NOT ENOUGH INFORMATION |
| :--- | :---: | :---: | :---: |
| **`SUPPORTS`** | 40 | 39 | 45 |
| **`CONTRADICTS`** | 0 | 43 | 21 |
| **`NOT ENOUGH INFORMATION`** | 7 | 49 | 56 |

#### Fine-Tuned Model:
| True \ Predicted | SUPPORTS | CONTRADICTS | NOT ENOUGH INFORMATION |
| :--- | :---: | :---: | :---: |
| **`SUPPORTS`** | **81** | 11 | 32 |
| **`CONTRADICTS`** | 9 | **38** | 17 |
| **`NOT ENOUGH INFORMATION`** | 25 | 21 | **66** |

**Key Observations**:
1. Correctly identified `SUPPORTS` claims doubled from 40 to **81** (recall jumped from 32.3% to 65.3%).
2. Misclassification of neutral (`NEI`) claims as `CONTRADICTS` dropped by more than half (from 49 down to 21).
3. Fine-tuning resolved the severe out-of-domain contradiction bias inherent in general NLI models.
---

## Limitations Identified & Future Roadmap

1. **Retriever Bottleneck**: Lexical BM25 achieves 92.55% Recall@10, meaning ~7.5% of claims with evidence are missed before verification begins. Introducing a dense bi-encoder (e.g. `allenai/specter2`) will bridge this gap.
2. **Joint Document-Level Reasoning**: While sentence evidence F1 improved from 0.1875 to 0.2541, joint multi-sentence reasoning across paragraphs remains an area for architecture enhancement.
3. **Threshold Calibration**: The current pipeline uses symmetric 0.50 thresholds. Learning per-class decision boundaries on `val_internal.jsonl` will yield further precision improvements.

---

## Phase 3: Working Web Application

In Phase 3, we built an academic research-evidence explorer web application that connects the existing fine-tuned model checkpoint (`models/finetuned_nli`) and BM25 retrieval index to an interactive React + Vite frontend.

### Architecture & System Design
```
       ┌────────────────────────────────────────────────────────┐
       │                   Vite React Frontend                  │
       │  (Port 3000 — Dark Navy Academic Aesthetic & UI)       │
       │                                                        │
       │  • Claim Input & SciFact Real Samples Chips            │
       │  • Verdict Card & Calibrated Confidence Meters         │
       │  • Interactive Evidence Graph (@xyflow/react)          │
       │  • Side Inspector Panel & Sentence-by-Sentence Modal   │
       │  • Empirical Model Performance & Benchmark Reports     │
       └───────────────────────────┬────────────────────────────┘
                                   │ /api proxy
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                  Python FastAPI Backend                │
       │            (Port 8000 — Loaded Once at Startup)        │
       │                                                        │
       │  GET  /api/health       -> Service, model, corpus info │
       │  POST /api/verify       -> BM25 + NLI model pipeline   │
       │  GET  /api/papers/{id}  -> Abstract details by doc_id  │
       │  GET  /api/samples      -> Curated SciFact dev samples │
       │  GET  /api/metrics      -> Official evaluation metrics │
       └───────────────────────────┬────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌──────────────────────┐                  ┌──────────────────────┐
   │ BM25 Lexical Index   │                  │ Fine-Tuned DeBERTa   │
   │ (5,183 SciFact Docs) │                  │ (models/finetuned_nli│
   └──────────────────────┘                  └──────────────────────┘
```

### Starting Backend and Frontend

Run the backend and frontend in two separate PowerShell terminals:

#### Terminal 1 — Backend API Server:
```powershell
cd "c:\Users\Balija Manohar\Downloads\ResearchMap"
.\.venv\Scripts\Activate.ps1
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
*The service will load the 5,183 abstracts, compile the BM25 index, and initialize the fine-tuned NLI model once at startup.*

#### Terminal 2 — Frontend Application:
```powershell
cd "c:\Users\Balija Manohar\Downloads\ResearchMap\frontend"
npm run dev
```
*Open `http://localhost:3000` in your browser. All `/api` requests are proxied directly to port 8000.*

### Running Verification Tests

#### Full Python Test Suite (27 Unit & Integration Tests):
```powershell
cd "c:\Users\Balija Manohar\Downloads\ResearchMap"
.\.venv\Scripts\pytest -v
```

#### Frontend Production Build:
```powershell
cd "c:\Users\Balija Manohar\Downloads\ResearchMap\frontend"
npm run build
```

#### Command-Line Ad-Hoc Inference:
```powershell
cd "c:\Users\Balija Manohar\Downloads\ResearchMap"
.\.venv\Scripts\python scripts/run_baseline.py --claim "0-dimensional biomaterials show inductive properties."
```

---

## Phase 6A — Evidence Selection & NLI Improvement

In Phase 6A, we conducted an experimental investigation into whether refined evidence sentence selection and multi-sentence aggregation could improve Natural Language Inference (NLI) classification on the expanded 10,835-abstract corpus without retraining the fine-tuned DeBERTa-v3 cross-encoder.

### Evaluated Approaches:
1. **Approach A (Existing Baseline Selection)**: Lexical BM25 sentence score with parent document prior ($\text{Score} = \text{BM25}(s) + 0.35 \cdot \text{BM25}(d)$, max 2 per doc).
2. **Approach B (BM25-based Selection)**: Pure sentence-level lexical relevance ($\text{doc\_score\_weight} = 0.0$, max 2 per doc).
3. **Approach C (Hybrid Document Score + Sentence Relevance)**: Normalized Hybrid RRF consensus document score fused with sentence relevance ($\text{BM25}(s) + 0.35 \cdot \widetilde{\text{RRF}}(d) \cdot 15.0$).
4. **Approach D (Evidence Diversity Selection)**: Maximal Marginal Relevance (MMR) with lexical Jaccard redundancy penalty and cross-document diversification.
5. **Approach E (Multi-Sentence Evidence Aggregation)**: Document-bounded multi-sentence premise concatenation and calibrated decision consensus.

### Benchmark Summary:
- **Validation Dataset**: 161 claims (`data/processed/val_internal.jsonl`)
- **Diagnostic Test Dataset**: 24 out-of-domain claims (`experiments/new_claims.jsonl`)
- **Best Validation Macro-F1**: **0.5635** (Approach C: Hybrid Document Score + Sentence Relevance)
- **Best Validation Accuracy**: **58.39%** (Tied across Approaches B, C, and D, outperforming baseline 57.76%)
- **Best Diagnostic Test Accuracy**: **83.33%** (Approaches A and B)
- **Fastest Experimental Strategy**: **Approach B** (Mean CPU latency: **244.21 ms**, P95: **315.07 ms** on validation; **236.81 ms** on test)
- **Multi-Sentence Aggregation Finding**: Approach E underperformed (56.52% validation accuracy), proving that concatenating multiple sentences dilutes specific claim evidence with background caveats and increases latency (+81%).

### Production Default Rationale:
**BM25 Configuration C remains the active production default** because:
1. It is thoroughly verified across all regression suites without requiring dense vector index memory or embedding cache dependencies.
2. It provides high precision on the diagnostic test suite (87.50% in initial benchmark, 83.33% in controlled runs) with zero risk of neural distractor drift.
3. If Hybrid Retrieval is deployed in staging, pairing it with **Approach B sentence selection** ($\text{doc\_score\_weight} = 0.0$) is recommended over multi-sentence aggregation to maintain sub-250ms CPU latency and eliminate parent-document bias.

For full methodology, error modes, confusion matrices, and the Claim 24 investigation, see [experiments/PHASE6A_EVIDENCE_SELECTION_REPORT.md](file:///C:/Users/Balija%20Manohar/Downloads/ResearchMap/experiments/PHASE6A_EVIDENCE_SELECTION_REPORT.md).
