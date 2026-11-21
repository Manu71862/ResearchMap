# ResearchMap — Scientific Claim Verification & Evidence Mapping

**ResearchMap** is a Natural Language Processing (NLP) capstone project developed for third-year Computer Science & Engineering. The system automatically verifies complex scientific claims against peer-reviewed scientific literature and classifies evidence relationships into three canonical categories:
- **`SUPPORTS`**: Grounded scientific evidence directly validates the claim.
- **`CONTRADICTS`**: Grounded scientific evidence refutes or falsifies the claim.
- **`NOT ENOUGH INFORMATION` (`NEI`)**: The available corpus lacks sufficient or conclusive evidence to substantiate or refute the claim.

---

## Production Pipeline Overview (Configuration C)

The active production pipeline couples lexical BM25 retrieval with a domain-adapted sequence classification cross-encoder:

```
                         [ Scientific Claim ]
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ BM25 Lexical Retrieval  │ (Abstract-level Top-10 &
                     │   (Configuration C)     │  Sentence-level Top-4, Doc Weight: 0.35)
                     └────────────┬────────────┘
                                  │ Candidate Evidence Sentences (Max 2/Doc)
                                  ▼
                     ┌─────────────────────────┐
                     │  Fine-Tuned Scientific  │ (cross-encoder/nli-deberta-v3-small
                     │  NLI Cross-Encoder      │  Active Checkpoint: models/finetuned_nli)
                     └────────────┬────────────┘
                                  │ Softmax Entailment Probabilities
                                  ▼
                   ┌──────────────────────────────┐
                   │ Evidence-Grounded Decision:  │
                   │ SUPPORTS / CONTRADICTS / NEI │
                   │ + Traceable Cited Evidence   │
                   └──────────────────────────────┘
```

### Production Components & Configuration:
- **Corpus**: 5,183 peer-reviewed biomedical abstracts from the SciFact corpus (`data/raw/corpus.jsonl`).
- **Retrieval Engine**: BM25Okapi with lowercase regex tokenization (`backend/retrieval.py`).
- **Sentence Scoring**: Configuration C combines sentence-level BM25 with parent document relevance:
  $$\text{Score}(s) = \text{BM25}(s, q) + 0.35 \cdot \text{BM25}(d, q)$$
  Sentences are capped at a maximum of 2 per candidate document to prevent distractor displacement.
- **Active NLI Model**: `models/finetuned_nli` (fine-tuned 3 epochs on SciFact claim-evidence pairs, 44M parameters).
- **Probability Output**: Standard uncalibrated softmax entailment probabilities $[P_{\text{CON}}, P_{\text{SUP}}, P_{\text{NEI}}]$ summing to 1.0 (probability calibration was explored under isolated Phase 4 research).
- **Decision Thresholds**: Symmetric $\tau_{\text{SUP}} = 0.50, \tau_{\text{CON}} = 0.50$ with default abstention to `NOT ENOUGH INFORMATION`.

---

## Project Structure

```
ResearchMap/
├── backend/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Configurable relative paths and Configuration C settings
│   ├── data_loader.py        # SciFact JSONL loaders, label normalization, split statistics
│   ├── dataset_builder.py    # Semantic pair construction (multi-sentence & distractor safety)
│   ├── retrieval.py          # BM25Okapi document and sentence lexical retrieval
│   ├── entailment.py         # NLI Cross-Encoder inference wrapper (dynamic id2label mapping)
│   ├── trainer.py            # PyTorch training loop, dynamic padding, checkpointing
│   ├── pipeline.py           # End-to-end claim verification pipeline
│   ├── evaluation.py         # Macro-F1, Recall@K, Sentence Evidence P/R/F1, confusion matrix
│   └── server.py             # FastAPI REST API (startup lifespan, health, verify, paper, metrics)
├── scripts/
│   ├── download_scifact.py   # Downloads & extracts official SciFact release
│   ├── prepare_splits.py     # Leakage-safe partitioning (training claims only)
│   ├── train.py              # Model fine-tuning CLI
│   ├── evaluate_baseline.py  # Repeatable benchmark evaluation runner
│   ├── compare_models.py     # Fair side-by-side model comparison runner
│   ├── run_baseline.py       # Single-claim ad-hoc verification via CLI
│   └── experimental/         # Isolated experimental research scripts (Phases 1–6A)
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # Tests FastAPI endpoints, validation, error handling
│   ├── test_data_loader.py   # Tests label parsing, evidence extraction, split stats
│   ├── test_retrieval.py     # Tests BM25 indexing, empty queries, sentence ranking
│   ├── test_entailment.py    # Tests NLI decision logic, thresholds, dynamic mapping
│   ├── test_evaluation.py    # Tests F1, recall, precision, and matrix calculations
│   ├── test_phase2.py        # Tests leakage audit, distractor safety, checkpoint reload
│   ├── test_experimental_entity_grounding.py # Tests 15 scientific entity grounding rules
│   ├── test_experimental_numerical.py        # Tests numerical comparison rules
│   └── test_regression_phase4.py             # Tests Phase 4 regression safety
├── data/
│   ├── raw/                  # Official SciFact release files (corpus, claims_train, claims_dev, claims_test)
│   ├── processed/            # Leakage-safe partitions (train_partition, val_internal, dev_heldout)
│   └── experimental/         # Research datasets (hard_negative_pairs, challenge_set, corpus_expanded_10k)
├── models/
│   ├── finetuned_nli/        # Active production model checkpoint (safetensors, config, tokenizer)
│   ├── finetuned_epoch1/     # Intermediate epoch 1 checkpoint
│   ├── experimental_augmented_nli/   # Phase 3 augmented research checkpoint
│   └── experimental_calibrated_nli/  # Phase 4 class-balanced loss weighting research checkpoint
├── reports/                  # Benchmark reports, metrics JSON, and predictions JSONL
├── experiments/              # Comprehensive research reports (Phases 1, 2, 3, 4, 5A, 5B, 6A, Audit)
├── frontend/                 # Interactive React 18 + Vite 6 web application
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx           # Root layout and tab management
│       └── components/       # UI components (EvidenceGraph, VerdictCard, Navbar, etc.)
├── requirements.txt          # Python dependencies with pinned versions
├── .env.example              # Template environment configuration (copy to .env; .env contains local settings and is untracked)
├── .gitignore                # Git exclusions (caches, virtualenvs, weights, local .env)
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

> [!NOTE]
> **Model Weights Note**: Binary model checkpoint weights (`model.safetensors`, ~541 MB each) are excluded from the Git repository due to GitHub file-size limits.
> - **Production Checkpoint Setup**: Follow Step 3 below to fine-tune the production checkpoint locally (`models/finetuned_nli`).
> - **Base-Model Fallback**: If you skip training, the backend automatically falls back to downloading the base pretrained model (`cross-encoder/nli-deberta-v3-small`) from Hugging Face on demand. All core pipeline components remain fully operational, and tests that specifically validate fine-tuned checkpoint files will gracefully skip.

### 1. Setup Virtual Environment & Dependencies
```powershell
# From the ResearchMap root directory:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Install frontend dependencies:
cd frontend
npm install
cd ..
```

### 2. Download Dataset & Prepare Leakage-Safe Partitions
```powershell
python scripts\download_scifact.py
python scripts\prepare_splits.py
```
*Note: Raw SciFact abstracts (`corpus.jsonl`, 8.3 MB) and raw annotations are excluded from the repository to maintain a minimal clone footprint (< 3 MB). Running `download_scifact.py` fetches the official release directly from AWS S3, and `prepare_splits.py` compiles the stratified splits into `data/processed/`.*

*(Optional) If reproducing experimental dense/hybrid retrieval benchmarks (Phases 5A & 5B), regenerate the excluded embedding cache (16.6 MB `.npy` matrix) via:*
```powershell
python scripts\experimental\build_dense_cache.py
```

### 3. Fine-Tune the Production Scientific NLI Model
```powershell
python scripts\train.py --model cross-encoder/nli-deberta-v3-small --epochs 3 --batch-size 8 --lr 2e-5 --output-dir models/finetuned_nli
```
*Reproduces the training procedure on leakage-safe SciFact pairs using the fixed random seed (42) and outputs the fine-tuned checkpoint to `models/finetuned_nli`. (Note: Due to floating-point nondeterminism across different GPU architectures and CUDA/PyTorch runtime kernels, local training reproduces the model training procedure and benchmark performance profile rather than bit-for-bit identical binary tensor weights). Once generated, the FastAPI server will automatically prioritize this local checkpoint.*

### 4. Run Automated Test Suite
```powershell
.\.venv\Scripts\pytest -v
```
*Executes all unit, integration, and regression tests. All 60 tests pass when the fine-tuned checkpoint is present. On fresh clones before running Step 3, 57 tests pass and 3 checkpoint-dependent tests skip gracefully with clear explanations.*

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

## Interactive Web Application

The interactive web application connects the active fine-tuned checkpoint (`models/finetuned_nli`) and BM25 index to a modern React 18 frontend:

### Architecture & System Design
```
       ┌────────────────────────────────────────────────────────┐
       │                   Vite React Frontend                  │
       │  (Port 3000 — Dark Navy Academic Aesthetic & UI)       │
       │                                                        │
       │  • Claim Input & SciFact Real Samples Chips            │
       │  • Verdict Card & Softmax Probability Meters           │
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
# From repository root:
.\.venv\Scripts\Activate.ps1
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
*The service loads the 5,183 abstracts, compiles the BM25 index, and initializes the fine-tuned NLI model once at startup on port 8000.*

#### Terminal 2 — Frontend Application:
```powershell
cd frontend
npm run dev
```
*Open `http://localhost:3000` in your browser. All `/api` requests are automatically proxied to port 8000.*

### Running Verification Tests

#### Full Python Test Suite (60 Passing Tests):
```powershell
# From repository root:
.\.venv\Scripts\pytest -v
```

#### Frontend Production Build:
```powershell
cd frontend
npm run build
```

---

## Experimental Research Catalog (Phases 1–6A)

To prevent data drift and preserve production stability, all advanced explorations were developed in **isolated research scripts** under `scripts/experimental/` and documented in `experiments/`. These models and dense indexes remain research artifacts and are **not active in production**:

| Research Phase | Topic & Focus | Report | Status | Key Finding |
| :---: | :--- | :--- | :---: | :--- |
| **Phase 1** | Evidence Aggregation Improvement | [PHASE1_AGGREGATION_REPORT.md](experiments/PHASE1_AGGREGATION_REPORT.md) | Research | Evaluated 4 aggregation strategies; consensus mitigated some false contradictions but reduced true positive predictions; voting did not fully solve the veto problem. |
| **Phase 2** | Scientific Entity Grounding Filter | [PHASE2_ENTITY_GROUNDING_REPORT.md](experiments/PHASE2_ENTITY_GROUNDING_REPORT.md) | Research | 15/15 unit tests pass; prevents cross-organism entity mismatches (+1.24% validation accuracy). |
| **Phase 3** | Hard-Negative Augmentation | [PHASE3_HARD_NEGATIVE_REPORT.md](experiments/PHASE3_HARD_NEGATIVE_REPORT.md) | Research | Added 1,500 hard negatives; fine-tuned `models/experimental_augmented_nli` (85% on 60-probe challenge set). |
| **Phase 4** | Targeted Error Analysis & Calibration | [PHASE4_ERROR_ANALYSIS_REPORT.md](experiments/PHASE4_ERROR_ANALYSIS_REPORT.md) | Research | Identified 61.3% NEI prior shift (+1.33 logit offset); trained class-balanced research checkpoint (`models/experimental_calibrated_nli`). Recommended retaining baseline in production. |
| **Phase 4A**| BM25 Configuration C Tuning | [PHASE4A_RETRIEVAL_IMPLEMENTATION.md](experiments/PHASE4A_RETRIEVAL_IMPLEMENTATION.md) | **Production** | Established Configuration C (parent doc weight 0.35, max 2 sents/doc) achieving 83.33% diagnostic accuracy and 89.11% validation Doc Recall@10. |
| **Phase 4B**| Corpus Expansion | [PHASE4B_CORPUS_EXPANSION.md](experiments/PHASE4B_CORPUS_EXPANSION.md) | Research | Expanded corpus to 10,835 abstracts via PubMed to stress-test distractor robustness. |
| **Phase 5A**| Hybrid Dense-Lexical Retrieval | [PHASE5A_HYBRID_RETRIEVAL_REPORT.md](experiments/PHASE5A_HYBRID_RETRIEVAL_REPORT.md) | Research | Evaluated `all-MiniLM-L6-v2` dense retrieval fused via Reciprocal Rank Fusion (RRF). |
| **Phase 5B**| Hybrid Latency Optimization | [PHASE5B_OPTIMIZATION_REPORT.md](experiments/PHASE5B_OPTIMIZATION_REPORT.md) | Research | Optimized hybrid candidate generation down to 242.8 ms CPU latency while maintaining 92.08% Recall@10. |
| **Phase 6A**| Evidence Selection Optimization | [PHASE6A_EVIDENCE_SELECTION_REPORT.md](experiments/PHASE6A_EVIDENCE_SELECTION_REPORT.md) | Research | Evaluated 5 sentence selection strategies; confirmed Approach B (pure sentence BM25) achieves sub-250ms latency. |

---

## Limitations Identified & Future Roadmap

1. **Retriever Bottleneck**: Lexical BM25 achieves 92.55% Recall@10, meaning ~7.5% of claims with evidence are missed before verification begins if lexical overlap is weak. Research Phase 5B demonstrated that hybrid RRF bridges this gap, but requires dense embedding dependencies.
2. **Distractor Sensitivity in Synthetic Negative Training**: Phase 4 error analysis revealed that training on synthetic hard negatives containing entity swaps increases sensitivity to unrelated retrieved passages. Future iterations should couple the Phase 2 entity grounding filter before NLI scoring.
3. **Joint Document-Level Reasoning**: Current cross-encoder scoring evaluates individual sentence premises. Paragraph-level and cross-document multi-hop reasoning remain areas for future architectural enhancement.

---

## License & Attribution

- **Project License**: This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
- **SciFact Dataset**: Released by the Allen Institute for AI under the **Apache License 2.0** (Wadden et al., EMNLP 2020).
- **Base Pretrained Model**: `cross-encoder/nli-deberta-v3-small` is released under the **MIT License** (Heitmann et al., Microsoft DeBERTa-v3).
- **PubMed / MEDLINE Abstracts**: Used in experimental research (`data/experimental/corpus_expanded_10k.jsonl`). While NLM citation metadata is open government data, individual journal article abstracts are subject to copyright and licensing terms that vary by publisher; they are not tracked in Git.
- **Third-Party Dependencies**: Backend and frontend dependencies are subject to their respective upstream licenses (FastAPI: MIT, PyTorch: BSD-3, Transformers: Apache 2.0, React: MIT, Vite: MIT, Tailwind CSS: MIT). Any unverified third-party assets remain unverified until audited.
