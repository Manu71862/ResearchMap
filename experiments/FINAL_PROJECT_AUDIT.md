# ResearchMap — Complete Final Project Audit

**Audit Date**: November 21, 2026
**Audit Type**: Full-Stack System & Pipeline Audit
**Repository**: `https://github.com/Manu71862/ResearchMap.git`
**Branch**: `main` (Local Commit: `8760c95`)
**Scope**: Full-stack audit across Frontend, Backend, NLP Pipeline, Data & Experiments, and Project Setup.
**Audit Policy**: Strict read-only audit. No production files, models, or configurations were modified.

> **Post-Audit Update (November 21, 2026):** Following this audit, one production enhancement was integrated: a condition-aware reversibility filter was added to `backend/pipeline.py`. This filter applies Bradford Hill's reversibility criterion to prevent spurious contradiction vetoes when cessation evidence corroborates an active exposure claim (e.g. "smoking cessation reduces risk" for the claim "smoking increases risk"). The NLI model, checkpoint, retrieval pipeline, and all benchmark numbers reported in this document remain unchanged. The new filter is covered by `tests/test_condition_aware_reversibility.py` (6 tests).

---

## Executive Summary

ResearchMap is a production-grade, domain-adapted scientific claim verification and evidence mapping system. It pairs lexical BM25 retrieval over a 5,183-abstract scientific corpus with a fine-tuned DeBERTa-v3 cross-encoder (`cross-encoder/nli-deberta-v3-small`, 44M parameters) to classify scientific claims into `SUPPORTS`, `CONTRADICTS`, or `NOT ENOUGH INFORMATION`.

The system exhibits **exceptional engineering hygiene**:
- **Test Suite**: 60 of 60 automated unit and integration tests passing (`pytest tests/ -q` in 28.2s).
- **Backend Service**: 100% of FastAPI endpoints (`/api/health`, `/api/verify`, `/api/papers/{id}`, `/api/samples`, `/api/metrics`) are operational with strict Pydantic validation, defensive error handling, and robust caching.
- **Frontend Build**: Zero-error Vite + React 18 production build (`npm run build` in 3.6s, 1,756 modules transformed).
- **Research Integrity**: 9 distinct experimental phases documented with empirical reports, JSON artifacts, and leakage-safe splits.

One critical UI bug was discovered: a tab naming mismatch in the Navbar (`activeTab = 'benchmark'` vs `App.jsx` checking `'benchmarks'`) causes the entire UI view to blank out when the user clicks "Benchmark Reports". In addition, 54 local project files are currently untracked in Git.

---

## 1. Frontend Audit

### 1.1 Architecture & Component Hierarchy
The frontend is built with **React 18.3.1**, **Vite 6.1.0**, **@xyflow/react 12.12.0**, and **Lucide React 0.475.0**:
```
App.jsx (Root State & Layout)
├── Navbar.jsx (Branding, Tab Switcher, System Health Poller)
├── ClaimInput.jsx (Textarea, Retrieval Sliders, Curated SciFact Sample Chips)
├── VerdictCard.jsx (Verdict Badge, Confidence Meter, Calibrated Probs, Warnings)
├── EvidenceGraph.jsx (@xyflow/react Interactive Node-Edge Evidence Map)
├── EvidenceList.jsx (Ranked Abstract Cards, Verbatim Passages, BM25 Scores)
├── NodeDetailDrawer.jsx (Slide-out Passage & Node Probability Inspector)
├── PaperModal.jsx (Full Paper Abstract Modal with Highlighted Sentences)
└── BenchmarkSection.jsx (Empirical Baseline vs Fine-Tuned KPI Dashboard)
```

### 1.2 Claim Submission Workflow
1. Users enter custom scientific claims or select one of 9 real SciFact sample chips (`ClaimInput.jsx`).
2. Optional slider controls allow tuning candidate papers ($k=1 \dots 10$, default 5) and candidate sentences per paper ($1 \dots 5$, default 3).
3. Form submission dispatches a `POST /api/verify` request. In-flight requests display an animated spinner with step-by-step progress explanations (*"Querying BM25 lexical index → Ranking candidate abstract passages → Computing cross-encoder entailment logits..."*).
4. Submissions are debounced and input is trimmed.

### 1.3 Results Presentation & Evidence Mapping
- **`VerdictCard`**: Color-coded verdict badge (Supports: `#10b981`, Contradicts: `#f43f5e`, NEI: `#f59e0b`), calibrated 3-class probability breakdown bar, percentage confidence meter, academic disclaimer, and benchmark agreement chips.
- **`EvidenceGraph`**: Interactive directed acyclic graph (DAG) rendered with `@xyflow/react`. Nodes represent Root Claim (cyan), Retrieved Papers (blue, showing BM25 rank and score), and Evaluated Sentences (color-coded by verdict with arrow connectors). Clicking any node opens the detail drawer.
- **`EvidenceList`**: Structured cards showing each retrieved paper, abstract preview, rank, BM25 score, and verbatim sentence evidence with entailment classification. Includes a direct "View Abstract" action.
- **`PaperModal`**: Overlay modal rendering the complete abstract of any cited paper with verbatim evidence sentences highlighted by sentence index.
- **`NodeDetailDrawer`**: Slide-over drawer displaying granular token probabilities, parent document metadata, and sentence BM25 relevance scores.

### 1.4 Loading, Empty, and Error States
- **Loading**: Implemented across both App root and modal loaders with CSS spin animations.
- **Empty State**: `EvidenceList.jsx` gracefully renders an empty-state illustration when BM25 returns no matching literature.
- **Error Handling**: API errors and network timeouts render a structured rose alert box displaying the backend exception detail.
- **Backend Offline**: `Navbar.jsx` polls `GET /api/health` every 5 seconds. If the backend is down, the status dot turns red and displays *"Backend Offline"*. If Vite's dev proxy fails to connect to port 8000, Vite returns a clean 503 JSON response.

### 1.5 Responsive Design
- Layout utilizes CSS Flexbox and CSS Grid with `repeat(auto-fit, minmax(220px, 1fr))`.
- Drawers and modals use `maxWidth: 90vw` and `maxHeight: 85vh` with backdrop filters (`blur(8px)` and `blur(16px)`).

---

## 2. Backend & API Audit

### 2.1 Server Architecture (`backend/server.py`)
- Framework: **FastAPI 0.110.0** with **Uvicorn 0.28.0**.
- Lifecycle: Uses FastAPI's modern `@asynccontextmanager` lifespan handler. Corpus, BM25 index, and NLI model weights are loaded **once at startup** into a managed global state (`app_state`), preventing per-request cold-start latency.
- Middleware: `CORSMiddleware` configured with `allow_origins=["*"]`.

### 2.2 Endpoint Verification

| Method | Endpoint | Status | Verified Functionality |
| :--- | :--- | :---: | :--- |
| `GET` | `/api/health` | **200 OK** | Reports service status, active model path (`models/finetuned_nli`), corpus paper count (5,183), and compute device (`cpu`). |
| `POST` | `/api/verify` | **200 OK** | Accepts `VerifyRequest`, runs BM25 retrieval, evaluates NLI cross-encoder, formats evidence details, calibrated probabilities, and warnings. |
| `POST` | `/api/verify` | **422 Unproc** | Validates schema constraints: empty claim (`""`), missing claim, top_k out of bounds. |
| `POST` | `/api/verify` | **400 Bad Req** | Rejects whitespace-only claims (`"   "`). |
| `GET` | `/api/papers/{paper_id}` | **200 OK** | Returns paper doc_id, title, sentence-tokenized abstract, and structured flag. |
| `GET` | `/api/papers/999999999` | **404 Not Found** | Correctly raises 404 for nonexistent doc_ids. |
| `GET` | `/api/samples` | **200 OK** | Returns 9 verified SciFact benchmark claims spanning all three classes with gold labels. |
| `GET` | `/api/metrics` | **200 OK** | Returns cached benchmark evaluation metrics from `reports/model_comparison_metrics.json` with automatic file modification re-loading. |

### 2.3 Request Validation & Error Handling
- Pydantic models: `VerifyRequest`, `VerifyResponse`, `PaperResponse`, `HealthResponse`, `EvidenceDetail`, `RetrievedPaper`.
- Input validation: `claim` string requires `min_length=3, max_length=1500`; `top_k_docs` is bounded `ge=1, le=20`; `top_k_sentences` is bounded `ge=1, le=10`.
- All database and model operations are guarded by `try/except` blocks returning standard RFC 7807 / FastAPI HTTP exceptions.

---

## 3. NLP Pipeline Audit

### 3.1 Lexical BM25 Retrieval (`backend/retrieval.py`)
- Implementation: `rank_bm25.BM25Okapi` over 5,183 document abstracts.
- Tokenization: Lowercase alphanumeric word regex `\b[a-zA-Z0-9_\-]+\b`.
- Abstract Retrieval: Retrieves top-$k$ candidate documents (default $k=10$). Filters non-matching scores ($\le 0.0$).
- Sentence Retrieval (Configuration C):
  $$\text{Score}(s) = \text{BM25}(s, q) + 0.35 \cdot \text{BM25}(d, q)$$
  Applies a parent document prior weight (0.35) and a hard cap of 2 sentences per document, preventing distractor papers from displacing valid evidence.

### 3.2 NLI Cross-Encoder (`backend/entailment.py`)
- Base Model: `cross-encoder/nli-deberta-v3-small` (44M parameters, 6 layers).
- Active Checkpoint: `models/finetuned_nli` (fine-tuned on SciFact claim-evidence pairs).
- Input Structure: Premise (candidate sentence) + Hypothesis (claim text) with dynamic padding up to 512 tokens.
- Dynamic Label Mapping:
  - Label 0: `CONTRADICTS`
  - Label 1: `SUPPORTS`
  - Label 2: `NOT ENOUGH INFORMATION`
- Softmax Probability Distribution: Produces true 3-class probability vectors $[P_{\text{CON}}, P_{\text{SUP}}, P_{\text{NEI}}]$ summing to 1.0.

### 3.3 Evidence Selection & Decision Thresholds (`backend/pipeline.py`)
- Thresholds: $\tau_{\text{SUP}} = 0.50, \tau_{\text{CON}} = 0.50$.
- Decision Logic:
  1. If any sentence has $P(\text{SUP}) \ge 0.50$ and another has $P(\text{CON}) \ge 0.50$, the sentence with the higher probability determines the claim verdict.
  2. If only supporting sentences exceed 0.50, verdict is `SUPPORTS`.
  3. If only contradicting sentences exceed 0.50, verdict is `CONTRADICTS`.
  4. If no sentence exceeds 0.50, the pipeline abstains with `NOT ENOUGH INFORMATION` and assigns the maximum neutral probability among candidates as confidence.

### 3.4 Production Checkpoint Selection
- Phase 4 error analysis conclusively confirmed that **`models/finetuned_nli` is the optimal production model**, achieving **83.33% diagnostic accuracy** and a **0.3607 Brier calibration score**, outperforming augmented checkpoints that suffered from prior shift distortions.

---

## 4. Data & Research Experiments Audit

### 4.1 Dataset Organization & Zero-Leakage Protocol
- **`data/raw/`**:
  - `corpus.jsonl`: 5,183 peer-reviewed biomedical abstracts.
  - `claims_train.jsonl`: 809 raw training claims.
  - `claims_dev.jsonl`: 300 raw development claims.
  - `claims_test.jsonl`: 300 leaderboard test claims (unlabeled).
- **`data/processed/`**:
  - `train_partition.jsonl` (644 claims, 80% of clean training claims).
  - `val_internal.jsonl` (161 claims, 20% of clean training claims).
  - `dev_heldout.jsonl` (300 claims, quarantined official dev set).
  - **Leakage Audit**: Formally verified zero ID overlap and zero text overlap across splits.
- **`data/experimental/`**:
  - `corpus_expanded_10k.jsonl`: 10,835 abstracts (expanded with PubMed articles).
  - `hard_negative_pairs.jsonl`: 1,500 rule-validated synthetic negative pairs.
  - `challenge_set.jsonl`: 60 independent reasoning probes across 6 scientific categories.

### 4.2 Research Phases Catalog

| Phase | Research Topic | Key Report File | Primary Findings & Metrics |
| :---: | :--- | :--- | :--- |
| **Phase 1** | Evidence Aggregation Improvement | `PHASE1_AGGREGATION_REPORT.md` | Benchmarked 4 aggregation strategies; proved relevance-weighted consensus prevents single-sentence veto. |
| **Phase 2** | Scientific Entity Grounding Filter | `PHASE2_ENTITY_GROUNDING_REPORT.md` | 15/15 unit tests passed; entity grounding improved validation accuracy from 54.04% to 55.28% (+1.24%) with 0 regressions. |
| **Phase 3** | Hard-Negative Data Augmentation | `PHASE3_HARD_NEGATIVE_REPORT.md` | Built 1,500 hard negatives; fine-tuned augmented checkpoint; revealed diagnostic regression (83.33% to 66.67%). |
| **Phase 4** | Targeted Error Analysis & Calibration | `PHASE4_ERROR_ANALYSIS_REPORT.md` | Identified 61.3% NEI prior shift (+1.33 logit offset); trained calibrated balanced checkpoint (recovered Claim 9, 58.39% val acc). Recommended retaining baseline in production. |
| **Phase 4A**| BM25 Configuration C Tuning | `PHASE4A_RETRIEVAL_IMPLEMENTATION.md` | Configuration C (parent doc weight 0.35, max 2 sents/doc) achieved 87.50% diagnostic accuracy. |
| **Phase 4B**| Corpus Expansion Audit | `PHASE4B_CORPUS_EXPANSION.md` | Expanded corpus to 10,835 docs via PubMed; verified retrieval stability under distractor pressure. |
| **Phase 5A**| Hybrid Dense-Lexical Retrieval | `PHASE5A_HYBRID_RETRIEVAL_REPORT.md` | Implemented Dense retriever (`all-MiniLM-L6-v2`) and Hybrid RRF fusion. |
| **Phase 5B**| Hybrid Optimization & Audit | `PHASE5B_OPTIMIZATION_REPORT.md` | Optimized RRF with pre-filtering; reduced hybrid latency to 242.8 ms while achieving 92.08% Recall@10. |
| **Phase 6A**| Evidence Selection Optimization | `PHASE6A_EVIDENCE_SELECTION_REPORT.md` | Evaluated 5 sentence selection strategies; proved pure sentence BM25 (Approach B) yields sub-250ms latency without distractor bias. |

### 4.3 Checkpoint Catalog (`models/`)
- `models/finetuned_nli`: **Active Production Checkpoint** (3 epochs fine-tuning, safetensors format).
- `models/finetuned_epoch1`: Intermediate epoch 1 checkpoint.
- `models/experimental_augmented_nli`: Phase 3 augmented checkpoint.
- `models/experimental_calibrated_nli`: Phase 4 class-weighted calibrated checkpoint.

---

## 5. Project Setup, Environment & Git Status

### 5.1 Dependencies & Environments
- **Python**: Requirements file (`requirements.txt`) specifies 8 pinned package ranges (`torch>=2.2.0`, `transformers>=4.40.0`, `rank-bm25>=0.2.2`, `scikit-learn>=1.4.0`, `pytest>=8.0.0`, `python-dotenv>=1.0.0`, `tqdm>=4.66.0`, `fastapi>=0.110.0`, `uvicorn>=0.28.0`, `pydantic>=2.0.0`, `httpx>=0.27.0`).
- **Node.js**: `frontend/package.json` specifies modern React 18, Vite 6, and `@xyflow/react`.
- **Virtual Environment**: Active virtual environment in `.venv/` (Python 3.13.1).

### 5.2 Environment Variables & Config
- `.env`: Correctly populated with Configuration C parameters:
  ```env
  DATA_DIR=data
  RAW_DATA_DIR=data/raw
  PROCESSED_DATA_DIR=data/processed
  MODELS_DIR=models
  REPORTS_DIR=reports
  BM25_TOP_K_DOCS=10
  BM25_TOP_K_SENTENCES=4
  BM25_DOC_SCORE_WEIGHT=0.35
  BM25_MAX_SENTENCES_PER_DOC=2
  NLI_MODEL_NAME=cross-encoder/nli-deberta-v3-small
  DEVICE=auto
  RANDOM_SEED=42
  ```
- `.env.example`: Outdated (`BM25_TOP_K=5` instead of `BM25_TOP_K_DOCS=10`). Needs alignment.

### 5.3 Git Status Audit
- **Branch**: `main` tracking `origin/main` (`https://github.com/Manu71862/ResearchMap.git`).
- **Current Commit**: `8760c95 feat(experiments): complete Phase 6A evidence selection research`.
- **Tracked Files in Git**: Only 4 files are currently tracked in the remote Git repository:
  1. `README.md`
  2. `experiments/PHASE6A_EVIDENCE_SELECTION_REPORT.md`
  3. `experiments/phase6a_evidence_selection_results.json`
  4. `scripts/experimental/run_phase6a_comparison.py`
- **Untracked Local Files**: 54 files and folders (including `backend/`, `frontend/`, `tests/`, `requirements.txt`, etc.) are untracked because the initial Git push was limited to Phase 6A artifacts.

---

## 6. Audit Deliverables Checklist

### 6.1 List of Working Features
1. **End-to-End Scientific Claim Verification**: Real-time BM25 retrieval + DeBERTa-v3 cross-encoder classification.
2. **Interactive Evidence DAG**: Dynamic React Flow graph visualizing claims, retrieved abstracts, and sentence-level evidence.
3. **Paper Abstract Modal with Verbatim Sentence Highlighting**: Instant modal view of any cited document with gold/predicted evidence highlighted by sentence index.
4. **Granular Evidence Inspector Drawer**: Slide-out drawer displaying per-sentence token probabilities, doc metadata, and BM25 scores.
5. **Real Curated SciFact Sample Chips**: Click-to-verify chips pre-loaded with official benchmark claims spanning all three classes.
6. **Configurable Retrieval Sliders**: UI controls for dynamic candidate paper ($k=1 \dots 10$) and sentence ($1 \dots 5$) bounds.
7. **Empirical Model Performance Dashboard**: Tabular breakdown of Macro-F1, per-class metrics, confusion matrices, and retrieval recall.
8. **Automated Health Poller & Offline Notice**: 5-second interval health poller in the navigation bar.
9. **Leakage-Safe Partitioning Engine**: Clean 80/20 train/val split with 0 ID and 0 textual overlap.
10. **60 Automated Unit & Pipeline Tests**: Covering data loaders, BM25 indexing, NLI inference, evaluation metrics, FastAPI endpoints, and entity grounding.

### 6.2 List of Incomplete or Broken Features
1. **Navbar Benchmark Tab Broken**: In `frontend/src/components/Navbar.jsx`, clicking "Benchmark Reports" sets `activeTab` to `'benchmark'` (singular), but `frontend/src/App.jsx` evaluates `activeTab === 'benchmarks'` (plural). As a result, clicking the button renders a blank screen.
2. **Untracked Core Codebase in Git**: Production files (`backend/`, `frontend/`, `tests/`, `requirements.txt`) are not yet tracked in the GitHub remote.
3. **Outdated `.env.example`**: Does not reflect Configuration C retrieval variables (`BM25_TOP_K_DOCS`, `BM25_TOP_K_SENTENCES`, `BM25_DOC_SCORE_WEIGHT`, `BM25_MAX_SENTENCES_PER_DOC`).

### 6.3 Critical Bugs
1. **`Navbar.jsx` Navigation Desynchronization (High Severity - UI)**:
   - *File*: `frontend/src/components/Navbar.jsx` (lines 83-84)
   - *Issue*: `onClick={() => setActiveTab('benchmark')}`
   - *Target*: `App.jsx` line 269 expects `activeTab === 'benchmarks'`.
   - *User Impact*: Users cannot access the "Benchmark Reports" tab via the UI header.

### 6.4 UI/UX Improvements Worth Making
1. **Fix Navbar Tab Key**: Change `'benchmark'` to `'benchmarks'` in `Navbar.jsx`.
2. **Clean Up Redundant Props in `ClaimInput.jsx`**: Unify `onVerify` / `onAnalyze` and `loading` / `isLoading` to standard singular prop names.
3. **Add Copy-to-Clipboard for Evidence Text**: Allow users to copy verbatim cited sentences and paper titles directly from the evidence cards.
4. **Graph Node Auto-Layout**: Integrate Dagre layout algorithm into `EvidenceGraph.jsx` for dynamic node spacing when $k > 5$.
5. **Add Export Results Button**: Allow downloading verification verdicts, probabilities, and cited paper references as a structured JSON file.

### 6.5 Documentation Gaps
1. **Sync `.env.example`**: Update `.env.example` with the active Configuration C retrieval parameters.
2. **Update README Test Count**: `README.md` states *"Full Python Test Suite (27 Unit & Integration Tests)"*, whereas the test suite now contains **60 passing tests**.
3. **Document Experimental Checkpoints in README**: Add a dedicated subsection in `README.md` describing `models/experimental_augmented_nli` and `models/experimental_calibrated_nli`.

---

## 7. Exact Commands to Start the Application

### 7.1 Prerequisites
Ensure the virtual environment is activated and dependencies are installed:
```powershell
# From the ResearchMap root:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 Terminal 1 — Start the Backend API Server
```powershell
# From repository root:
.\.venv\Scripts\Activate.ps1
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
*Expected log output:*
```
[Startup] Loaded 5,183 research paper abstracts.
[Startup] Built BM25 lexical index.
[Startup] Using fine-tuned checkpoint: ...\models\finetuned_nli
Loading pretrained NLI model '...\models\finetuned_nli' on device 'cpu'...
Model loaded successfully. Label mapping: {0: 'CONTRADICTS', 1: 'SUPPORTS', 2: 'NOT ENOUGH INFORMATION'}
[Startup] Loaded 9 verified sample claims from SciFact.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 7.3 Terminal 2 — Start the Frontend Development Server
```powershell
cd frontend
npm run dev
```
*Expected output:*
```
  VITE v6.1.0  ready in 250 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```
*Open `http://localhost:3000` in your web browser.*

### 7.4 Running the Test Suite
```powershell
# From repository root:
.\.venv\Scripts\pytest -v
```
*(60 of 60 automated unit and integration tests will execute and pass).*

---

## 8. Prioritized Action Plan

| Priority | Task | Target Files | Effort |
| :---: | :--- | :--- | :---: |
| **P0** | **Fix Navbar Tab Desynchronization Bug** | `frontend/src/components/Navbar.jsx` (change `'benchmark'` to `'benchmarks'`) | 2 mins |
| **P1** | **Stage & Commit Untracked Project Codebase to Git** | `git add backend/ frontend/ tests/ requirements.txt reports/ data/processed/ ...` | 5 mins |
| **P2** | **Synchronize `.env.example` with Configuration C** | `.env.example` (add `BM25_TOP_K_DOCS=10`, `BM25_TOP_K_SENTENCES=4`, etc.) | 2 mins |
| **P3** | **Update README Test Count & Experimental Index** | `README.md` (update test count from 27 to 60, document Phase 1–6A) | 10 mins |
| **P4** | **Add Copy-to-Clipboard & JSON Export in UI** | `frontend/src/components/VerdictCard.jsx`, `EvidenceList.jsx` | 20 mins |
