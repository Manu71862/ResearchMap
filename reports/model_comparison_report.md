# ResearchMap — Phase 2 Model Comparison Report

**Date**: 2026-10-28 16:35:57
**Split Evaluated**: `SciFact Official Dev (claims_dev.jsonl)` (300 claims)
**Baseline Model**: `cross-encoder/nli-deberta-v3-small` (Zero-Shot)
**Fine-Tuned Model**: `models/finetuned_nli` (Trained on SciFact train partition)

## 1. Executive Comparison Table

| Metric | Zero-Shot Baseline | Fine-Tuned Model | Delta (Absolute) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Claim Verification Macro-F1** | **0.4756** | **0.6262** | **+0.1505** | **+31.65%** |
| **Weighted-F1** | 0.4796 | 0.6385 | +0.1589 | +33.14% |
| **Evidence Doc Recall@5 (BM25)** | 87.77% | 87.77% | +0.00% | Shared Retriever |
| **Evidence Doc Recall@10 (BM25)** | 92.55% | 92.55% | +0.00% | Shared Retriever |
| **Sentence Evidence Precision** | 0.3239 | 0.4263 | +0.1025 | - |
| **Sentence Evidence Recall** | 0.1557 | 0.2213 | +0.0656 | - |
| **Sentence Evidence F1** | 0.2103 | 0.2914 | +0.0810 | - |

## 2. Per-Class Performance Breakdown

| Class | Baseline Precision | Baseline Recall | Baseline F1 | Fine-Tuned Precision | Fine-Tuned Recall | Fine-Tuned F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SUPPORTS** | 0.840 | 0.339 | 0.483 | 0.716 | 0.669 | **0.692** | **+0.209** |
| **CONTRADICTS** | 0.341 | 0.672 | 0.453 | 0.527 | 0.609 | **0.565** | **+0.113** |
| **NOT ENOUGH INFORMATION** | 0.468 | 0.518 | 0.492 | 0.627 | 0.616 | **0.622** | **+0.130** |

## 3. Confusion Matrices

### Baseline (Zero-Shot) Confusion Matrix:
| True \ Pred | SUPPORTS | CONTRADICTS | NOT ENOUGH INFO |
| :--- | :---: | :---: | :---: |
| **SUPPORTS** | 42 | 36 | 46 |
| **CONTRADICTS** | 1 | 43 | 20 |
| **NOT ENOUGH INFORMATION** | 7 | 47 | 58 |

### Fine-Tuned Model Confusion Matrix:
| True \ Pred | SUPPORTS | CONTRADICTS | NOT ENOUGH INFO |
| :--- | :---: | :---: | :---: |
| **SUPPORTS** | 83 | 14 | 27 |
| **CONTRADICTS** | 11 | 39 | 14 |
| **NOT ENOUGH INFORMATION** | 22 | 21 | 69 |
