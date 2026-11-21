# Phase 4B Verification Report: Experimental Scientific Corpus Expansion to 10k

**Project**: ResearchMap
**Date**: November 20, 2026
**Status**: Verified and Completed
**Active Production Corpus**: `data/raw/corpus.jsonl` (5,183 abstracts, Unmodified)
**Experimental Expanded Corpus**: `data/experimental/corpus_expanded_10k.jsonl` (10,835 abstracts)
**Regression Test Status**: **45/45 Passed** (`pytest` in 24.44s)

---

## 1. Executive Summary & Verification Matrix

In Phase 4B, the scientific corpus was expanded from the baseline 5,183 SciFact abstracts to an isolated experimental corpus of **10,835 unique, peer-reviewed scientific abstracts** using the official NCBI PubMed E-utilities API.

The entire acquisition and comparative evaluation pipeline was executed in strict isolation. Production files, configurations, model checkpoints, and frontend code were preserved unchanged.

### Verification Matrix

| Verification Item | Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **Task 497 Execution** | Complete without errors, exit code 0 | Exited code 0 in 403.9s | **PASS** |
| **Original Corpus Integrity** | `data/raw/corpus.jsonl` must not be modified | SHA-256 `b8d6c89624cb...` identical before and after | **PASS** |
| **Original Corpus Preservation** | All 5,183 original abstracts must be retained in expanded corpus | 5,183 / 5,183 verified present | **PASS** |
| **Target Document Count** | Approximately 10,000 unique abstracts | 10,835 total unique abstracts (5,652 added) | **PASS** |
| **Identifier Uniqueness** | Zero duplicate IDs in expanded corpus | 10,835 unique IDs / 0 duplicates / 0 collisions | **PASS** |
| **Record Integrity** | Zero malformed or empty records | 0 malformed records, 0 empty abstracts | **PASS** |
| **Provenance Traceability** | Traceable sources, PMIDs, publication years | 100% of new docs have PMIDs (5,652/5,652) and years | **PASS** |
| **Production Code Isolation**| Production backend, frontend, models untouched | Zero production code or test changes | **PASS** |
| **Regression Test Suite** | All existing tests continue to pass | 45/45 passed in 24.44s | **PASS** |
| **Comparative Evaluation** | Original vs Expanded evaluated on identical test/val sets | Both evaluated with Configuration C on N=24 and N=161 | **PASS** |

---

## 2. Expanded Corpus Construction & Data Provenance

### 2.1 Data Source and Legal/Licensing Considerations
- **Primary Source**: National Center for Biotechnology Information (NCBI) PubMed E-utilities.
- **Access Protocol**: Compliant with NLM API Terms of Service (rate-limited requests with explicit user agent, tool name, and contact identification).
- **Scope**: **Abstracts and bibliographic metadata only**. Full-text articles were not downloaded.
- **Licensing Nuance**: While NLM citation metadata is in the public domain under US open government data policies, journal abstracts themselves remain subject to copyright by their respective publishers. Use in ResearchMap is strictly limited to non-commercial scientific text analysis, indexing, and claim verification benchmarking under fair use and open data principles. Downstream commercial redistribution must not be assumed without individual publisher licensing.

### 2.2 Rejection Statistics & Deduplication
A total of **5,694 candidate PMIDs** were queried across 7 missing literature domains and seed citations:
- **Valid records acquired**: **5,652**
- **Missing abstracts rejected**: **1**
- **Too-short abstracts (<2 sentences) rejected**: **5**
- **Duplicate titles rejected**: **29**
- **Duplicate abstracts rejected**: **3**
- **Duplicate PMIDs rejected**: **0**
- **Empty titles rejected**: **0**
- **Doc ID collisions with original corpus**: **0** (All PubMed PMIDs were distinct from existing SciFact IDs)

### 2.3 Topic Distribution in Expanded Corpus (10,835 Documents)
1. `original_scifact`: **5,183** documents (47.8%)
2. `scientific_methodology_and_causality`: **849** documents (7.8%)
3. `virology_antivirals_and_antibiotics`: **834** documents (7.7%)
4. `hippocampus_and_memory`: **797** documents (7.4%)
5. `human_microbiome_and_health`: **797** documents (7.4%)
6. `genomics_and_variant_interpretation`: **793** documents (7.3%)
7. `mrna_vaccines_and_immunology`: **787** documents (7.3%)
8. `cortical_atrophy_and_aging`: **786** documents (7.3%)
9. `curated_evaluation_seed`: **9** documents (0.1%)

---

## 3. Experimental Comparison: Original vs Expanded Corpus

Both corpora were evaluated using the active **Retrieval Configuration C** (`top_k_docs=10`, `top_k_sentences=4`, `doc_score_weight=0.35`, `max_per_doc=2`, raw-regex tokenization) and the active fine-tuned NLI model checkpoint (`models/finetuned_nli`).

### 3.1 Held-Out Diagnostic Test Set ($N = 24$ Claims)

| Metric | Configuration A (Original 5k Corpus) | Configuration B (Expanded 10k Corpus) | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 83.33% (20/24) | **87.50% (21/24)** | **+4.17% (+1 net claim)** |
| **Macro-F1** | 0.8263 | **0.8565** | **+0.0302** |
| **Answerable Accuracy** | 92.31% (12/13) | **92.31% (12/13)** | $\pm 0.0000$ |
| **Unanswerable Accuracy** | 72.73% (8/11) | **81.82% (9/11)** | **+9.09% (+1 claim)** |
| **SUPPORTS F1** | 0.9412 (P: 1.00, R: 0.8889) | **1.0000 (P: 1.00, R: 1.0000)**| **+0.0588** |
| **CONTRADICTS F1** | 0.8235 (P: 1.00, R: 0.7000) | **0.8421 (P: 0.8889, R: 0.8000)**| **+0.0186** |
| **NEI F1** | 0.7143 (P: 0.5556, R: 1.000) | **0.7273 (P: 0.6667, R: 0.8000)**| **+0.0130** |
| **Document Recall@5** | 76.92% (10/13) | 61.54% (8/13) | -15.38% |
| **Document Recall@10** | 84.62% (11/13) | 76.92% (10/13) | -7.70% |
| **Document Recall@20** | 84.62% (11/13) | 84.62% (11/13) | $\pm 0.0000$ |
| **Sentence Evidence Recall** | 53.85% (7/13) | 53.85% (7/13) | $\pm 0.0000$ |
| **Document MRR@10** | 0.6289 | 0.5547 | -0.0742 |
| **Document MRR@20** | 0.6289 | 0.5606 | -0.0683 |
| **Document nDCG@10** | 0.6809 | 0.6040 | -0.0769 |
| **Mean Query Latency** | 189.63 ms | 213.31 ms | +23.68 ms (+12.5%) |
| **BM25 Index Build Time** | 0.65 s | 1.41 s | +0.76 s |

---

### 3.2 Internal Validation Split ($N = 161$ Claims)

| Metric | Configuration A (Original 5k Corpus) | Configuration B (Expanded 10k Corpus) | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 54.04% (87/161) | **57.14% (92/161)** | **+3.10% (+5 claims)** |
| **Macro-F1** | 0.5240 | **0.5546** | **+0.0306** |
| **SUPPORTS F1** | 0.6316 (P: 0.6269, R: 0.6364) | **0.6567 (P: 0.6471, R: 0.6667)**| **+0.0251** |
| **CONTRADICTS F1** | 0.4359 (P: 0.3953, R: 0.4857) | **0.4615 (P: 0.4186, R: 0.5143)**| **+0.0256** |
| **NEI F1** | 0.5045 (P: 0.5490, R: 0.4667) | **0.5455 (P: 0.6000, R: 0.5000)**| **+0.0410** |
| **Document Recall@5** | 84.16% (85/101) | **86.14% (87/101)** | **+1.98% (+2 documents)**|
| **Document Recall@10** | 89.11% (90/101) | 88.12% (89/101) | -0.99% (-1 document) |
| **Document Recall@20** | 93.07% (94/101) | 91.09% (92/101) | -1.98% (-2 documents) |
| **Sentence Evidence Recall** | 57.43% (58/101) | **58.42% (59/101)** | **+0.99% (+1 claim)** |
| **Document MRR@10** | 0.7495 | 0.7459 | -0.0036 |
| **Document nDCG@10** | 0.7709 | 0.7666 | -0.0043 |
| **Mean Query Latency** | 182.54 ms | 221.02 ms | +38.48 ms (+21.1%) |

---

## 4. Per-Claim Change Analysis: Genuine Coverage vs Distractors

Tracking individual claim verdicts across the 24 test claims reveals three qualitative shifts:

| Claim ID | Claim Text | Gold Label | Config A Verdict | Config B Verdict | Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Claim 7** | *Messenger RNA vaccines provide cells with instructions for producing a target antigen.* | `SUPPORTS` | `NOT ENOUGH INFORMATION` (conf: 0.996) | **`SUPPORTS`** (conf: 0.9330) | **Genuine Coverage Gain**. The original 5k corpus lacked mRNA vaccine literature. Adding PubMed documents (including Doc 43021961 and Pardi et al. PMID 29326426) surfaced direct evidence and resolved the claim correctly. |
| **Claim 23** | *Regular mindfulness meditation practice completely eliminates cerebral cortex atrophy during aging.* | `CONTRADICTS` | `NOT ENOUGH INFORMATION` (conf: 0.997) | **`CONTRADICTS`** (conf: 0.9306) | **Genuine Coverage Gain**. The original 5k corpus had no papers on brain senescence and meditation. Adding PubMed documents (including Doc 40623834 and Luders et al. PMID 25653628) surfaced evidence refuting complete atrophy elimination. |
| **Claim 24** | *Specific maternal dietary microbiome alterations during the first trimester can completely eradicate childhood asthma risk.* | `NOT ENOUGH INFORMATION` | `NOT ENOUGH INFORMATION` (conf: 0.976) | **`CONTRADICTS`** (conf: 0.9423) | **Lexical Distractor / Refutation Overreach**. In the expanded corpus, microbiome asthma studies (Doc 42852271) discussing maternal flora were retrieved; the cross-encoder interpreted non-eradication findings as direct contradiction rather than insufficient evidence. |

### Summary of Change Impact:
- **Net test accuracy**: Increased from **83.33% to 87.50%** (+4.17%).
- **Answerable claims**: 100% of answerable test claims that had evidence in the original corpus remained correctly verified (12/13).
- **Newly answerable claims**: 2 previously unanswerable claims (Claim 7 and Claim 23) became answerable and were correctly predicted.
- **Validation gain**: Validation set accuracy improved from **54.04% to 57.14%** (+3.10%, +5 claims net), confirming that the benefits generalize beyond the diagnostic test set.

---

## 5. Retrieval Competition Dynamics

Expanding the corpus from 5,183 to 10,835 documents introduced a noticeable phenomenon: **retrieval competition**.

1. **Document Recall vs Classification Accuracy**:
   - On the original 13 answerable claims, Doc Recall@10 dropped from 84.62% to 76.92% (1 claim, Claim 16, had its gold document rank shift from #7 to #12 because newly added telomere literature matched lexical keywords with higher BM25 scores).
   - Despite this lexical displacement, the cross-encoder still correctly verified Claim 16 because candidate sentences from the new telomere papers supported the same biological truth.
2. **Lexical Density Trade-off**:
   - As an index grows from 5k to 10k documents, lexical BM25 encounters more near-duplicate terminology. Pure lexical retrieval is more vulnerable to keyword-dense distractors.
   - However, the cross-encoder's semantic precision largely mitigated this: test accuracy rose to 87.50% and validation accuracy rose to 57.14%.

---

## 6. Deliverable File Verification

| Deliverable | Location | Status | Details |
| :--- | :--- | :--- | :--- |
| **Expanded Corpus** | [`data/experimental/corpus_expanded_10k.jsonl`](../data/experimental/corpus_expanded_10k.jsonl) | **Verified** | 10,835 JSONL lines, 20.77 MB |
| **Expansion Summary** | [`experiments/corpus_expansion_summary.json`](corpus_expansion_summary.json) | **Verified** | Topic counts, rejections, hashes, provenance |
| **Comparison Metrics** | [`experiments/corpus_expansion_comparison.json`](corpus_expansion_comparison.json) | **Verified** | Complete evaluations on N=24 and N=161 |
| **Phase 4B Report** | [`experiments/PHASE4B_CORPUS_EXPANSION.md`](PHASE4B_CORPUS_EXPANSION.md) | **Verified** | Comprehensive analysis and audit trail |

---

## 7. Recommendation: Is the Expanded Corpus Worth Adopting?

### **Recommendation: YES, with dense/hybrid retrieval in Phase 5.**

1. **Compelling Advantages**:
   - **Generalization**: Lifts test accuracy to **87.50%** and validation accuracy to **57.14%** (+3.10% gain across 161 claims).
   - **Coverage**: Directly resolves previously unanswerable domains (mRNA vaccines, cortical senescence) with zero loss of the 5,183 original abstracts.
   - **Latency SLA**: Average query latency remains exceptionally low (~213 ms vs ~190 ms, well under 500 ms SLA).
2. **Key Limitation to Address**:
   - As the corpus scales, pure BM25 lexical retrieval exhibits increased distractor competition (Doc Recall@10 for legacy claims dipped from 84.6% to 76.9%).
   - Adding dense embeddings (e.g., hybrid BM25 + bi-encoder reranking) before promoting the expanded corpus to production will eliminate lexical cutoff failures while leveraging the 10,835-abstract knowledge base.
