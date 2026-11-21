# ResearchMap — Baseline NLP Evaluation Report

**Date & Timestamp**: 2026-10-28T16:34:58.812672
**Dataset**: SciFact (EMNLP 2020)
**Split Evaluated**: `SciFact Official Dev (claims_dev.jsonl)` (300 claims)
**Lexical Retrieval**: BM25Okapi (Top-5 Docs, Top-3 Sentences)
**NLI Model**: `cross-encoder/nli-deberta-v3-small`
**Hardware / Device**: `cpu` (nt Windows 11)

## 1. Executive Summary & Headline Metrics

| Metric | Baseline Value | Description |
| :--- | :--- | :--- |
| **Macro-F1 (Claim Verification)** | **0.4756** | Unweighted mean of F1 scores across 3 classes |
| **Weighted-F1** | 0.4796 | Class-prevalence weighted F1 score |
| **Evidence Retrieval Recall@5** | **0.8777** | % of claims where at least one gold doc is in Top-5 BM25 |
| **Evidence Retrieval Recall@10** | **0.9255** | % of claims where at least one gold doc is in Top-10 BM25 |
| **Evidence Sentence Precision** | 0.3239 | Precision of retrieved evidence sentences vs gold |
| **Evidence Sentence Recall** | 0.1557 | Recall of retrieved evidence sentences vs gold |
| **Evidence Sentence F1** | 0.2103 | Harmonic mean of sentence evidence P & R |

## 2. Per-Class Claim Verification Breakdown

| Class | Precision | Recall | F1-Score | Support (Gold Count) |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTS** | 0.8400 | 0.3387 | 0.4828 | 124.0 |
| **CONTRADICTS** | 0.3413 | 0.6719 | 0.4526 | 64.0 |
| **NOT ENOUGH INFORMATION** | 0.4677 | 0.5179 | 0.4915 | 112.0 |

## 3. Confusion Matrix

| True \ Pred | SUPPORTS | CONTRADICTS | NOT ENOUGH INFO |
| :--- | :---: | :---: | :---: |
| **SUPPORTS** | 42 | 36 | 46 |
| **CONTRADICTS** | 1 | 43 | 20 |
| **NOT ENOUGH INFORMATION** | 7 | 47 | 58 |

## 4. Evidence Retrieval Performance

- **Eligible Claims with Gold Evidence**: 188
- **Doc Recall@5 (Any Gold Doc)**: 87.77%
- **Doc Recall@10 (Any Gold Doc)**: 92.55%
- **Strict Doc Recall@5 (All Gold Docs)**: 84.04%
- **Strict Doc Recall@10 (All Gold Docs)**: 89.36%

## 5. Sentence Evidence Performance

- **Claims with Gold Sentences**: 188
- **True Positive Sentences**: 57
- **False Positive Sentences**: 119
- **False Negative Sentences**: 309
- **Sentence Evidence Precision**: 0.3239
- **Sentence Evidence Recall**: 0.1557
- **Sentence Evidence F1**: 0.2103

## 6. Inspectable Prediction Examples

### Example 1: Claim ID 1

- **Claim**: *"0-dimensional biomaterials show inductive properties."*
- **Gold Label**: `NOT ENOUGH INFORMATION`
- **Predicted Label**: `CONTRADICTS` (INCORRECT)
- **Confidence Score**: `0.9973`
- **Top Retrieved Evidence (Doc 10906636, Sent 1)**:
  > *"Here we show that the constitutive activity and constitutive endocytosis properties of US28 are separable entities in this viral chemokine receptor."*
  - Document Title: *The carboxyl terminus of human cytomegalovirus-encoded 7 transmembrane receptor US28 camouflages agonism by mediating constitutive endocytosis.*
  - BM25 Lexical Score: `3.22`
  - NLI Probabilities: SUPPORTS=0.000, CONTRADICTS=0.995, NEI=0.005

### Example 2: Claim ID 3

- **Claim**: *"1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants."*
- **Gold Label**: `SUPPORTS`
- **Predicted Label**: `CONTRADICTS` (INCORRECT)
- **Confidence Score**: `0.9930`
- **Top Retrieved Evidence (Doc 3672261, Sent 1)**:
  > *"To identify the relative effects of environmental and genetic factors on variation in the parameters of innate and adaptive immune cells in homeostatic conditions, we combined standardized flow cytometry of blood leukocytes and genome-wide DNA genotyping of 1,000 healthy, unrelated people of Western European ancestry."*
  - Document Title: *Natural variation in the parameters of innate immune cells is preferentially driven by genetic factors*
  - BM25 Lexical Score: `10.22`
  - NLI Probabilities: SUPPORTS=0.000, CONTRADICTS=0.000, NEI=0.999

### Example 3: Claim ID 5

- **Claim**: *"1/2000 in UK have abnormal PrP positivity."*
- **Gold Label**: `SUPPORTS`
- **Predicted Label**: `NOT ENOUGH INFORMATION` (INCORRECT)
- **Confidence Score**: `0.9770`
- **Top Retrieved Evidence (Doc 18617259, Sent 4)**:
  > *"These findings have major implications for future estimates and surveillance of vCJD in the UK."*
  - Document Title: *Research Letters*
  - BM25 Lexical Score: `7.26`
  - NLI Probabilities: SUPPORTS=0.001, CONTRADICTS=0.022, NEI=0.977

### Example 4: Claim ID 13

- **Claim**: *"5% of perinatal mortality is due to low birth weight."*
- **Gold Label**: `NOT ENOUGH INFORMATION`
- **Predicted Label**: `NOT ENOUGH INFORMATION` (CORRECT)
- **Confidence Score**: `0.9994`
- **Top Retrieved Evidence (Doc 1263446, Sent 11)**:
  > *"Low birth weight and short birth interval infants as well as perinatal health services factors, such as the availability of skilled birth attendance and postnatal care utilization should be taken into account when planning the interventions to reduce neonatal mortality in Indonesia."*
  - Document Title: *Determinants of neonatal mortality in Indonesia*
  - BM25 Lexical Score: `7.35`
  - NLI Probabilities: SUPPORTS=0.000, CONTRADICTS=0.001, NEI=0.999

### Example 5: Claim ID 42

- **Claim**: *"A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects."*
- **Gold Label**: `CONTRADICTS`
- **Predicted Label**: `CONTRADICTS` (CORRECT)
- **Confidence Score**: `0.9839`
- **Top Retrieved Evidence (Doc 18174210, Sent 10)**:
  > *"CONCLUSIONS The increased erythrocyte count and microcytosis in children homozygous for alpha(+)-thalassaemia may contribute substantially to their protection against SMA."*
  - Document Title: *Increased Microerythrocyte Count in Homozygous α+-Thalassaemia Contributes to Protection against Severe Malarial Anaemia*
  - BM25 Lexical Score: `6.49`
  - NLI Probabilities: SUPPORTS=0.004, CONTRADICTS=0.984, NEI=0.013

### Example 6: Claim ID 48

- **Claim**: *"A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection."*
- **Gold Label**: `CONTRADICTS`
- **Predicted Label**: `NOT ENOUGH INFORMATION` (INCORRECT)
- **Confidence Score**: `0.9963`
- **Top Retrieved Evidence (Doc 18617259, Sent 4)**:
  > *"These findings have major implications for future estimates and surveillance of vCJD in the UK."*
  - Document Title: *Research Letters*
  - BM25 Lexical Score: `9.31`
  - NLI Probabilities: SUPPORTS=0.000, CONTRADICTS=0.004, NEI=0.996

## 7. Environment & Reproducibility Details

- **Python Version**: `3.13.1`
- **Random Seed**: `42`
- **Packages**: `torch==2.14.1+cpu`, `transformers==5.20.2`, `rank_bm25==0.2.2`, `scikit-learn==1.9.1`
- **Dataset Source**: SciFact Official AI2 Release (`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`)
- **License**: Apache 2.0 (Allen Institute for AI)
