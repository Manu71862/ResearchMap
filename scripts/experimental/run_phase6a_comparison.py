"""
Phase 6A: Evidence Selection and NLI Classification Improvement
Controlled Benchmark comparing 5 Evidence Selection and Aggregation Approaches:
A. Existing Sentence Selection (Baseline Pipeline: sent_bm25 + 0.35 * doc_bm25, max_per_doc=2)
B. BM25-based Sentence Selection (Pure sentence lexical match, doc_score_weight=0.0)
C. Hybrid Document Score + Sentence Relevance Selection (Global RRF prior + BM25 sentence score)
D. Evidence Diversity Selection (MMR Lexical Jaccard Diversity Penalty + Cross-Document Diversification)
E. Multi-Sentence Evidence Aggregation (Document-bounded multi-sentence premise evaluation)

Evaluated under identical conditions:
- Expanded 10,835-document corpus (data/experimental/corpus_expanded_10k.jsonl)
- Pretrained fine-tuned NLI checkpoint (models/finetuned_nli)
- 24 held-out diagnostic claims (experiments/new_claims.jsonl)
- 161 validation claims (data/processed/val_internal.jsonl)
- Output saved to experiments/phase6a_evidence_selection_results.json
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np

from backend.config import Config
from backend.data_loader import (
    load_corpus,
    load_claims,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.retrieval import BM25Retriever, tokenize
from backend.entailment import NLIEntailmentModel
from scripts.experimental.dense_retriever import DenseRetriever
from scripts.experimental.optimized_hybrid_retriever import OptimizedHybridRetriever

EXPANDED_CORPUS_PATH = BASE_DIR / "data" / "experimental" / "corpus_expanded_10k.jsonl"
TEST_CLAIMS_PATH = BASE_DIR / "experiments" / "new_claims.jsonl"
VAL_CLAIMS_PATH = BASE_DIR / "data" / "processed" / "val_internal.jsonl"
CACHE_DIR = BASE_DIR / "experiments" / "cache"
EMB_PATH = CACHE_DIR / "doc_embeddings_minilm_10k.npy"
IDS_PATH = CACHE_DIR / "doc_ids_10k.json"
RESULTS_JSON_PATH = BASE_DIR / "experiments" / "phase6a_evidence_selection_results.json"

TOP_K_DOCS = 10
TOP_K_SENTENCES = 4

def jaccard_similarity(text1: str, text2: str) -> float:
    t1 = set(tokenize(text1))
    t2 = set(tokenize(text2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = CANONICAL_LABELS) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / n

    per_class = {}
    f1_list = []
    cm = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    cm_matrix = [[cm[l1][l2] for l2 in labels] for l1 in labels]

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm_matrix,
        "total": n,
        "correct": correct
    }

class EvidenceSelectors:
    def __init__(self, corpus: Dict[int, Dict[str, Any]], bm25: BM25Retriever, hybrid_retriever: OptimizedHybridRetriever):
        self.corpus = corpus
        self.bm25 = bm25
        self.hybrid_retriever = hybrid_retriever

    def select_approach_a(self, query: str, cand_docs: List[Dict[str, Any]], top_k: int = TOP_K_SENTENCES) -> List[Dict[str, Any]]:
        """A. Existing Sentence Selection (sent_bm25 + 0.35 * doc_bm25, max 2 per doc)."""
        return self.bm25.retrieve_sentences(
            query, cand_docs, top_k_sentences=top_k, doc_score_weight=0.35, max_per_doc=2
        )

    def select_approach_b(self, query: str, cand_docs: List[Dict[str, Any]], top_k: int = TOP_K_SENTENCES) -> List[Dict[str, Any]]:
        """B. BM25-based Sentence Selection (pure sentence BM25, doc_score_weight=0.0, max 2 per doc)."""
        return self.bm25.retrieve_sentences(
            query, cand_docs, top_k_sentences=top_k, doc_score_weight=0.0, max_per_doc=2
        )

    def select_approach_c(self, query: str, cand_docs: List[Dict[str, Any]], top_k: int = TOP_K_SENTENCES) -> List[Dict[str, Any]]:
        """C. Hybrid Document Score + Sentence Relevance Selection across all candidate docs."""
        return self.hybrid_retriever.retrieve_sentences(
            query, cand_docs, top_k_sentences=top_k, doc_score_weight=0.35, max_per_doc=2
        )

    def select_approach_d(self, query: str, cand_docs: List[Dict[str, Any]], top_k: int = TOP_K_SENTENCES) -> List[Dict[str, Any]]:
        """D. Evidence Diversity Selection (MMR diversity penalty + cross-document diversification)."""
        raw_candidates = []
        for d in cand_docs:
            did = d["doc_id"]
            d_prior = d.get("rrf_score", d.get("bm25_score", 0.0))
            for sidx, stext in enumerate(d.get("abstract", [])):
                tokens = tokenize(stext)
                if not tokens:
                    continue
                raw_candidates.append({
                    "doc_id": did,
                    "sentence_idx": sidx,
                    "text": stext,
                    "d_prior": d_prior
                })
        if not raw_candidates or not query:
            return []

        tokenized_sents = [tokenize(s["text"]) for s in raw_candidates]
        from rank_bm25 import BM25Okapi
        bm25_s = BM25Okapi(tokenized_sents)
        q_tokens = tokenize(query)
        scores = bm25_s.get_scores(q_tokens)
        max_s = max(scores) if max(scores) > 0 else 1.0

        for i, s in enumerate(raw_candidates):
            norm_s = scores[i] / max_s
            s["rel_score"] = float(norm_s + 0.35 * s["d_prior"])
            s["bm25_score"] = float(scores[i])

        selected = []
        doc_counts = {}
        remaining = list(raw_candidates)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = -1e9
            for idx, cand in enumerate(remaining):
                did = cand["doc_id"]
                doc_pen = 0.30 if doc_counts.get(did, 0) >= 1 else 0.0

                if not selected:
                    mmr = cand["rel_score"] - doc_pen
                else:
                    max_sim = max(jaccard_similarity(cand["text"], sel["text"]) for sel in selected)
                    mmr = 0.70 * cand["rel_score"] - 0.30 * max_sim - doc_pen

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0 and best_mmr > 0.0:
                chosen = remaining.pop(best_idx)
                selected.append(chosen)
                doc_counts[chosen["doc_id"]] = doc_counts.get(chosen["doc_id"], 0) + 1
            else:
                break

        return selected

    def select_approach_e(self, query: str, cand_docs: List[Dict[str, Any]], top_k: int = TOP_K_SENTENCES) -> List[Dict[str, Any]]:
        """E. Multi-Sentence Evidence Candidate Selection (uses Hybrid selection base)."""
        return self.select_approach_c(query, cand_docs, top_k=top_k)

def run_nli_standard_inference(
    nli_model: NLIEntailmentModel,
    sents: List[Dict[str, Any]],
    claim_text: str,
    entail_thresh: float = 0.50,
    contradict_thresh: float = 0.50,
) -> Tuple[str, float, List[Dict[str, Any]], List[Tuple[int, int]]]:
    if not sents:
        return LABEL_NEI, 0.0, [], []

    pairs = [(s["text"], claim_text) for s in sents]
    nli_outs = nli_model.predict_batch(pairs)

    evidence_details = []
    best_sup = None
    best_con = None

    for s, out in zip(sents, nli_outs):
        detail = {
            "doc_id": s["doc_id"],
            "sentence_idx": s["sentence_idx"],
            "text": s["text"],
            "probabilities": out["probabilities"],
            "prediction": out["prediction"],
            "confidence": out["confidence"]
        }
        evidence_details.append(detail)

        p_sup = out["probabilities"][LABEL_SUPPORTS]
        p_con = out["probabilities"][LABEL_CONTRADICTS]

        if p_sup >= entail_thresh:
            if best_sup is None or p_sup > best_sup["probabilities"][LABEL_SUPPORTS]:
                best_sup = detail
        if p_con >= contradict_thresh:
            if best_con is None or p_con > best_con["probabilities"][LABEL_CONTRADICTS]:
                best_con = detail

    if best_sup and best_con:
        s_prob = best_sup["probabilities"][LABEL_SUPPORTS]
        c_prob = best_con["probabilities"][LABEL_CONTRADICTS]
        if s_prob >= c_prob:
            return LABEL_SUPPORTS, s_prob, evidence_details, [(best_sup["doc_id"], best_sup["sentence_idx"])]
        else:
            return LABEL_CONTRADICTS, c_prob, evidence_details, [(best_con["doc_id"], best_con["sentence_idx"])]
    elif best_sup:
        return LABEL_SUPPORTS, best_sup["probabilities"][LABEL_SUPPORTS], evidence_details, [(best_sup["doc_id"], best_sup["sentence_idx"])]
    elif best_con:
        return LABEL_CONTRADICTS, best_con["probabilities"][LABEL_CONTRADICTS], evidence_details, [(best_con["doc_id"], best_con["sentence_idx"])]
    else:
        return LABEL_NEI, 0.0, evidence_details, []

def run_nli_multi_sentence_inference(
    nli_model: NLIEntailmentModel,
    corpus: Dict[int, Dict[str, Any]],
    sents: List[Dict[str, Any]],
    claim_text: str,
    entail_thresh: float = 0.50,
    contradict_thresh: float = 0.50,
) -> Tuple[str, float, List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    Approach E: Multi-Sentence Premise Aggregation.
    Evaluates individual sentences alongside document-bounded multi-sentence premises
    to prevent isolated out-of-context misclassifications.
    """
    if not sents:
        return LABEL_NEI, 0.0, [], []

    # 1. Individual evaluations
    pairs = [(s["text"], claim_text) for s in sents]
    nli_outs = nli_model.predict_batch(pairs)

    evidence_details = []
    best_sup = None
    best_con = None

    for s, out in zip(sents, nli_outs):
        detail = {
            "doc_id": s["doc_id"],
            "sentence_idx": s["sentence_idx"],
            "text": s["text"],
            "probabilities": out["probabilities"],
            "prediction": out["prediction"],
            "confidence": out["confidence"]
        }
        evidence_details.append(detail)
        p_sup = out["probabilities"][LABEL_SUPPORTS]
        p_con = out["probabilities"][LABEL_CONTRADICTS]

        if p_sup >= entail_thresh:
            if best_sup is None or p_sup > best_sup["probabilities"][LABEL_SUPPORTS]:
                best_sup = detail
        if p_con >= contradict_thresh:
            if best_con is None or p_con > best_con["probabilities"][LABEL_CONTRADICTS]:
                best_con = detail

    # 2. Document-bounded multi-sentence evaluation
    doc_sents: Dict[int, List[Dict[str, Any]]] = {}
    for s in sents:
        doc_sents.setdefault(s["doc_id"], []).append(s)

    multi_pairs = []
    doc_meta = []
    for did, slist in doc_sents.items():
        doc_info = corpus.get(did, {})
        title = doc_info.get("title", "")
        # Concatenate sentences from this document
        joined_text = " ".join(s["text"] for s in slist)
        if title:
            joined_text = f"{title}. {joined_text}"
        multi_pairs.append((joined_text, claim_text))
        doc_meta.append((did, [s["sentence_idx"] for s in slist]))

    multi_outs = nli_model.predict_batch(multi_pairs) if multi_pairs else []

    best_multi_sup = 0.0
    best_multi_con = 0.0
    best_multi_nei = 0.0
    best_multi_sup_meta = None
    best_multi_con_meta = None

    for meta, out in zip(doc_meta, multi_outs):
        p_s = out["probabilities"][LABEL_SUPPORTS]
        p_c = out["probabilities"][LABEL_CONTRADICTS]
        p_n = out["probabilities"][LABEL_NEI]

        if p_s > best_multi_sup:
            best_multi_sup = p_s
            best_multi_sup_meta = meta
        if p_c > best_multi_con:
            best_multi_con = p_c
            best_multi_con_meta = meta
        if p_n > best_multi_nei:
            best_multi_nei = p_n

    # Decision Consensus:
    # A single sentence with marginal confidence (< 0.65) that is contradicted by high multi-sentence NEI (> 0.70)
    # is flagged as out-of-context distractor.
    single_sup_val = best_sup["probabilities"][LABEL_SUPPORTS] if best_sup else 0.0
    single_con_val = best_con["probabilities"][LABEL_CONTRADICTS] if best_con else 0.0

    if single_con_val >= contradict_thresh and single_con_val > single_sup_val:
        if single_con_val < 0.65 and best_multi_nei > 0.70:
            return LABEL_NEI, best_multi_nei, evidence_details, []
        return LABEL_CONTRADICTS, single_con_val, evidence_details, [(best_con["doc_id"], best_con["sentence_idx"])]

    if single_sup_val >= entail_thresh and single_sup_val >= single_con_val:
        if single_sup_val < 0.65 and best_multi_nei > 0.70:
            return LABEL_NEI, best_multi_nei, evidence_details, []
        return LABEL_SUPPORTS, single_sup_val, evidence_details, [(best_sup["doc_id"], best_sup["sentence_idx"])]

    # If single sentences were inconclusive, check if multi-sentence context established confirmation
    if best_multi_sup >= 0.55 and best_multi_sup > best_multi_con:
        did, sidxs = best_multi_sup_meta
        return LABEL_SUPPORTS, best_multi_sup, evidence_details, [(did, sidxs[0])]
    if best_multi_con >= 0.55 and best_multi_con > best_multi_sup:
        did, sidxs = best_multi_con_meta
        return LABEL_CONTRADICTS, best_multi_con, evidence_details, [(did, sidxs[0])]

    return LABEL_NEI, 0.0, evidence_details, []

def compute_retrieval_and_sentence_metrics(
    eval_claims: List[Dict[str, Any]],
    hybrid_retriever: OptimizedHybridRetriever,
    selector_fn,
) -> Dict[str, Any]:
    doc_r5_hits = 0
    doc_r10_hits = 0
    doc_r20_hits = 0
    sent_hits = 0
    reciprocal_ranks_10 = []
    reciprocal_ranks_20 = []
    ndcg_10_scores = []
    answerable_count = 0

    for item in eval_claims:
        gold_doc_ids: Set[int] = item.get("gold_doc_ids", set())
        gold_sentences: Set[Tuple[int, int]] = item.get("gold_sentences", set())

        if not gold_doc_ids:
            continue
        answerable_count += 1

        claim_text = item["claim"]
        docs_20 = hybrid_retriever.retrieve_documents(claim_text, top_k=20)
        retrieved_ids = [d["doc_id"] for d in docs_20]

        if any(did in gold_doc_ids for did in retrieved_ids[:5]):
            doc_r5_hits += 1
        if any(did in gold_doc_ids for did in retrieved_ids[:10]):
            doc_r10_hits += 1
        if any(did in gold_doc_ids for did in retrieved_ids[:20]):
            doc_r20_hits += 1

        rr10 = 0.0
        for rank, did in enumerate(retrieved_ids[:10], start=1):
            if did in gold_doc_ids:
                rr10 = 1.0 / rank
                break
        reciprocal_ranks_10.append(rr10)

        rr20 = 0.0
        for rank, did in enumerate(retrieved_ids[:20], start=1):
            if did in gold_doc_ids:
                rr20 = 1.0 / rank
                break
        reciprocal_ranks_20.append(rr20)

        dcg10 = sum(
            1.0 / math.log2(rank + 1)
            for rank, did in enumerate(retrieved_ids[:10], start=1)
            if did in gold_doc_ids
        )
        ideal_count = min(len(gold_doc_ids), 10)
        idcg10 = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_count + 1)) if ideal_count > 0 else 1.0
        ndcg_10_scores.append(dcg10 / idcg10 if idcg10 > 0 else 0.0)

        # Sentence selection evaluation
        docs_10 = docs_20[:TOP_K_DOCS]
        sentences_top = selector_fn(claim_text, docs_10, top_k=TOP_K_SENTENCES)
        retrieved_sents = {(s["doc_id"], s["sentence_idx"]) for s in sentences_top}
        if gold_sentences and (retrieved_sents & gold_sentences):
            sent_hits += 1

    if answerable_count == 0:
        return {}

    return {
        "answerable_count": answerable_count,
        "doc_recall@5": round(doc_r5_hits / answerable_count, 4),
        "doc_recall@10": round(doc_r10_hits / answerable_count, 4),
        "doc_recall@20": round(doc_r20_hits / answerable_count, 4),
        "sentence_evidence_recall": round(sent_hits / answerable_count, 4),
        "doc_mrr@10": round(sum(reciprocal_ranks_10) / answerable_count, 4),
        "doc_mrr@20": round(sum(reciprocal_ranks_20) / answerable_count, 4),
        "doc_ndcg@10": round(sum(ndcg_10_scores) / answerable_count, 4),
    }

def evaluate_approach(
    name: str,
    selector_fn,
    is_multi_sentence: bool,
    corpus: Dict[int, Dict[str, Any]],
    hybrid_retriever: OptimizedHybridRetriever,
    nli_model: NLIEntailmentModel,
    test_claims: List[Dict[str, Any]],
    val_claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating Approach: {name}")
    print(f"=======================================================")

    # 1. Held-out Test Evaluation (N=24)
    test_preds = []
    latencies_test = []
    y_true_test = []
    y_pred_test = []

    for item in test_claims:
        cid = item["id"]
        claim_text = item["claim"]
        gold_label = item["gold_label"]
        y_true_test.append(gold_label)

        t_start = time.time()
        cand_docs = hybrid_retriever.retrieve_documents(claim_text, top_k=TOP_K_DOCS)
        sents = selector_fn(claim_text, cand_docs, top_k=TOP_K_SENTENCES)

        if is_multi_sentence:
            pred_label, conf, details, pred_sents = run_nli_multi_sentence_inference(
                nli_model, corpus, sents, claim_text
            )
        else:
            pred_label, conf, details, pred_sents = run_nli_standard_inference(
                nli_model, sents, claim_text
            )

        dt_ms = (time.time() - t_start) * 1000.0
        latencies_test.append(dt_ms)
        y_pred_test.append(pred_label)

        test_preds.append({
            "claim_id": cid,
            "claim": claim_text,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "confidence": round(float(conf), 4),
            "is_correct": pred_label == gold_label,
            "predicted_evidence_sentences": pred_sents,
            "latency_ms": round(dt_ms, 2)
        })

    test_metrics = compute_metrics(y_true_test, y_pred_test)
    test_retrieval = compute_retrieval_and_sentence_metrics(test_claims, hybrid_retriever, selector_fn)
    mean_test_latency = round(float(np.mean(latencies_test)), 2)
    p95_test_latency = round(float(np.percentile(latencies_test, 95)), 2)

    # 2. Validation Evaluation (N=161)
    val_preds = []
    latencies_val = []
    y_true_val = []
    y_pred_val = []

    for item in val_claims:
        cid = item["id"]
        claim_text = item["claim"]
        gold_label = item["gold_label"]
        y_true_val.append(gold_label)

        t_start = time.time()
        cand_docs = hybrid_retriever.retrieve_documents(claim_text, top_k=TOP_K_DOCS)
        sents = selector_fn(claim_text, cand_docs, top_k=TOP_K_SENTENCES)

        if is_multi_sentence:
            pred_label, conf, details, pred_sents = run_nli_multi_sentence_inference(
                nli_model, corpus, sents, claim_text
            )
        else:
            pred_label, conf, details, pred_sents = run_nli_standard_inference(
                nli_model, sents, claim_text
            )

        dt_ms = (time.time() - t_start) * 1000.0
        latencies_val.append(dt_ms)
        y_pred_val.append(pred_label)

        val_preds.append({
            "claim_id": cid,
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "is_correct": pred_label == gold_label,
            "latency_ms": round(dt_ms, 2)
        })

    val_metrics = compute_metrics(y_true_val, y_pred_val)
    val_retrieval = compute_retrieval_and_sentence_metrics(val_claims, hybrid_retriever, selector_fn)
    mean_val_latency = round(float(np.mean(latencies_val)), 2)
    p95_val_latency = round(float(np.percentile(latencies_val, 95)), 2)

    print(f"  Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test Sent Recall: {test_retrieval.get('sentence_evidence_recall', 0.0)*100:.2f}%")
    print(f"  Test Latency:  Mean={mean_test_latency} ms | P95={p95_test_latency} ms")
    print(f"  Val Accuracy:  {val_metrics['accuracy']*100:.2f}% | Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Val Sent Recall: {val_retrieval.get('sentence_evidence_recall', 0.0)*100:.2f}%")
    print(f"  Val Latency:   Mean={mean_val_latency} ms | P95={p95_val_latency} ms")

    claim_24_result = next((p for p in test_preds if p["claim_id"] == 24), {})

    return {
        "approach_name": name,
        "mean_test_latency_ms": mean_test_latency,
        "p95_test_latency_ms": p95_test_latency,
        "mean_val_latency_ms": mean_val_latency,
        "p95_val_latency_ms": p95_val_latency,
        "test_metrics_n24": {
            "overall_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "retrieval": test_retrieval,
        },
        "val_metrics_n161": {
            "overall_accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "per_class": val_metrics["per_class"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "retrieval": val_retrieval,
        },
        "claim_24_diagnostic": claim_24_result,
        "test_predictions": test_preds,
    }

def main():
    print("=" * 70)
    print("ResearchMap Phase 6A: Evidence Selection & Aggregation Benchmark")
    print("=" * 70)

    # 1. Load corpus
    t0 = time.time()
    corpus = load_corpus(EXPANDED_CORPUS_PATH)
    print(f"Loaded {len(corpus)} documents in {time.time()-t0:.2f}s.")

    # 2. Load NLI model
    model_path = Config.MODELS_DIR / "finetuned_nli"
    nli_model = NLIEntailmentModel(model_name=str(model_path), device="cpu")

    # 3. Load datasets
    with open(TEST_CLAIMS_PATH, "r", encoding="utf-8") as f:
        raw_test = [json.loads(line) for line in f if line.strip()]

    test_claims = []
    for c in raw_test:
        gold_dids = set()
        if c.get("evidence_doc_id") is not None:
            gold_dids.add(int(c["evidence_doc_id"]))
        gold_sents = set()
        did = c.get("evidence_doc_id")
        if did is not None:
            for sidx in c.get("gold_sentence_indices", []):
                gold_sents.add((int(did), int(sidx)))

        test_claims.append({
            "id": c["id"],
            "claim": c["claim"],
            "gold_label": c["gold_label"],
            "evidence_doc_id": c.get("evidence_doc_id"),
            "gold_doc_ids": gold_dids,
            "gold_sentences": gold_sents,
            "answerable_from_corpus": c.get("answerable_from_corpus", False)
        })

    raw_val = load_claims(VAL_CLAIMS_PATH)
    val_claims = []
    for c in raw_val:
        val_claims.append({
            "id": c["id"],
            "claim": c["claim"],
            "gold_label": c["gold_label"],
            "gold_doc_ids": c.get("gold_doc_evidence", set()),
            "gold_sentences": c.get("gold_sentence_evidence", set())
        })

    print(f"Loaded {len(test_claims)} test claims and {len(val_claims)} validation claims.")

    # 4. Initialize retrievers and selector harness
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, EMB_PATH, IDS_PATH, device="cpu")
    hybrid = OptimizedHybridRetriever(bm25, dense, rrf_k=60, candidate_pool=50, sentence_doc_gating=10)
    selectors = EvidenceSelectors(corpus, bm25, hybrid)

    # 5. Evaluate all 5 approaches
    # Approach A: Existing Sentence Selection
    res_a = evaluate_approach(
        "A. Existing Sentence Selection (Baseline)",
        selectors.select_approach_a, False,
        corpus, hybrid, nli_model, test_claims, val_claims
    )

    # Approach B: BM25-based Sentence Selection
    res_b = evaluate_approach(
        "B. BM25-based Sentence Selection",
        selectors.select_approach_b, False,
        corpus, hybrid, nli_model, test_claims, val_claims
    )

    # Approach C: Hybrid Document Score + Sentence Relevance Selection
    res_c = evaluate_approach(
        "C. Hybrid Document Score + Sentence Relevance Selection",
        selectors.select_approach_c, False,
        corpus, hybrid, nli_model, test_claims, val_claims
    )

    # Approach D: Evidence Diversity Selection
    res_d = evaluate_approach(
        "D. Evidence Diversity Selection (MMR)",
        selectors.select_approach_d, False,
        corpus, hybrid, nli_model, test_claims, val_claims
    )

    # Approach E: Multi-Sentence Evidence Aggregation
    res_e = evaluate_approach(
        "E. Multi-Sentence Evidence Aggregation",
        selectors.select_approach_e, True,
        corpus, hybrid, nli_model, test_claims, val_claims
    )

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_document_count": len(corpus),
        "nli_model": str(model_path),
        "approaches": {
            "A_Existing": res_a,
            "B_BM25": res_b,
            "C_Hybrid": res_c,
            "D_Diversity": res_d,
            "E_MultiSentence": res_e,
        }
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f_out:
        json.dump(results_data, f_out, indent=2)
    print(f"\nSaved Phase 6A evaluation results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()

