"""
Phase 4 Step 1: Detailed Error Audit & Confidence Distribution Analysis
Compares models/finetuned_nli vs models/experimental_augmented_nli
across Validation (N=161), Diagnostic (N=24), and Challenge (N=60).
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy.special import softmax
from sklearn.metrics import brier_score_loss

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.data_loader import (
    load_corpus,
    load_claims,
    derive_gold_claim_label,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.config import Config
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel


def multiclass_brier_score(golds_idx: List[int], prob_matrix: np.ndarray) -> float:
    """Calculate multi-class Brier score: mean squared difference between predicted probability and 1-hot target."""
    n_samples = len(golds_idx)
    num_classes = prob_matrix.shape[1]
    one_hot = np.zeros((n_samples, num_classes))
    for i, idx in enumerate(golds_idx):
        one_hot[i, idx] = 1.0
    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def run_detailed_audit():
    corpus = load_corpus(Config.RAW_DATA_DIR / "corpus.jsonl")
    retriever = BM25Retriever(corpus)

    val_claims = load_claims(Config.PROCESSED_DATA_DIR / "val_internal.jsonl")
    test_claims = [json.loads(l) for l in open(Config.BASE_DIR / "experiments" / "new_claims.jsonl", encoding="utf-8") if l.strip()]
    challenge_pairs = [json.loads(l) for l in open(Config.BASE_DIR / "data" / "experimental" / "challenge_set.jsonl", encoding="utf-8") if l.strip()]

    model_base = NLIEntailmentModel(model_name=str(Config.MODELS_DIR / "finetuned_nli"), device="cpu")
    model_aug = NLIEntailmentModel(model_name=str(Config.MODELS_DIR / "experimental_augmented_nli"), device="cpu")

    label2idx = {LABEL_CONTRADICTS: 0, LABEL_SUPPORTS: 1, LABEL_NEI: 2}

    print("Loaded datasets and models. Auditing Diagnostic Claims...")

    # Diagnostic Audit
    diag_audit = []
    diag_base_probs = []
    diag_aug_probs = []
    diag_golds_idx = []

    for c in test_claims:
        cid = c["id"]
        claim_text = c["claim"]
        gold = c["gold_label"]
        gold_idx = label2idx[gold]
        diag_golds_idx.append(gold_idx)

        candidate_docs = retriever.retrieve_documents(claim_text, top_k=Config.BM25_TOP_K_DOCS)
        candidate_sentences = retriever.retrieve_sentences(
            claim_text, candidate_docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES
        )

        pairs = [(s["text"], claim_text) for s in candidate_sentences]
        out_base = model_base.predict_batch(pairs)
        out_aug = model_aug.predict_batch(pairs)

        def eval_pipeline(candidates, outputs):
            best_sup = None
            best_con = None
            for s, out in zip(candidates, outputs):
                p_sup = out["probabilities"].get(LABEL_SUPPORTS, 0.0)
                p_con = out["probabilities"].get(LABEL_CONTRADICTS, 0.0)
                if p_sup >= Config.ENTAIL_THRESHOLD:
                    if best_sup is None or p_sup > best_sup["p"]:
                        best_sup = {"p": p_sup, "sent": s["text"], "doc_id": s["doc_id"], "probs": out["probabilities"]}
                if p_con >= Config.CONTRADICT_THRESHOLD:
                    if best_con is None or p_con > best_con["p"]:
                        best_con = {"p": p_con, "sent": s["text"], "doc_id": s["doc_id"], "probs": out["probabilities"]}
            if best_sup and best_con:
                pred = LABEL_SUPPORTS if best_sup["p"] >= best_con["p"] else LABEL_CONTRADICTS
                best_ev = best_sup if best_sup["p"] >= best_con["p"] else best_con
            elif best_sup:
                pred = LABEL_SUPPORTS
                best_ev = best_sup
            elif best_con:
                pred = LABEL_CONTRADICTS
                best_ev = best_con
            else:
                pred = LABEL_NEI
                # For NEI, find sentence with highest NEI prob
                best_ev = max(
                    [{"p": o["probabilities"].get(LABEL_NEI, 0.0), "sent": s["text"], "doc_id": s["doc_id"], "probs": o["probabilities"]} for s, o in zip(candidates, outputs)],
                    key=lambda x: x["p"],
                    default={"p": 0.0, "probs": {LABEL_CONTRADICTS: 0.33, LABEL_SUPPORTS: 0.33, LABEL_NEI: 0.34}}
                )
            # Create a 3-class probability vector for the claim
            probs = [
                best_ev["probs"].get(LABEL_CONTRADICTS, 0.0),
                best_ev["probs"].get(LABEL_SUPPORTS, 0.0),
                best_ev["probs"].get(LABEL_NEI, 0.0),
            ]
            return pred, best_ev, probs

        pred_base, ev_base, p_base = eval_pipeline(candidate_sentences, out_base)
        pred_aug, ev_aug, p_aug = eval_pipeline(candidate_sentences, out_aug)

        diag_base_probs.append(p_base)
        diag_aug_probs.append(p_aug)

        is_flipped = (pred_base != pred_aug)
        is_regression = (pred_base == gold and pred_aug != gold)

        diag_audit.append({
            "claim_id": cid,
            "claim": claim_text,
            "gold_label": gold,
            "baseline": {
                "prediction": pred_base,
                "confidence": ev_base["p"],
                "probabilities": ev_base["probs"],
                "evidence_doc": ev_base["doc_id"],
                "evidence_text": ev_base["sent"],
            },
            "augmented": {
                "prediction": pred_aug,
                "confidence": ev_aug["p"],
                "probabilities": ev_aug["probs"],
                "evidence_doc": ev_aug["doc_id"],
                "evidence_text": ev_aug["sent"],
            },
            "is_flipped": is_flipped,
            "is_regression": is_regression,
        })

    brier_diag_base = multiclass_brier_score(diag_golds_idx, np.array(diag_base_probs))
    brier_diag_aug = multiclass_brier_score(diag_golds_idx, np.array(diag_aug_probs))

    print(f"\n--- Diagnostic Audit Summary ---")
    print(f"Total Diagnostic Claims: {len(test_claims)}")
    regressions = [a for a in diag_audit if a["is_regression"]]
    print(f"Total Baseline Correct -> Augmented Incorrect (Regressions): {len(regressions)}")
    for r in regressions:
        print(f"\n[Claim {r['claim_id']}] Gold: {r['gold_label']}")
        print(f"  Claim: {r['claim']}")
        print(f"  Base -> Pred: {r['baseline']['prediction']} (P(CON)={r['baseline']['probabilities']['CONTRADICTS']:.3f}, P(SUP)={r['baseline']['probabilities']['SUPPORTS']:.3f}, P(NEI)={r['baseline']['probabilities']['NOT ENOUGH INFORMATION']:.3f})")
        print(f"  Aug  -> Pred: {r['augmented']['prediction']} (P(CON)={r['augmented']['probabilities']['CONTRADICTS']:.3f}, P(SUP)={r['augmented']['probabilities']['SUPPORTS']:.3f}, P(NEI)={r['augmented']['probabilities']['NOT ENOUGH INFORMATION']:.3f})")
        print(f"  Evidence: Doc {r['baseline']['evidence_doc']}: {r['baseline']['evidence_text'][:90]}...")

    print(f"\nDiagnostic Brier Score (Lower is better):")
    print(f"  Baseline:  {brier_diag_base:.4f}")
    print(f"  Augmented: {brier_diag_aug:.4f}")

    # Now evaluate Challenge Set probabilities and Brier score
    ch_text_pairs = [(p["evidence_text"], p["claim"]) for p in challenge_pairs]
    ch_golds_idx = [p["gold_label_idx"] for p in challenge_pairs]

    out_ch_base = model_base.predict_batch(ch_text_pairs)
    out_ch_aug = model_aug.predict_batch(ch_text_pairs)

    p_ch_base = np.array([[o["probabilities"][LABEL_CONTRADICTS], o["probabilities"][LABEL_SUPPORTS], o["probabilities"][LABEL_NEI]] for o in out_ch_base])
    p_ch_aug = np.array([[o["probabilities"][LABEL_CONTRADICTS], o["probabilities"][LABEL_SUPPORTS], o["probabilities"][LABEL_NEI]] for o in out_ch_aug])

    brier_ch_base = multiclass_brier_score(ch_golds_idx, p_ch_base)
    brier_ch_aug = multiclass_brier_score(ch_golds_idx, p_ch_aug)

    print(f"\nChallenge Set Brier Score (Lower is better):")
    print(f"  Baseline:  {brier_ch_base:.4f}")
    print(f"  Augmented: {brier_ch_aug:.4f}")

    # Now evaluate Validation Set probabilities and Brier score
    val_base_probs = []
    val_aug_probs = []
    val_golds_idx = []
    val_regressions = []
    val_corrections = []

    for c in val_claims:
        claim_text = c["claim"]
        gold = derive_gold_claim_label(c.get("evidence", {}))
        gold_idx = label2idx[gold]
        val_golds_idx.append(gold_idx)

        candidate_docs = retriever.retrieve_documents(claim_text, top_k=Config.BM25_TOP_K_DOCS)
        candidate_sentences = retriever.retrieve_sentences(
            claim_text, candidate_docs, top_k_sentences=Config.BM25_TOP_K_SENTENCES
        )

        pairs = [(s["text"], claim_text) for s in candidate_sentences]
        out_base = model_base.predict_batch(pairs)
        out_aug = model_aug.predict_batch(pairs)

        pred_base, ev_base, p_base = eval_pipeline(candidate_sentences, out_base)
        pred_aug, ev_aug, p_aug = eval_pipeline(candidate_sentences, out_aug)

        val_base_probs.append(p_base)
        val_aug_probs.append(p_aug)

        if pred_base == gold and pred_aug != gold:
            val_regressions.append(c["id"])
        elif pred_base != gold and pred_aug == gold:
            val_corrections.append(c["id"])

    brier_val_base = multiclass_brier_score(val_golds_idx, np.array(val_base_probs))
    brier_val_aug = multiclass_brier_score(val_golds_idx, np.array(val_aug_probs))

    print(f"\nValidation Set Brier Score (Lower is better):")
    print(f"  Baseline:  {brier_val_base:.4f} (Corrections: {len(val_corrections)}, Regressions: {len(val_regressions)}, Net: +{len(val_corrections) - len(val_regressions)})")
    print(f"  Augmented: {brier_val_aug:.4f}")

    return {
        "diagnostic_regressions": regressions,
        "brier_scores": {
            "validation": {"baseline": brier_val_base, "augmented": brier_val_aug},
            "diagnostic": {"baseline": brier_diag_base, "augmented": brier_diag_aug},
            "challenge": {"baseline": brier_ch_base, "augmented": brier_ch_aug},
        },
        "val_corrections_count": len(val_corrections),
        "val_regressions_count": len(val_regressions),
    }

if __name__ == "__main__":
    run_detailed_audit()
