"""
Dataset Builder for Scientific Claim Verification Fine-Tuning.

Constructs labelled (evidence_text, claim_text, label_idx) training pairs
from SciFact annotations with strict semantic integrity:
- SUPPORT evidence sets -> label: SUPPORTS (1)
- CONTRADICT evidence sets -> label: CONTRADICTS (0)
- Distractor sentences & ungrounded claims -> label: NOT ENOUGH INFORMATION (2)
- NEVER treats unrelated or neutral text as contradictory.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
import random

from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)

LABEL_TO_IDX = {
    LABEL_CONTRADICTS: 0,
    LABEL_SUPPORTS: 1,
    LABEL_NEI: 2,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

def build_claim_evidence_pairs(
    claims: List[Dict[str, Any]],
    corpus: Dict[int, Dict[str, Any]],
    include_distractors: bool = True,
    max_distractors_per_claim: int = 1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build labelled training pairs from claims and corpus.

    Returns:
        List of dicts:
        [
            {
                "claim_id": int,
                "claim": str,
                "evidence_text": str,
                "doc_id": Optional[int],
                "sentence_indices": List[int],
                "label": str,       # "SUPPORTS", "CONTRADICTS", "NOT ENOUGH INFORMATION"
                "label_idx": int,   # 0: CONTRADICTS, 1: SUPPORTS, 2: NEI
                "is_distractor": bool
            },
            ...
        ]
    """
    rng = random.Random(seed)
    pairs = []

    for claim_rec in claims:
        claim_id = claim_rec["id"]
        claim_text = claim_rec["claim"]
        evidence_dict = claim_rec.get("evidence_raw", {})
        cited_docs = claim_rec.get("cited_doc_ids", [])

        has_annotated_evidence = bool(evidence_dict)

        if has_annotated_evidence:
            for doc_id_str, ev_list in evidence_dict.items():
                doc_id = int(doc_id_str)
                doc_data = corpus.get(doc_id)
                if not doc_data:
                    continue
                abstract = doc_data.get("abstract", [])
                gold_sentence_set = set()

                for ev in ev_list:
                    s_indices = sorted(ev.get("sentences", []))
                    if not s_indices:
                        continue
                    gold_sentence_set.update(s_indices)

                    # Extract concatenated evidence sentences
                    evidence_sents = [abstract[i] for i in s_indices if 0 <= i < len(abstract)]
                    evidence_text = " ".join(evidence_sents).strip()
                    if not evidence_text:
                        continue

                    raw_label = ev.get("label", "").upper()
                    if "SUPPORT" in raw_label:
                        label = LABEL_SUPPORTS
                    elif "CONTRA" in raw_label:
                        label = LABEL_CONTRADICTS
                    else:
                        label = LABEL_NEI

                    pairs.append({
                        "claim_id": claim_id,
                        "claim": claim_text,
                        "evidence_text": evidence_text,
                        "doc_id": doc_id,
                        "sentence_indices": s_indices,
                        "label": label,
                        "label_idx": LABEL_TO_IDX[label],
                        "is_distractor": False,
                    })

                # Sample non-evidence distractor sentence(s) from the same paper as NEI
                if include_distractors and abstract:
                    non_gold_indices = [i for i in range(len(abstract)) if i not in gold_sentence_set]
                    if non_gold_indices:
                        rng.shuffle(non_gold_indices)
                        for d_idx in non_gold_indices[:max_distractors_per_claim]:
                            distractor_text = abstract[d_idx].strip()
                            if distractor_text:
                                pairs.append({
                                    "claim_id": claim_id,
                                    "claim": claim_text,
                                    "evidence_text": distractor_text,
                                    "doc_id": doc_id,
                                    "sentence_indices": [d_idx],
                                    "label": LABEL_NEI,
                                    "label_idx": LABEL_TO_IDX[LABEL_NEI],
                                    "is_distractor": True,
                                })

        else:
            # Claim with no evidence (Gold NEI)
            # Sample candidate sentences from cited documents as NEI
            for doc_id in cited_docs[:1]:
                doc_data = corpus.get(doc_id)
                if not doc_data:
                    continue
                abstract = doc_data.get("abstract", [])
                if abstract:
                    # Pick first or middle sentence from cited paper
                    d_idx = 0
                    distractor_text = abstract[d_idx].strip()
                    if distractor_text:
                        pairs.append({
                            "claim_id": claim_id,
                            "claim": claim_text,
                            "evidence_text": distractor_text,
                            "doc_id": doc_id,
                            "sentence_indices": [d_idx],
                            "label": LABEL_NEI,
                            "label_idx": LABEL_TO_IDX[LABEL_NEI],
                            "is_distractor": True,
                        })

    # Deterministic shuffle
    rng.shuffle(pairs)
    return pairs
