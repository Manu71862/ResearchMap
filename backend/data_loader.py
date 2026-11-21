import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

LABEL_SUPPORTS = "SUPPORTS"
LABEL_CONTRADICTS = "CONTRADICTS"
LABEL_NEI = "NOT ENOUGH INFORMATION"
CANONICAL_LABELS = [LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI]

def normalize_label(label: str) -> str:
    """Normalize dataset labels to canonical 3-class system."""
    clean = label.strip().upper()
    if clean in ("SUPPORT", "SUPPORTS", "ENTAILMENT"):
        return LABEL_SUPPORTS
    if clean in ("CONTRADICT", "CONTRADICTS", "CONTRADICTION"):
        return LABEL_CONTRADICTS
    if clean in ("NEI", "NOT ENOUGH INFO", "NOT ENOUGH INFORMATION", "NEUTRAL"):
        return LABEL_NEI
    return LABEL_NEI

def load_corpus(corpus_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load corpus.jsonl containing research paper abstracts.
    Returns:
        Dict mapping doc_id -> {
            "doc_id": int,
            "title": str,
            "abstract": List[str],  # List of sentence strings
            "structured": bool
        }
    """
    corpus = {}
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = int(data["doc_id"])
            corpus[doc_id] = {
                "doc_id": doc_id,
                "title": data.get("title", ""),
                "abstract": data.get("abstract", []),
                "structured": data.get("structured", False),
            }
    return corpus

def derive_gold_claim_label(evidence_dict: Dict[str, Any]) -> str:
    """
    Derive the claim-level label from SciFact evidence annotations:
    - If evidence is empty -> NOT ENOUGH INFORMATION
    - If evidence contains SUPPORT -> SUPPORTS
    - If evidence contains CONTRADICT -> CONTRADICTS
    """
    if not evidence_dict:
        return LABEL_NEI

    has_support = False
    has_contradict = False

    for doc_id_str, ev_list in evidence_dict.items():
        for ev in ev_list:
            lbl = ev.get("label", "").upper()
            if "SUPPORT" in lbl:
                has_support = True
            elif "CONTRADICT" in lbl:
                has_contradict = True

    if has_support and not has_contradict:
        return LABEL_SUPPORTS
    elif has_contradict and not has_support:
        return LABEL_CONTRADICTS
    elif has_support and has_contradict:
        # Edge case: both present
        return LABEL_SUPPORTS
    return LABEL_NEI

def extract_gold_evidence_sentences(evidence_dict: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """
    Extract set of (doc_id, sentence_index) tuples from gold evidence annotations.
    """
    gold_sentences = set()
    for doc_id_str, ev_list in evidence_dict.items():
        doc_id = int(doc_id_str)
        for ev in ev_list:
            for s_idx in ev.get("sentences", []):
                gold_sentences.add((doc_id, int(s_idx)))
    return gold_sentences

def extract_gold_evidence_docs(evidence_dict: Dict[str, Any]) -> Set[int]:
    """Extract set of doc_ids that contain gold evidence."""
    return {int(doc_id) for doc_id in evidence_dict.keys()}

def load_claims(claims_path: Path) -> List[Dict[str, Any]]:
    """
    Load claims from a jsonl file (e.g. claims_dev.jsonl, claims_train.jsonl).
    Enriches each record with parsed gold label and evidence sets.
    """
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            evidence = data.get("evidence", {})
            gold_label = derive_gold_claim_label(evidence)
            gold_sent_evidence = extract_gold_evidence_sentences(evidence)
            gold_doc_evidence = extract_gold_evidence_docs(evidence)

            claims.append({
                "id": data["id"],
                "claim": data["claim"],
                "evidence_raw": evidence,
                "cited_doc_ids": [int(d) for d in data.get("cited_doc_ids", [])],
                "gold_label": gold_label,
                "gold_doc_evidence": gold_doc_evidence,
                "gold_sentence_evidence": gold_sent_evidence,
            })
    return claims

def get_split_statistics(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate claim counts and class distribution."""
    total = len(claims)
    label_counts = {lbl: 0 for lbl in CANONICAL_LABELS}
    with_evidence_count = 0

    for c in claims:
        lbl = c["gold_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if len(c["gold_doc_evidence"]) > 0:
            with_evidence_count += 1

    return {
        "total_claims": total,
        "with_evidence": with_evidence_count,
        "without_evidence_nei": label_counts[LABEL_NEI],
        "label_distribution": label_counts,
        "label_percentages": {
            lbl: (cnt / total * 100.0) if total > 0 else 0.0
            for lbl, cnt in label_counts.items()
        }
    }
