import pytest
import json
from pathlib import Path

from backend.data_loader import (
    normalize_label,
    derive_gold_claim_label,
    extract_gold_evidence_sentences,
    extract_gold_evidence_docs,
    load_corpus,
    load_claims,
    get_split_statistics,
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
)

def test_normalize_label():
    assert normalize_label("SUPPORT") == LABEL_SUPPORTS
    assert normalize_label("supports") == LABEL_SUPPORTS
    assert normalize_label("entailment") == LABEL_SUPPORTS
    assert normalize_label("CONTRADICT") == LABEL_CONTRADICTS
    assert normalize_label("contradiction") == LABEL_CONTRADICTS
    assert normalize_label("NEI") == LABEL_NEI
    assert normalize_label("not enough info") == LABEL_NEI
    assert normalize_label("random_string") == LABEL_NEI

def test_derive_gold_claim_label():
    # Empty evidence -> NEI
    assert derive_gold_claim_label({}) == LABEL_NEI

    # Support evidence
    sup_ev = {"101": [{"sentences": [1], "label": "SUPPORT"}]}
    assert derive_gold_claim_label(sup_ev) == LABEL_SUPPORTS

    # Contradict evidence
    con_ev = {"202": [{"sentences": [0, 2], "label": "CONTRADICT"}]}
    assert derive_gold_claim_label(con_ev) == LABEL_CONTRADICTS

    # Multi-doc support
    multi_ev = {
        "101": [{"sentences": [1], "label": "SUPPORT"}],
        "102": [{"sentences": [3], "label": "SUPPORT"}]
    }
    assert derive_gold_claim_label(multi_ev) == LABEL_SUPPORTS

def test_extract_gold_evidence():
    ev = {
        "101": [
            {"sentences": [1, 2], "label": "SUPPORT"},
            {"sentences": [5], "label": "SUPPORT"}
        ],
        "202": [
            {"sentences": [0], "label": "CONTRADICT"}
        ]
    }
    sentences = extract_gold_evidence_sentences(ev)
    assert (101, 1) in sentences
    assert (101, 2) in sentences
    assert (101, 5) in sentences
    assert (202, 0) in sentences
    assert len(sentences) == 4

    docs = extract_gold_evidence_docs(ev)
    assert docs == {101, 202}

def test_load_corpus_and_claims_dummy(tmp_path: Path):
    # Create temporary corpus
    corpus_file = tmp_path / "dummy_corpus.jsonl"
    corpus_data = [
        {"doc_id": 1, "title": "Paper One", "abstract": ["Sentence A.", "Sentence B."]},
        {"doc_id": 2, "title": "Paper Two", "abstract": ["Sentence C."]}
    ]
    with open(corpus_file, "w", encoding="utf-8") as f:
        for r in corpus_data:
            f.write(json.dumps(r) + "\n")

    corpus = load_corpus(corpus_file)
    assert len(corpus) == 2
    assert corpus[1]["title"] == "Paper One"
    assert len(corpus[1]["abstract"]) == 2

    # Create temporary claims
    claims_file = tmp_path / "dummy_claims.jsonl"
    claims_data = [
        {
            "id": 10,
            "claim": "Claim one is true.",
            "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]},
            "cited_doc_ids": [1]
        },
        {
            "id": 11,
            "claim": "Claim two has no evidence.",
            "evidence": {},
            "cited_doc_ids": []
        }
    ]
    with open(claims_file, "w", encoding="utf-8") as f:
        for r in claims_data:
            f.write(json.dumps(r) + "\n")

    claims = load_claims(claims_file)
    assert len(claims) == 2
    assert claims[0]["gold_label"] == LABEL_SUPPORTS
    assert claims[0]["gold_doc_evidence"] == {1}
    assert (1, 0) in claims[0]["gold_sentence_evidence"]
    assert claims[1]["gold_label"] == LABEL_NEI
    assert claims[1]["gold_doc_evidence"] == set()

    # Split statistics
    stats = get_split_statistics(claims)
    assert stats["total_claims"] == 2
    assert stats["with_evidence"] == 1
    assert stats["without_evidence_nei"] == 1
    assert stats["label_distribution"][LABEL_SUPPORTS] == 1
    assert stats["label_distribution"][LABEL_NEI] == 1

def test_missing_files_raise_error():
    with pytest.raises(FileNotFoundError):
        load_corpus(Path("non_existent_corpus.jsonl"))

    with pytest.raises(FileNotFoundError):
        load_claims(Path("non_existent_claims.jsonl"))
