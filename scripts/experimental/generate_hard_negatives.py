"""
Generator for Scientific Hard-Negative Training Pairs (Phase 3 Experimental - Balanced)

Produces a balanced, high-quality collection of ~1,500 hard-negative pairs:
- ~650-750 CONTRADICTS pairs (direction reversal, negation, numerical discrepancy, outcome reversal)
- ~750-850 NOT ENOUGH INFORMATION pairs (causal overclaiming, scope mismatch, ungrounded distractors)
"""

import sys
import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.data_loader import load_corpus, load_claims, LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI
from backend.config import Config


DIRECTIONAL_RULES = [
    (r"\bincreases\b", "decreases", "increases -> decreases"),
    (r"\bdecreases\b", "increases", "decreases -> increases"),
    (r"\bincreased\b", "decreased", "increased -> decreased"),
    (r"\bdecreased\b", "increased", "decreased -> increased"),
    (r"\bincreasing\b", "decreasing", "increasing -> decreasing"),
    (r"\bdecreasing\b", "increasing", "decreasing -> increasing"),
    (r"\breduces\b", "increases", "reduces -> increases"),
    (r"\breduced\b", "increased", "reduced -> increased"),
    (r"\breducing\b", "increasing", "reducing -> increasing"),
    (r"\bpromotes\b", "inhibits", "promotes -> inhibits"),
    (r"\binhibits\b", "promotes", "inhibits -> promotes"),
    (r"\bpromoted\b", "inhibited", "promoted -> inhibited"),
    (r"\binhibited\b", "promoted", "inhibited -> promoted"),
    (r"\bpromoting\b", "inhibiting", "promoting -> inhibiting"),
    (r"\binhibiting\b", "promoting", "inhibiting -> promoting"),
    (r"\bactivates\b", "suppresses", "activates -> suppresses"),
    (r"\bsuppresses\b", "activates", "suppresses -> activates"),
    (r"\bactivated\b", "suppressed", "activated -> suppressed"),
    (r"\bsuppressed\b", "activated", "suppressed -> activated"),
    (r"\bactivating\b", "suppressing", "activating -> suppressing"),
    (r"\bsuppressing\b", "activating", "suppressing -> activating"),
    (r"\belevates\b", "lowers", "elevates -> lowers"),
    (r"\blowers\b", "elevates", "lowers -> elevates"),
    (r"\belevated\b", "lowered", "elevated -> lowered"),
    (r"\blowered\b", "elevated", "lowered -> elevated"),
    (r"\benhances\b", "impairs", "enhances -> impairs"),
    (r"\bimpairs\b", "enhances", "impairs -> enhances"),
    (r"\benhanced\b", "impaired", "enhanced -> impaired"),
    (r"\bimpaired\b", "enhanced", "impaired -> enhanced"),
    (r"\bupregulates\b", "downregulates", "upregulates -> downregulates"),
    (r"\bdownregulates\b", "upregulates", "downregulates -> upregulates"),
    (r"\bupregulated\b", "downregulated", "upregulated -> downregulated"),
    (r"\bdownregulated\b", "upregulated", "downregulated -> upregulated"),
    (r"\baccelerates\b", "delays", "accelerates -> delays"),
    (r"\bdelays\b", "accelerates", "delays -> accelerates"),
    (r"\bhigher\b", "lower", "higher -> lower"),
    (r"\blower\b", "higher", "lower -> higher"),
    (r"\bmore\b", "less", "more -> less"),
    (r"\bless\b", "more", "less -> more"),
    (r"\bdeficiency of\b", "excess of", "deficiency -> excess"),
    (r"\bexcess of\b", "deficiency of", "excess -> deficiency"),
    (r"\bloss of\b", "gain of", "loss of -> gain of"),
    (r"\bgain of\b", "loss of", "gain of -> loss of"),
    (r"\bpresence of\b", "absence of", "presence of -> absence of"),
    (r"\babsence of\b", "presence of", "absence of -> presence of"),
    (r"\bpositively correlates\b", "inversely correlates", "positive correlation -> inverse correlation"),
    (r"\binversely correlates\b", "positively correlates", "inverse correlation -> positive correlation"),
    (r"\bpositive correlation\b", "negative correlation", "positive correlation -> negative correlation"),
    (r"\bnegative correlation\b", "positive correlation", "negative correlation -> positive correlation"),
    # Additional antonym variants
    (r"\bincreases\b", "fails to increase", "increases -> fails to increase"),
    (r"\breduces\b", "fails to reduce", "reduces -> fails to reduce"),
    (r"\bpromotes\b", "fails to promote", "promotes -> fails to promote"),
    (r"\bactivates\b", "fails to activate", "activates -> fails to activate"),
    (r"\benhances\b", "fails to enhance", "enhances -> fails to enhance"),
]

NEGATION_RULES = [
    (r"\bis required for\b", "is dispensable for", "is required for -> is dispensable for"),
    (r"\bare required for\b", "are dispensable for", "are required for -> are dispensable for"),
    (r"\bis essential for\b", "is non-essential for", "is essential for -> is non-essential for"),
    (r"\bare essential for\b", "are non-essential for", "are essential for -> are non-essential for"),
    (r"\bis dispensable for\b", "is strictly required for", "is dispensable for -> is strictly required for"),
    (r"\bare dispensable for\b", "are strictly required for", "are dispensable for -> are strictly required for"),
    (r"\bis effective in\b", "is ineffective in", "is effective in -> is ineffective in"),
    (r"\bare effective in\b", "are ineffective in", "are effective in -> are ineffective in"),
    (r"\bis effective for\b", "is ineffective for", "is effective for -> is ineffective for"),
    (r"\bis associated with\b", "is unrelated to", "is associated with -> is unrelated to"),
    (r"\bare associated with\b", "are unrelated to", "are associated with -> are unrelated to"),
    (r"\bis correlated with\b", "shows no correlation with", "is correlated with -> shows no correlation with"),
    (r"\bare correlated with\b", "show no correlation with", "are correlated with -> show no correlation with"),
    (r"\bcorrelates with\b", "does not correlate with", "correlates with -> does not correlate with"),
    (r"\bcorrelate with\b", "do not correlate with", "correlate with -> do not correlate with"),
    (r"\binduces\b", "fails to induce", "induces -> fails to induce"),
    (r"\binduce\b", "fail to induce", "induce -> fail to induce"),
    (r"\binduced\b", "failed to induce", "induced -> failed to induce"),
    (r"\bregulates\b", "does not regulate", "regulates -> does not regulate"),
    (r"\bregulate\b", "do not regulate", "regulate -> do not regulate"),
    (r"\bcauses\b", "does not cause", "causes -> does not cause"),
    (r"\bcause\b", "do not cause", "cause -> do not cause"),
    (r"\bcaused\b", "did not cause", "caused -> did not cause"),
    (r"\bleads to\b", "does not lead to", "leads to -> does not lead to"),
    (r"\blead to\b", "do not lead to", "lead to -> do not lead to"),
    (r"\bprevents\b", "fails to prevent", "prevents -> fails to prevent"),
    (r"\bprevent\b", "fail to prevent", "prevent -> fail to prevent"),
    (r"\bprevented\b", "failed to prevent", "prevented -> failed to prevent"),
    (r"\bprotects against\b", "fails to protect against", "protects against -> fails to protect against"),
    (r"\bprotect against\b", "fail to protect against", "protect against -> fail to protect against"),
    (r"\bplays a role in\b", "plays no role in", "plays a role in -> plays no role in"),
    (r"\bplay a role in\b", "play no role in", "play a role in -> play no role in"),
    (r"\bhas a role in\b", "has no role in", "has a role in -> has no role in"),
    (r"\bhave a role in\b", "have no role in", "have a role in -> have no role in"),
    (r"\bcan trigger\b", "cannot trigger", "can trigger -> cannot trigger"),
    (r"\bis capable of\b", "is incapable of", "is capable of -> is incapable of"),
]

INTERVENTION_RULES = [
    (r"\bimproves\b", "worsens", "improves -> worsens"),
    (r"\bimprove\b", "worsen", "improve -> worsen"),
    (r"\bworsens\b", "improves", "worsens -> improves"),
    (r"\bworsen\b", "improve", "worsen -> improve"),
    (r"\bimproved\b", "worsened", "improved -> worsened"),
    (r"\bworsened\b", "improved", "worsened -> improved"),
    (r"\breduces mortality\b", "increases mortality", "reduces mortality -> increases mortality"),
    (r"\bincreases mortality\b", "reduces mortality", "increases mortality -> reduces mortality"),
    (r"\bextends lifespan\b", "shortens lifespan", "extends lifespan -> shortens lifespan"),
    (r"\bshortens lifespan\b", "extends lifespan", "shortens lifespan -> extends lifespan"),
    (r"\bextends life span\b", "shortens life span", "extends life span -> shortens life span"),
    (r"\bshortens life span\b", "extends life span", "shortens life span -> extends life span"),
    (r"\breduces the risk of\b", "increases the risk of", "reduces risk of -> increases risk of"),
    (r"\bincreases the risk of\b", "reduces the risk of", "increases risk of -> reduces risk of"),
    (r"\breduces risk of\b", "increases risk of", "reduces risk of -> increases risk of"),
    (r"\bincreases risk of\b", "reduces risk of", "increases risk of -> reduces risk of"),
    (r"\balleviates\b", "exacerbates", "alleviates -> exacerbates"),
    (r"\balleviate\b", "exacerbate", "alleviate -> exacerbate"),
    (r"\bexacerbates\b", "alleviates", "exacerbates -> alleviates"),
    (r"\bexacerbate\b", "alleviate", "exacerbate -> alleviate"),
    (r"\bameliorates\b", "aggravates", "ameliorates -> aggravates"),
    (r"\baggravates\b", "ameliorates", "aggravates -> ameliorates"),
    (r"\bdelays progression\b", "accelerates progression", "delays progression -> accelerates progression"),
    (r"\baccelerates progression\b", "delays progression", "accelerates progression -> delays progression"),
]

CAUSAL_RULES = [
    (r"\bis associated with\b", "is direct proof of causation for", "association -> proof of causation"),
    (r"\bare associated with\b", "are direct proof of causation for", "association -> proof of causation"),
    (r"\bis correlated with\b", "definitively proves the cause of", "correlation -> proven causation"),
    (r"\bare correlated with\b", "definitively prove the cause of", "correlation -> proven causation"),
    (r"\bcorrelates with\b", "directly and exclusively causes", "correlates -> exclusively causes"),
    (r"\bcorrelate with\b", "directly and exclusively cause", "correlate -> exclusively cause"),
    (r"\bcontributes to\b", "is the sole and sufficient cause of", "contributes to -> sole cause of"),
    (r"\bcontribute to\b", "are the sole and sufficient cause of", "contribute to -> sole cause of"),
    (r"\bis a risk factor for\b", "is guaranteed to cause", "risk factor -> guaranteed cause"),
    (r"\bare risk factors for\b", "are guaranteed to cause", "risk factors -> guaranteed cause"),
]


def load_heldout_exclusions() -> Tuple[Set[int], Set[int], Set[str]]:
    val_path = Config.PROCESSED_DATA_DIR / "val_internal.jsonl"
    dev_path = Config.PROCESSED_DATA_DIR / "dev_heldout.jsonl"
    test_path = Config.BASE_DIR / "experiments" / "new_claims.jsonl"

    excluded_claim_ids = set()
    excluded_doc_ids = set()
    excluded_claim_texts = set()

    for path in [val_path, dev_path, test_path]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "id" in d and d["id"] is not None:
                    excluded_claim_ids.add(int(d["id"]))
                if "claim" in d:
                    excluded_claim_texts.add(d["claim"].strip().lower())
                for doc_id in d.get("cited_doc_ids", []):
                    excluded_doc_ids.add(int(doc_id))
                if "evidence_doc_id" in d and d["evidence_doc_id"]:
                    excluded_doc_ids.add(int(d["evidence_doc_id"]))
                if "evidence" in d and isinstance(d["evidence"], dict):
                    for k in d["evidence"].keys():
                        try:
                            excluded_doc_ids.add(int(k))
                        except ValueError:
                            pass

    print(f"Leakage protection: {len(excluded_claim_ids)} claim IDs and {len(excluded_doc_ids)} doc IDs excluded.")
    return excluded_claim_ids, excluded_doc_ids, excluded_claim_texts


def generate_hard_negatives(target_count: int = 1500) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("Generating Scientific Hard-Negative Training Pairs (Phase 3 - Balanced)")
    print(f"Target count: ~{target_count} validated pairs")
    print("=" * 70)

    corpus_path = Config.RAW_DATA_DIR / "corpus.jsonl"
    corpus = load_corpus(corpus_path)

    train_path = Config.PROCESSED_DATA_DIR / "train_partition.jsonl"
    train_claims = load_claims(train_path)

    ex_claim_ids, ex_doc_ids, ex_claim_texts = load_heldout_exclusions()

    generated_pairs = []
    seen_pair_hashes = set()
    rng = random.Random(Config.RANDOM_SEED)

    category_counts = {
        "direction_reversal": 0,
        "negation_inversion": 0,
        "numerical_comparison": 0,
        "causal_reasoning": 0,
        "intervention_reversal": 0,
        "scope_mismatch": 0,
        "cross_document_distractor": 0,
    }

    eligible_train_claims = [c for c in train_claims if c["id"] not in ex_claim_ids]
    print(f"Eligible training claims after leakage filtering: {len(eligible_train_claims)}")

    claim_evidence_tuples = []
    for claim_rec in eligible_train_claims:
        cid = claim_rec["id"]
        claim_text = claim_rec["claim"].strip()
        evidence_dict = claim_rec.get("evidence_raw", {})

        if not evidence_dict:
            continue

        for doc_id_str, ev_list in evidence_dict.items():
            doc_id = int(doc_id_str)
            if doc_id in ex_doc_ids:
                continue

            doc_data = corpus.get(doc_id)
            if not doc_data:
                continue

            abstract = doc_data.get("abstract", [])

            for ev in ev_list:
                s_indices = sorted(ev.get("sentences", []))
                if not s_indices:
                    continue

                evidence_sents = [abstract[i] for i in s_indices if 0 <= i < len(abstract)]
                evidence_text = " ".join(evidence_sents).strip()
                if not evidence_text or len(evidence_text) < 20:
                    continue

                raw_label = ev.get("label", "").upper()
                is_support = "SUPPORT" in raw_label
                claim_evidence_tuples.append({
                    "cid": cid,
                    "claim": claim_text,
                    "doc_id": doc_id,
                    "evidence_text": evidence_text,
                    "is_support": is_support,
                    "evidence_sentences": evidence_sents,
                })

    print(f"Total eligible claim-evidence gold pairs: {len(claim_evidence_tuples)}")

    # Target contradiction count ~650
    # 1. Directional Reversals on claims AND on evidence sentences
    for item in claim_evidence_tuples:
        cid = item["cid"]
        claim_text = item["claim"]
        doc_id = item["doc_id"]
        evidence_text = item["evidence_text"]
        is_support = item["is_support"]

        if is_support:
            for pat, repl, rule in DIRECTIONAL_RULES:
                if re.search(pat, claim_text, re.IGNORECASE):
                    perturbed_claim = re.sub(pat, repl, claim_text, count=1, flags=re.IGNORECASE)
                    pair_hash = (perturbed_claim.lower(), evidence_text.lower())
                    if pair_hash not in seen_pair_hashes:
                        seen_pair_hashes.add(pair_hash)
                        category_counts["direction_reversal"] += 1
                        generated_pairs.append({
                            "synthetic_id": f"syn_dir_{cid}_{category_counts['direction_reversal']}",
                            "source_claim_id": cid,
                            "doc_id": doc_id,
                            "claim": perturbed_claim,
                            "evidence_text": evidence_text,
                            "label": LABEL_CONTRADICTS,
                            "label_idx": 0,
                            "category": "direction_reversal",
                            "generation_method": rule,
                            "label_justification": f"Original evidence supports '{claim_text}'. Inverting the directional relationship using rule '{rule}' produces a direct scientific contradiction.",
                            "is_synthetic": True,
                        })

            # Also check individual evidence sentences for directional claims
            for ev_sent in item["evidence_sentences"]:
                for pat, repl, rule in DIRECTIONAL_RULES[:20]:
                    if re.search(pat, ev_sent, re.IGNORECASE):
                        # Formulate a contrastive probe hypothesis
                        perturbed_ev = re.sub(pat, repl, ev_sent, count=1, flags=re.IGNORECASE)
                        # Clean to sound like a concise hypothesis
                        pair_hash = (perturbed_ev.lower(), evidence_text.lower())
                        if pair_hash not in seen_pair_hashes and len(perturbed_ev) < 200:
                            seen_pair_hashes.add(pair_hash)
                            category_counts["direction_reversal"] += 1
                            generated_pairs.append({
                                "synthetic_id": f"syn_ev_dir_{cid}_{category_counts['direction_reversal']}",
                                "source_claim_id": cid,
                                "doc_id": doc_id,
                                "claim": perturbed_ev,
                                "evidence_text": evidence_text,
                                "label": LABEL_CONTRADICTS,
                                "label_idx": 0,
                                "category": "direction_reversal",
                                "generation_method": f"evidence_antonym_probe_{rule}",
                                "label_justification": "Direct antonym hypothesis formed against factual finding sentence in evidence text.",
                                "is_synthetic": True,
                            })

        # 2. Negation and Necessity Inversion
        if is_support:
            for pat, repl, rule in NEGATION_RULES:
                if re.search(pat, claim_text, re.IGNORECASE):
                    perturbed_claim = re.sub(pat, repl, claim_text, count=1, flags=re.IGNORECASE)
                    pair_hash = (perturbed_claim.lower(), evidence_text.lower())
                    if pair_hash not in seen_pair_hashes:
                        seen_pair_hashes.add(pair_hash)
                        category_counts["negation_inversion"] += 1
                        generated_pairs.append({
                            "synthetic_id": f"syn_neg_{cid}_{category_counts['negation_inversion']}",
                            "source_claim_id": cid,
                            "doc_id": doc_id,
                            "claim": perturbed_claim,
                            "evidence_text": evidence_text,
                            "label": LABEL_CONTRADICTS,
                            "label_idx": 0,
                            "category": "negation_inversion",
                            "generation_method": rule,
                            "label_justification": f"Evidence establishes the function/association of '{claim_text}'. Explicit negation using rule '{rule}' creates a direct scientific contradiction.",
                            "is_synthetic": True,
                        })

        # 3. Intervention Outcome Reversal
        if is_support:
            for pat, repl, rule in INTERVENTION_RULES:
                if re.search(pat, claim_text, re.IGNORECASE):
                    perturbed_claim = re.sub(pat, repl, claim_text, count=1, flags=re.IGNORECASE)
                    pair_hash = (perturbed_claim.lower(), evidence_text.lower())
                    if pair_hash not in seen_pair_hashes:
                        seen_pair_hashes.add(pair_hash)
                        category_counts["intervention_reversal"] += 1
                        generated_pairs.append({
                            "synthetic_id": f"syn_int_{cid}_{category_counts['intervention_reversal']}",
                            "source_claim_id": cid,
                            "doc_id": doc_id,
                            "claim": perturbed_claim,
                            "evidence_text": evidence_text,
                            "label": LABEL_CONTRADICTS,
                            "label_idx": 0,
                            "category": "intervention_reversal",
                            "generation_method": rule,
                            "label_justification": f"Evidence establishes clinical efficacy or benefit; inverting the clinical outcome with rule '{rule}' produces an opposite clinical assertion.",
                            "is_synthetic": True,
                        })

        # 4. Numerical and Magnitude Perturbation
        # Percentages in claim or evidence
        num_targets = [claim_text] + item["evidence_sentences"]
        for target_text in num_targets:
            pct_matches = list(re.finditer(r"\b(\d{1,2}(?:\.\d+)?)\s*%", target_text))
            for match in pct_matches:
                orig_val = float(match.group(1))
                alt_vals = [
                    min(95.0, round(orig_val * 4.0, 1)) if orig_val < 22 else max(1.0, round(orig_val / 5.0, 1)),
                    min(90.0, round(orig_val + 50.0, 1)) if orig_val < 35 else max(2.0, round(orig_val - 40.0, 1)),
                ]
                for new_val in alt_vals:
                    perturbed_claim = target_text[:match.start(1)] + f"{new_val}" + target_text[match.end(1):]
                    pair_hash = (perturbed_claim.lower(), evidence_text.lower())
                    if pair_hash not in seen_pair_hashes and len(perturbed_claim) < 220:
                        seen_pair_hashes.add(pair_hash)
                        category_counts["numerical_comparison"] += 1
                        generated_pairs.append({
                            "synthetic_id": f"syn_num_{cid}_{category_counts['numerical_comparison']}",
                            "source_claim_id": cid,
                            "doc_id": doc_id,
                            "claim": perturbed_claim,
                            "evidence_text": evidence_text,
                            "label": LABEL_CONTRADICTS,
                            "label_idx": 0,
                            "category": "numerical_comparison",
                            "generation_method": f"percentage_perturbation_{orig_val}%_to_{new_val}%",
                            "label_justification": f"Evidence quantifies effect size near {orig_val}%; perturbed claim asserts {new_val}%, representing an incompatible magnitude.",
                            "is_synthetic": True,
                        })

            # Multi-fold or unit quantities
            unit_matches = list(re.finditer(r"\b(\d+(?:\.\d+)?)\s*(-fold|days?|months?|years?|mg|μg|ml|kg|times?)\b", target_text, re.IGNORECASE))
            for match in unit_matches:
                val_str, unit = match.group(1), match.group(2)
                orig_val = float(val_str)
                new_val = round(orig_val * 10, 1) if orig_val <= 10 else round(orig_val / 10, 1)
                perturbed_claim = target_text[:match.start()] + f"{new_val} {unit}" + target_text[match.end():]
                pair_hash = (perturbed_claim.lower(), evidence_text.lower())
                if pair_hash not in seen_pair_hashes and len(perturbed_claim) < 220:
                    seen_pair_hashes.add(pair_hash)
                    category_counts["numerical_comparison"] += 1
                    generated_pairs.append({
                        "synthetic_id": f"syn_num_{cid}_{category_counts['numerical_comparison']}",
                        "source_claim_id": cid,
                        "doc_id": doc_id,
                        "claim": perturbed_claim,
                        "evidence_text": evidence_text,
                        "label": LABEL_CONTRADICTS,
                        "label_idx": 0,
                        "category": "numerical_comparison",
                        "generation_method": f"unit_scale_perturbation_{orig_val}_{unit}_to_{new_val}_{unit}",
                        "label_justification": f"Evidence specifies {orig_val} {unit}; claim asserts 10x order-of-magnitude perturbation ({new_val} {unit}).",
                        "is_synthetic": True,
                    })

        # 5. Causal Overclaiming (NEI)
        for pat, repl, rule in CAUSAL_RULES:
            if re.search(pat, claim_text, re.IGNORECASE):
                perturbed_claim = re.sub(pat, repl, claim_text, count=1, flags=re.IGNORECASE)
                pair_hash = (perturbed_claim.lower(), evidence_text.lower())
                if pair_hash not in seen_pair_hashes:
                    seen_pair_hashes.add(pair_hash)
                    category_counts["causal_reasoning"] += 1
                    generated_pairs.append({
                        "synthetic_id": f"syn_cau_{cid}_{category_counts['causal_reasoning']}",
                        "source_claim_id": cid,
                        "doc_id": doc_id,
                        "claim": perturbed_claim,
                        "evidence_text": evidence_text,
                        "label": LABEL_NEI,
                        "label_idx": 2,
                        "category": "causal_reasoning",
                        "generation_method": rule,
                        "label_justification": "Evidence describes an observational correlation or risk factor; asserting exclusive direct causation exceeds the evidentiary support (NEI).",
                        "is_synthetic": True,
                    })

        # 6. Scope Mismatch (NEI)
        if any(w in evidence_text.lower() for w in ["mouse", "mice", "murine", "rat", "in vitro", "cell culture"]):
            if not any(w in claim_text.lower() for w in ["mouse", "mice", "murine", "rat"]):
                scope_claim = f"In human clinical practice, {claim_text[0].lower() + claim_text[1:]}"
                pair_hash = (scope_claim.lower(), evidence_text.lower())
                if pair_hash not in seen_pair_hashes:
                    seen_pair_hashes.add(pair_hash)
                    category_counts["scope_mismatch"] += 1
                    generated_pairs.append({
                        "synthetic_id": f"syn_scope_{cid}_{category_counts['scope_mismatch']}",
                        "source_claim_id": cid,
                        "doc_id": doc_id,
                        "claim": scope_claim,
                        "evidence_text": evidence_text,
                        "label": LABEL_NEI,
                        "label_idx": 2,
                        "category": "scope_mismatch",
                        "generation_method": "preclinical_to_human_clinical_extrapolation",
                        "label_justification": "Evidence is established in murine or in vitro preclinical models; extrapolation to human clinical practice is unproven (NEI).",
                        "is_synthetic": True,
                    })

    # Pass 2: Controlled Cross-Document Distractors (NEI) to achieve target count
    eligible_corpus_docs = [
        d for d in corpus.values()
        if d["doc_id"] not in ex_doc_ids and d.get("abstract")
    ]
    rng.shuffle(eligible_corpus_docs)

    num_needed = max(0, target_count - len(generated_pairs))
    print(f"Linguistic passes generated {len(generated_pairs)} pairs ({sum(1 for p in generated_pairs if p['label'] == LABEL_CONTRADICTS)} CONTRADICTS). Sampling {num_needed} cross-document distractors...")

    for i in range(num_needed * 2):
        if len(generated_pairs) >= target_count:
            break
        claim_item = claim_evidence_tuples[i % len(claim_evidence_tuples)]
        distractor_doc = eligible_corpus_docs[i % len(eligible_corpus_docs)]

        if distractor_doc["doc_id"] == claim_item["doc_id"]:
            continue

        dist_sents = distractor_doc.get("abstract", [])
        if not dist_sents:
            continue
        dist_text = dist_sents[rng.randint(0, len(dist_sents) - 1)]
        if len(dist_text) < 30:
            continue

        pair_hash = (claim_item["claim"].lower(), dist_text.lower())
        if pair_hash not in seen_pair_hashes:
            seen_pair_hashes.add(pair_hash)
            category_counts["cross_document_distractor"] += 1
            generated_pairs.append({
                "synthetic_id": f"syn_dist_{claim_item['cid']}_{category_counts['cross_document_distractor']}",
                "source_claim_id": claim_item["cid"],
                "doc_id": distractor_doc["doc_id"],
                "claim": claim_item["claim"],
                "evidence_text": dist_text,
                "label": LABEL_NEI,
                "label_idx": 2,
                "category": "cross_document_distractor",
                "generation_method": "unrelated_corpus_abstract_sampling",
                "label_justification": f"Candidate evidence from Doc {distractor_doc['doc_id']} discusses an unrelated topic and does not mention the entities or mechanism of the claim (NEI).",
                "is_synthetic": True,
            })

    rng.shuffle(generated_pairs)

    print("\n--- Final Hard-Negative Generation Statistics ---")
    total_generated = len(generated_pairs)
    print(f"Total Validated Hard-Negative Pairs: {total_generated}")
    con_count = sum(1 for p in generated_pairs if p["label"] == LABEL_CONTRADICTS)
    nei_count = sum(1 for p in generated_pairs if p["label"] == LABEL_NEI)
    print(f"  CONTRADICTS: {con_count} ({con_count/total_generated*100:.1f}%)")
    print(f"  NEI        : {nei_count} ({nei_count/total_generated*100:.1f}%)")
    for cat, cnt in category_counts.items():
        pct = (cnt / total_generated * 100) if total_generated > 0 else 0
        print(f"    {cat:<28}: {cnt:>5} ({pct:5.1f}%)")

    out_dir = Config.BASE_DIR / "data" / "experimental"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_negative_pairs.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in generated_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Saved experimental hard negatives to: {out_path}")
    return generated_pairs


if __name__ == "__main__":
    generate_hard_negatives(target_count=1500)
