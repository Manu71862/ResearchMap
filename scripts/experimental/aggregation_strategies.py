"""
Phase 1: Evidence Aggregation Improvement Module
Implements and compares four evidence aggregation strategies:
1. Baseline: Existing Max-Probability Single-Sentence Thresholding
2. Confidence-Weighted Voting: Aggregates probabilities weighted by sentence prediction confidence
3. Relevance-Weighted Voting: Aggregates probabilities weighted by normalized retrieval relevance scores
4. Consensus-Based Aggregation: Calibrated multi-evidence consensus with conflict abstention and outlier dampening
"""

from typing import Dict, List, Any, Tuple, Optional
import math
import numpy as np

from backend.data_loader import LABEL_SUPPORTS, LABEL_CONTRADICTS, LABEL_NEI, CANONICAL_LABELS

class AggregationStrategy:
    """Base class for evidence aggregation strategies."""
    def aggregate(
        self,
        evidence_details: List[Dict[str, Any]],
        claim_text: str
    ) -> Tuple[str, float, List[Tuple[int, int]]]:
        raise NotImplementedError

class BaselineAggregation(AggregationStrategy):
    """
    Strategy 1: Existing Baseline (Production Pipeline logic).
    Takes max single-sentence probabilities with symmetric 0.50 thresholds.
    Vulnerable to the single-sentence veto flaw.
    """
    def __init__(self, entail_thresh: float = 0.50, contradict_thresh: float = 0.50):
        self.entail_thresh = entail_thresh
        self.contradict_thresh = contradict_thresh

    def aggregate(
        self,
        evidence_details: List[Dict[str, Any]],
        claim_text: str
    ) -> Tuple[str, float, List[Tuple[int, int]]]:
        if not evidence_details:
            return LABEL_NEI, 0.0, []

        best_sup = None
        best_con = None

        for d in evidence_details:
            p_sup = d["probabilities"].get(LABEL_SUPPORTS, 0.0)
            p_con = d["probabilities"].get(LABEL_CONTRADICTS, 0.0)

            if p_sup >= self.entail_thresh:
                if best_sup is None or p_sup > best_sup["probabilities"][LABEL_SUPPORTS]:
                    best_sup = d

            if p_con >= self.contradict_thresh:
                if best_con is None or p_con > best_con["probabilities"][LABEL_CONTRADICTS]:
                    best_con = d

        if best_support := best_sup:
            if best_contradict := best_con:
                sup_conf = best_support["probabilities"][LABEL_SUPPORTS]
                con_conf = best_contradict["probabilities"][LABEL_CONTRADICTS]
                if sup_conf >= con_conf:
                    return LABEL_SUPPORTS, sup_conf, [(best_support["doc_id"], best_support["sentence_idx"])]
                else:
                    return LABEL_CONTRADICTS, con_conf, [(best_contradict["doc_id"], best_contradict["sentence_idx"])]
            return LABEL_SUPPORTS, best_support["probabilities"][LABEL_SUPPORTS], [(best_support["doc_id"], best_support["sentence_idx"])]
        elif best_contradict := best_con:
            return LABEL_CONTRADICTS, best_contradict["probabilities"][LABEL_CONTRADICTS], [(best_contradict["doc_id"], best_contradict["sentence_idx"])]
        else:
            max_nei = max((d["probabilities"].get(LABEL_NEI, 0.0) for d in evidence_details), default=0.0)
            return LABEL_NEI, max_nei, []

class ConfidenceWeightedVotingAggregation(AggregationStrategy):
    """
    Strategy 2: Confidence-Weighted Voting.
    Aggregates class probabilities across candidate sentences weighted by sentence margin.
    Includes conflict margin threshold to abstain to NEI when support and contradiction are balanced.
    """
    def __init__(self, entail_thresh: float = 0.45, contradict_thresh: float = 0.45, conflict_margin: float = 0.15):
        self.entail_thresh = entail_thresh
        self.contradict_thresh = contradict_thresh
        self.conflict_margin = conflict_margin

    def aggregate(
        self,
        evidence_details: List[Dict[str, Any]],
        claim_text: str
    ) -> Tuple[str, float, List[Tuple[int, int]]]:
        if not evidence_details:
            return LABEL_NEI, 0.0, []

        weights = []
        for d in evidence_details:
            conf = d.get("confidence", 0.5)
            # Weight is scaled by distance from uniform distribution (1/3)
            w = max(conf - 0.333, 0.05)
            weights.append(w)

        total_weight = sum(weights)
        if total_weight <= 0:
            total_weight = 1.0

        v_sup = sum(w * d["probabilities"].get(LABEL_SUPPORTS, 0.0) for w, d in zip(weights, evidence_details)) / total_weight
        v_con = sum(w * d["probabilities"].get(LABEL_CONTRADICTS, 0.0) for w, d in zip(weights, evidence_details)) / total_weight
        v_nei = sum(w * d["probabilities"].get(LABEL_NEI, 0.0) for w, d in zip(weights, evidence_details)) / total_weight

        # Check for conflict impasse
        if v_sup >= self.entail_thresh and v_con >= self.contradict_thresh:
            if abs(v_sup - v_con) < self.conflict_margin:
                # Evidence is in genuine conflict; abstain to NEI
                return LABEL_NEI, float(max(v_nei, 0.5)), []

        if v_sup >= self.entail_thresh and v_sup > v_con and v_sup > v_nei:
            best_s = max(evidence_details, key=lambda d: d["probabilities"].get(LABEL_SUPPORTS, 0.0))
            return LABEL_SUPPORTS, float(v_sup), [(best_s["doc_id"], best_s["sentence_idx"])]

        if v_con >= self.contradict_thresh and v_con > v_sup and v_con > v_nei:
            best_c = max(evidence_details, key=lambda d: d["probabilities"].get(LABEL_CONTRADICTS, 0.0))
            return LABEL_CONTRADICTS, float(v_con), [(best_c["doc_id"], best_c["sentence_idx"])]

        return LABEL_NEI, float(v_nei), []

class RelevanceWeightedVotingAggregation(AggregationStrategy):
    """
    Strategy 3: Relevance-Weighted Voting.
    Weights each sentence's NLI distribution by its retrieval relevance score (BM25 sentence score).
    Sentences with higher lexical overlap have higher voting power than marginal distractors.
    """
    def __init__(self, entail_thresh: float = 0.45, contradict_thresh: float = 0.45, conflict_margin: float = 0.15):
        self.entail_thresh = entail_thresh
        self.contradict_thresh = contradict_thresh
        self.conflict_margin = conflict_margin

    def aggregate(
        self,
        evidence_details: List[Dict[str, Any]],
        claim_text: str
    ) -> Tuple[str, float, List[Tuple[int, int]]]:
        if not evidence_details:
            return LABEL_NEI, 0.0, []

        raw_rel = [max(float(d.get("bm25_score", 1.0)), 0.1) for d in evidence_details]
        total_rel = sum(raw_rel)
        if total_rel <= 0:
            total_rel = 1.0

        weights = [r / total_rel for r in raw_rel]

        v_sup = sum(w * d["probabilities"].get(LABEL_SUPPORTS, 0.0) for w, d in zip(weights, evidence_details))
        v_con = sum(w * d["probabilities"].get(LABEL_CONTRADICTS, 0.0) for w, d in zip(weights, evidence_details))
        v_nei = sum(w * d["probabilities"].get(LABEL_NEI, 0.0) for w, d in zip(weights, evidence_details))

        if v_sup >= self.entail_thresh and v_con >= self.contradict_thresh:
            if abs(v_sup - v_con) < self.conflict_margin:
                return LABEL_NEI, float(max(v_nei, 0.5)), []

        if v_sup >= self.entail_thresh and v_sup > v_con and v_sup > v_nei:
            best_s = max(evidence_details, key=lambda d: d["probabilities"].get(LABEL_SUPPORTS, 0.0))
            return LABEL_SUPPORTS, float(v_sup), [(best_s["doc_id"], best_s["sentence_idx"])]

        if v_con >= self.contradict_thresh and v_con > v_sup and v_con > v_nei:
            best_c = max(evidence_details, key=lambda d: d["probabilities"].get(LABEL_CONTRADICTS, 0.0))
            return LABEL_CONTRADICTS, float(v_con), [(best_c["doc_id"], best_c["sentence_idx"])]

        return LABEL_NEI, float(v_nei), []

class ConsensusBasedAggregation(AggregationStrategy):
    """
    Strategy 4: Calibrated Consensus-Based Aggregation.
    Specifically designed to eliminate the single-sentence veto flaw:
    - Multi-evidence support/contradiction is accepted immediately.
    - A single refuting sentence is only accepted if it has decisive confidence (P >= 0.75)
      and is not overwhelmingly opposed by multiple confirming NEI sentences.
    - If evidence is conflicting, abstains to NOT ENOUGH INFORMATION.
    """
    def __init__(
        self,
        base_thresh: float = 0.50,
        high_conf_single_thresh: float = 0.75,
        neutral_consensus_thresh: float = 0.70,
        conflict_tolerance: float = 0.15
    ):
        self.base_thresh = base_thresh
        self.high_conf_single_thresh = high_conf_single_thresh
        self.neutral_consensus_thresh = neutral_consensus_thresh
        self.conflict_tolerance = conflict_tolerance

    def aggregate(
        self,
        evidence_details: List[Dict[str, Any]],
        claim_text: str
    ) -> Tuple[str, float, List[Tuple[int, int]]]:
        if not evidence_details:
            return LABEL_NEI, 0.0, []

        sup_sentences = []
        con_sentences = []
        nei_sentences = []

        for d in evidence_details:
            p_s = d["probabilities"].get(LABEL_SUPPORTS, 0.0)
            p_c = d["probabilities"].get(LABEL_CONTRADICTS, 0.0)
            p_n = d["probabilities"].get(LABEL_NEI, 0.0)

            if p_s >= self.base_thresh and p_s > p_c:
                sup_sentences.append(d)
            elif p_c >= self.base_thresh and p_c > p_s:
                con_sentences.append(d)
            else:
                nei_sentences.append(d)

        avg_nei_prob = np.mean([d["probabilities"].get(LABEL_NEI, 0.0) for d in evidence_details])

        # 1. Check for genuine conflict impasse between sentences
        if sup_sentences and con_sentences:
            max_s = max(d["probabilities"][LABEL_SUPPORTS] for d in sup_sentences)
            max_c = max(d["probabilities"][LABEL_CONTRADICTS] for d in con_sentences)
            if abs(max_s - max_c) < self.conflict_tolerance:
                # Impasse: candidate sentences contradict each other with comparable confidence
                return LABEL_NEI, float(avg_nei_prob), []
            elif max_s > max_c and len(sup_sentences) >= len(con_sentences):
                best = max(sup_sentences, key=lambda d: d["probabilities"][LABEL_SUPPORTS])
                return LABEL_SUPPORTS, float(max_s), [(best["doc_id"], best["sentence_idx"])]
            elif max_c > max_s and len(con_sentences) >= len(sup_sentences):
                best = max(con_sentences, key=lambda d: d["probabilities"][LABEL_CONTRADICTS])
                return LABEL_CONTRADICTS, float(max_c), [(best["doc_id"], best["sentence_idx"])]

        # 2. Support consensus
        if len(sup_sentences) >= 2:
            best = max(sup_sentences, key=lambda d: d["probabilities"][LABEL_SUPPORTS])
            return LABEL_SUPPORTS, float(best["probabilities"][LABEL_SUPPORTS]), [(best["doc_id"], best["sentence_idx"])]
        elif len(sup_sentences) == 1:
            best = sup_sentences[0]
            p_s = best["probabilities"][LABEL_SUPPORTS]
            # If solitary support sentence has marginal confidence and remaining sentences are heavily neutral
            if p_s < 0.65 and avg_nei_prob > self.neutral_consensus_thresh:
                return LABEL_NEI, float(avg_nei_prob), []
            if p_s >= self.base_thresh:
                return LABEL_SUPPORTS, float(p_s), [(best["doc_id"], best["sentence_idx"])]

        # 3. Contradiction consensus (Veto mitigation)
        if len(con_sentences) >= 2:
            best = max(con_sentences, key=lambda d: d["probabilities"][LABEL_CONTRADICTS])
            return LABEL_CONTRADICTS, float(best["probabilities"][LABEL_CONTRADICTS]), [(best["doc_id"], best["sentence_idx"])]
        elif len(con_sentences) == 1:
            best = con_sentences[0]
            p_c = best["probabilities"][LABEL_CONTRADICTS]
            # Key Veto Protection:
            # If only 1 sentence contradicts, but it has marginal confidence (< 0.75) and multiple sentences are neutral
            if p_c < self.high_conf_single_thresh and (len(nei_sentences) >= 2 or avg_nei_prob > 0.65):
                # Dampen isolated outlier; abstain to NEI
                return LABEL_NEI, float(avg_nei_prob), []
            if p_c >= self.base_thresh:
                return LABEL_CONTRADICTS, float(p_c), [(best["doc_id"], best["sentence_idx"])]

        # Default to NOT ENOUGH INFORMATION
        max_nei = max((d["probabilities"].get(LABEL_NEI, 0.0) for d in evidence_details), default=0.0)
        return LABEL_NEI, float(max_nei), []
