import re
from typing import Dict, List, Any, Optional
from backend.data_loader import (
    LABEL_SUPPORTS,
    LABEL_CONTRADICTS,
    LABEL_NEI,
    CANONICAL_LABELS,
)
from backend.retrieval import BM25Retriever
from backend.entailment import NLIEntailmentModel
from backend.config import Config

# ==============================================================================
# Condition-Aware Reversibility Criterion Linguistic Patterns
# Evaluated and validated in Phase 11 & Final Safety Audit
# ==============================================================================

CESSATION_MODIFIERS = [
    r"\bcessation\b", r"\bquitting\b", r"\bwithdrawal\b",
    r"\bdiscontinuation\b", r"\babstinence\b", r"\bstopping\b"
]

REDUCTION_PREDICATES = [
    r"\breduces?\b", r"\breduced\b", r"\breducing\b", r"\breduction\b",
    r"\bdecreases?\b", r"\bdecreased\b", r"\bdecreasing\b",
    r"\blowers?\b", r"\blowered\b", r"\blowering\b",
    r"\bprevents?\b", r"\bprevented\b", r"\bpreventing\b", r"\bprevention\b",
    r"\bhalts?\b", r"\bhalted\b", r"\bhalting\b"
]

INCREASE_PREDICATES = [
    r"\bincreases?\b", r"\bincreased\b", r"\bincreasing\b",
    r"\belevates?\b", r"\belevated\b", r"\belevating\b",
    r"\bpromotes?\b", r"\bpromoted\b", r"\bpromoting\b",
    r"\baccelerates?\b", r"\baccelerated\b", r"\baccelerating\b",
    r"\bcauses?\b", r"\bcaused\b", r"\bcausing\b"
]

def is_exposure_claim(claim_text: str) -> bool:
    """Checks whether the claim asserts an active exposure increases/causes an adverse outcome (E -> +Y)."""
    c_low = claim_text.lower()
    has_inc = any(re.search(p, c_low) for p in INCREASE_PREDICATES)
    has_cess = any(re.search(p, c_low) for p in CESSATION_MODIFIERS)
    return has_inc and not has_cess

def is_cessation_reversibility_sentence(sentence_text: str) -> bool:
    """
    Checks whether the sentence describes an intervention/cessation reducing risk
    (Bradford Hill's Reversibility Criterion: Cessation(E) -> -Y).
    """
    s_low = sentence_text.lower()
    has_cess = any(re.search(p, s_low) for p in CESSATION_MODIFIERS)
    has_red = any(re.search(p, s_low) for p in REDUCTION_PREDICATES)
    return has_cess and has_red


class ClaimVerificationPipeline:
    """
    End-to-end scientific claim verification pipeline.
    Combines BM25 lexical candidate retrieval with pretrained NLI cross-encoder entailment scoring.
    Incorporates Bradford Hill Condition-Aware Reversibility Filtering.
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        entailment_model: NLIEntailmentModel,
        top_k_docs: int = Config.BM25_TOP_K_DOCS,
        top_k_sentences: int = Config.BM25_TOP_K_SENTENCES,
        entail_threshold: float = Config.ENTAIL_THRESHOLD,
        contradict_threshold: float = Config.CONTRADICT_THRESHOLD,
    ):
        self.retriever = retriever
        self.entailment_model = entailment_model
        self.top_k_docs = top_k_docs
        self.top_k_sentences = top_k_sentences
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold

    def verify_claim(self, claim_text: str, claim_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run verification on a single claim string.
        """
        # Step 1: Lexical BM25 retrieval of candidate documents
        candidate_docs = self.retriever.retrieve_documents(claim_text, top_k=self.top_k_docs)
        retrieved_doc_ids = [doc["doc_id"] for doc in candidate_docs]

        if not candidate_docs:
            return {
                "claim_id": claim_id,
                "claim": claim_text,
                "predicted_label": LABEL_NEI,
                "confidence": 0.0,
                "predicted_evidence_doc_ids": [],
                "predicted_evidence_sentences": [],
                "evidence_details": [],
                "retrieved_candidate_doc_ids": [],
                "explanation": "No matching candidate documents retrieved by BM25.",
                "reversibility_detected": False,
            }

        # Step 2: Lexical BM25 retrieval of candidate sentences across candidates
        candidate_sentences = self.retriever.retrieve_sentences(
            claim_text, candidate_docs, top_k_sentences=self.top_k_sentences
        )

        if not candidate_sentences:
            return {
                "claim_id": claim_id,
                "claim": claim_text,
                "predicted_label": LABEL_NEI,
                "confidence": 0.0,
                "predicted_evidence_doc_ids": [],
                "predicted_evidence_sentences": [],
                "evidence_details": [],
                "retrieved_candidate_doc_ids": retrieved_doc_ids,
                "explanation": "No candidate sentences extracted from retrieved documents.",
                "reversibility_detected": False,
            }

        # Step 3: NLI cross-encoder evaluation of (sentence, claim) pairs
        pairs = [(s["text"], claim_text) for s in candidate_sentences]
        nli_outputs = self.entailment_model.predict_batch(pairs)

        # Step 4: Associate predictions with sentence details and evaluate evidence
        evidence_details = []
        best_support = None
        best_contradict = None
        claim_is_exposure = is_exposure_claim(claim_text)
        reversibility_detected = False

        for sent_info, nli_out in zip(candidate_sentences, nli_outputs):
            doc_id = sent_info["doc_id"]
            doc_data = self.retriever.corpus.get(doc_id, {})
            title = doc_data.get("title", "")

            # Check if this sentence satisfies Hill's Reversibility Criterion:
            # Condition 1: Claim asserts active exposure increases an adverse outcome
            # Condition 2: Evidence sentence describes cessation/reversibility and risk reduction
            is_rev = claim_is_exposure and is_cessation_reversibility_sentence(sent_info["text"])
            if is_rev:
                reversibility_detected = True
                rev_rationale = (
                    "Excluded from contradiction veto under Hill's Reversibility Criterion: "
                    "Evidence demonstrates that cessation or removal of exposure reduces adverse outcome, "
                    "which corroborates rather than refutes causal risk."
                )
            else:
                rev_rationale = None

            detail = {
                "doc_id": doc_id,
                "sentence_idx": sent_info["sentence_idx"],
                "text": sent_info["text"],
                "title": title,
                "bm25_score": sent_info["bm25_score"],
                "entailment_label": nli_out["prediction"],
                "confidence": nli_out["confidence"],
                "probabilities": dict(nli_out["probabilities"]),
                "veto_masked": is_rev,
                "reversibility_rationale": rev_rationale,
            }
            evidence_details.append(detail)

            # Check threshold criteria
            p_sup = nli_out["probabilities"].get(LABEL_SUPPORTS, 0.0)
            p_con = nli_out["probabilities"].get(LABEL_CONTRADICTS, 0.0)

            if p_sup >= self.entail_threshold:
                if best_support is None or p_sup > best_support["probabilities"][LABEL_SUPPORTS]:
                    best_support = detail

            # Only consider for contradiction veto if veto is not masked under reversibility
            if not is_rev and p_con >= self.contradict_threshold:
                if best_contradict is None or p_con > best_contradict["probabilities"][LABEL_CONTRADICTS]:
                    best_contradict = detail

        # Step 5: Claim-level decision
        if best_support and best_contradict:
            # Resolving conflict by highest probability
            sup_conf = best_support["probabilities"][LABEL_SUPPORTS]
            con_conf = best_contradict["probabilities"][LABEL_CONTRADICTS]
            if sup_conf >= con_conf:
                pred_label = LABEL_SUPPORTS
                selected_ev = best_support
                confidence = sup_conf
            else:
                pred_label = LABEL_CONTRADICTS
                selected_ev = best_contradict
                confidence = con_conf
            pred_docs = [selected_ev["doc_id"]]
            pred_sents = [(selected_ev["doc_id"], selected_ev["sentence_idx"])]
        elif best_support:
            pred_label = LABEL_SUPPORTS
            selected_ev = best_support
            confidence = best_support["probabilities"][LABEL_SUPPORTS]
            pred_docs = [selected_ev["doc_id"]]
            pred_sents = [(selected_ev["doc_id"], selected_ev["sentence_idx"])]
        elif best_contradict:
            pred_label = LABEL_CONTRADICTS
            selected_ev = best_contradict
            confidence = best_contradict["probabilities"][LABEL_CONTRADICTS]
            pred_docs = [selected_ev["doc_id"]]
            pred_sents = [(selected_ev["doc_id"], selected_ev["sentence_idx"])]
        else:
            # Evidence insufficient to support or contradict
            pred_label = LABEL_NEI
            # Max neutral probability among candidate sentences as confidence
            max_neutral = max(
                (d["probabilities"].get(LABEL_NEI, 0.0) for d in evidence_details),
                default=0.0
            )
            confidence = max_neutral
            pred_docs = []
            pred_sents = []

        return {
            "claim_id": claim_id,
            "claim": claim_text,
            "predicted_label": pred_label,
            "confidence": float(confidence),
            "predicted_evidence_doc_ids": pred_docs,
            "predicted_evidence_sentences": pred_sents,
            "evidence_details": evidence_details,
            "retrieved_candidate_doc_ids": retrieved_doc_ids,
            "reversibility_detected": reversibility_detected,
        }
