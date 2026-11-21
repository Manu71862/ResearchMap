"""
Unit tests for ScientificEntityGroundingFilter (Phase 2 Experimental)
"""

import pytest
from scripts.experimental.entity_grounding import ScientificEntityGroundingFilter, GroundingResult


@pytest.fixture
def filter_eg():
    return ScientificEntityGroundingFilter()


class TestEntityGroundingFilter:
    """Test suite for ScientificEntityGroundingFilter."""

    # 1. Different Organisms
    def test_different_organisms_yeast_vs_rat(self, filter_eg):
        claim = "Glucose restriction reduces RLS in S. cerevisiae."
        sent = "Increasing energy intake in rats failed to increase growth."
        title = "Low methionine ingestion by rats extends life span."
        abstract = ["We studied Fischer 344 rats."]
        result = filter_eg.evaluate_grounding(claim, sent, title, abstract)
        assert result.status == "REJECTED"
        assert not result.is_accepted
        assert any("organism mismatch" in r.lower() for r in result.rejection_reasons)

    def test_different_organisms_bacteria_vs_human(self, filter_eg):
        claim = "E. coli replication initiation requires DnaA ATP binding."
        sent = "Human clinical trials showed no toxicity in adult patients."
        title = "Phase I evaluation in human volunteers."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "REJECTED"
        assert not result.is_accepted

    def test_different_organisms_nematode_vs_mouse(self, filter_eg):
        claim = "DAF-16 regulates longevity in C. elegans."
        sent = "Transgenic mice exhibited increased cardiac hypertrophy."
        title = "Murine models of heart failure."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "REJECTED"
        assert not result.is_accepted

    # 2. Different Diseases
    def test_different_diseases_prostate_vs_breast(self, filter_eg):
        claim = "Alpha-tocopheryl acetate reduces the risk of prostate cancer."
        sent = "Tamoxifen significantly reduced recurrence in breast cancer patients."
        title = "Adjuvant therapy in estrogen-receptor positive breast cancer."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "REJECTED"
        assert not result.is_accepted
        assert any("disease mismatch" in r.lower() for r in result.rejection_reasons)

    def test_different_diseases_asthma_vs_alzheimers(self, filter_eg):
        claim = "Childhood asthma is reduced by maternal microbiome diversity."
        sent = "Amyloid plaque accumulation was measured in Alzheimer's disease brain tissue."
        title = "Neuropathological hallmarks of Alzheimer's disease."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "REJECTED"
        assert not result.is_accepted

    # 3. Different Interventions
    def test_different_interventions_glucose_vs_methionine(self, filter_eg):
        claim = "Glucose restriction to 0.05% extends lifespan."
        sent = "Methionine supplementation did not alter animal body weight."
        title = "Dietary methionine restriction and life span."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "REJECTED"
        assert not result.is_accepted
        assert any("intervention mismatch" in r.lower() for r in result.rejection_reasons)

    def test_different_interventions_antibiotics_vs_sunscreen(self, filter_eg):
        claim = "Antibiotic treatment disrupts intestinal microbiota."
        sent = "Daily application of sunscreen prevents ultraviolet-induced erythema."
        title = "Photoprotection using broad-spectrum sunscreen."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "REJECTED"
        assert not result.is_accepted

    # 4. Synonyms and Abbreviations
    def test_synonyms_yeast_abbreviations(self, filter_eg):
        claim = "S. cerevisiae genes are essential for viability."
        sent = "Budding yeast mutants were screened on synthetic complete media."
        title = "Functional profiling in baker's yeast."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "ACCEPTED"
        assert result.is_accepted
        assert any("Organism match" in m for m in result.matches)

    def test_synonyms_vitamins(self, filter_eg):
        claim = "Taking α-tocopheryl acetate reduces cancer incidence."
        sent = "Vitamin E supplementation was evaluated in clinical trials."
        title = "Alpha-tocopherol and total mortality."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "ACCEPTED"
        assert result.is_accepted
        assert any("Intervention match" in m for m in result.matches)

    def test_synonyms_bacteria(self, filter_eg):
        claim = "Escherichia coli growth is inhibited by bacteriocins."
        sent = "E. coli cultures were exposed to various compounds."
        title = "Antimicrobial resistance in E. coli."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status == "ACCEPTED"
        assert result.is_accepted

    # 5. Missing Entities
    def test_missing_entities_epistemological_claim(self, filter_eg):
        claim = "A scientific study's association between two variables necessarily proves causation."
        sent = "Observational epidemiological findings often fail to replicate in randomized trials."
        title = "Methodological limitations of cohort studies."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.status in ("ACCEPTED", "FLAGGED_UNCERTAIN")
        assert result.is_accepted

    # 6. Ambiguous Entities & Hierarchical Compatibility
    def test_ambiguous_entities_general_cancer_compatibility(self, filter_eg):
        claim = "Common variants in BRCA1 do not cause cancer."
        sent = "We evaluated breast cancer and ovarian cancer incidence."
        title = "BRCA1 polymorphisms in familial breast cancer cohorts."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        assert result.is_accepted
        assert not result.rejection_reasons

    def test_ambiguous_unspecified_organism_preservation(self, filter_eg):
        claim = "Glucose restriction slows aging in S. cerevisiae."
        sent = "Kinase phosphorylation of the activation loop was measured by western blot."
        title = "Biochemical characterization of AMPK phosphorylation."
        result = filter_eg.evaluate_grounding(claim, sent, title, [])
        # When organism is not mentioned and no conflicting organism exists, it should be preserved!
        assert result.is_accepted
        assert result.status == "FLAGGED_UNCERTAIN"

    # 7. Relevant Evidence with Implicit Subjects
    def test_implicit_subject_in_sentence_resolved_by_context(self, filter_eg):
        claim = "Glucose restriction reduces RLS in S. cerevisiae."
        # The sentence itself contains NO mention of yeast or S. cerevisiae:
        sent = "Replicative life span was diminished by 30% under standard conditions."
        # But document title and abstract specify the model organism:
        title = "Caloric restriction and life span in Saccharomyces cerevisiae."
        abstract = ["We examined budding yeast senescence.", "Sir2 regulates replicative longevity."]
        result = filter_eg.evaluate_grounding(claim, sent, title, abstract)
        assert result.status == "ACCEPTED"
        assert result.is_accepted
        assert any("Organism match" in m for m in result.matches)

    # 8. Filter Candidate Sentences Helper
    def test_filter_candidate_sentences(self, filter_eg):
        claim = "Glucose restriction reduces RLS in S. cerevisiae."
        corpus = {
            100: {
                "title": "Caloric restriction in Saccharomyces cerevisiae.",
                "abstract": ["Budding yeast lifespan was examined."],
            },
            200: {
                "title": "Low methionine ingestion by rats extends life span.",
                "abstract": ["Fischer 344 rats were studied."],
            },
        }
        candidates = [
            {"doc_id": 100, "sentence_idx": 0, "text": "Budding yeast lifespan was examined."},
            {"doc_id": 200, "sentence_idx": 0, "text": "Increasing energy intake in rats failed to increase growth."},
        ]
        accepted, rejected = filter_eg.filter_candidate_sentences(claim, candidates, corpus)
        assert len(accepted) == 1
        assert accepted[0]["doc_id"] == 100
        assert len(rejected) == 1
        assert rejected[0]["doc_id"] == 200
