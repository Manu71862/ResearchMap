"""
Unit tests for the isolated experimental numerical reasoning module.

Tests:
1. 493 per million vs 1 in 2,000: approximately equivalent.
2. 25% vs 1 in 4: mathematically equivalent.
3. 10% vs 50%: contradictory.
4. 493 per million vs 1 in 5 million: contradictory.
5. Confidence interval containment (95% CI 282 to 784 per million).
6. Missing quantities (text without numbers -> UNKNOWN).
7. Ambiguous expressions -> UNKNOWN.
8. Rounding tolerance behavior.
9. Verification that context caveat is explicitly provided.
"""

import pytest
from backend.experimental_numerical import (
    ScientificNumericalParser,
    RELATION_EQUIVALENT,
    RELATION_CONTRADICTORY,
    RELATION_UNKNOWN,
)


@pytest.fixture
def parser():
    return ScientificNumericalParser(rounding_tolerance=0.20)


def test_493_per_million_vs_1_in_2000_approximately_equivalent(parser):
    premise = "16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population."
    claim = "1/2000 in UK have abnormal PrP positivity."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_EQUIVALENT
    assert result.relative_difference is not None
    assert result.relative_difference < 0.02  # ~1.4%
    assert result.premise_quantity is not None
    assert result.claim_quantity is not None


def test_confidence_interval_containment(parser):
    premise = (
        "16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population "
        "(95% confidence interval 282 to 784 per million)."
    )
    claim = "1 in 2,000 in UK have abnormal PrP positivity."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_EQUIVALENT
    assert result.is_within_confidence_interval is True
    assert result.premise_quantity.ci_lower == pytest.approx(282 / 1e6)
    assert result.premise_quantity.ci_upper == pytest.approx(784 / 1e6)


def test_25_percent_vs_1_in_4_equivalent(parser):
    premise = "The response rate was 25 percent across the trial cohort."
    claim = "Approximately 1 in 4 patients responded to therapy."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_EQUIVALENT
    assert result.relative_difference == pytest.approx(0.0)


def test_10_percent_vs_50_percent_contradictory(parser):
    premise = "Overall 5-year survival was 10 percent in the treated group."
    claim = "Overall 5-year survival was 50% in the treated group."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_CONTRADICTORY
    assert "contradictory" in result.explanation.lower()


def test_493_per_million_vs_1_in_5_million_contradictory(parser):
    premise = "Prevalence of abnormal PrP was 493 per million in the cohort."
    claim = "1 in 5 million in UK have abnormal PrP positivity."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_CONTRADICTORY
    assert result.premise_quantity.normalized_value > result.claim_quantity.normalized_value * 1000


def test_missing_quantities_returns_unknown(parser):
    premise = "ADAR1 forms a complex with Dicer to promote pre-miRNA cleavage."
    claim = "ADAR1 binds to Dicer to cleave pre-miRNA."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_UNKNOWN
    assert result.premise_quantity is None
    assert result.claim_quantity is None


def test_ambiguous_denominators_returns_unknown(parser):
    premise = "A significant proportion of several individuals tested positive."
    claim = "Prevalence is 1 in several thousands."
    result = parser.compare(premise, claim)

    assert result.relation == RELATION_UNKNOWN


def test_rounding_tolerance_threshold(parser):
    # 1 in 1000 (0.001) vs 1 in 1050 (0.000952): diff ~4.8% -> within tolerance -> EQUIVALENT
    res_close = parser.compare("Prevalence is 1 in 1000.", "Prevalence is 1 in 1050.")
    assert res_close.relation == RELATION_EQUIVALENT

    # 1 in 1000 (0.001) vs 1 in 1600 (0.000625): diff ~37.5% -> outside tolerance -> UNKNOWN
    res_mid = parser.compare("Prevalence is 1 in 1000.", "Prevalence is 1 in 1600.")
    assert res_mid.relation == RELATION_UNKNOWN


def test_context_caveat_preserved(parser):
    premise = "Prevalence was 25 percent in adult males."
    claim = "1 in 4 pediatric patients experienced remission."
    result = parser.compare(premise, claim)

    # Numbers match mathematically (0.25 vs 0.25)
    assert result.relation == RELATION_EQUIVALENT
    # But context caveat explicitly states this does not prove clinical entailment
    assert "does not independently establish semantic entailment" in result.context_caveat
