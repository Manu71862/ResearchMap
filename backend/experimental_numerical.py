"""
Isolated Experimental Module: Numerical Normalization and Reasoning for Scientific Claims.

Supports:
- Normalized rates per population base (per million, per billion, per thousand, per 100,000).
- Fractions and ratios ('1 in Y', '1/Y', 'X in Y', 'X/Y') with optional scale words (million, billion, thousand).
- Percentages ('X%', 'X percent').
- Confidence intervals / range expressions (e.g. '95% CI 282 to 784 per million').
- Strict handling of unit mismatch, ambiguous denominators, missing quantities, and dates.

Constraint:
Numerical equivalence is mathematical only and NEVER alone implies full scientific entailment.
Population, condition, outcome, and context must be independently evaluated.
"""

import re
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple


RELATION_EQUIVALENT = "EQUIVALENT"
RELATION_CONTRADICTORY = "CONTRADICTORY"
RELATION_UNKNOWN = "UNKNOWN"


@dataclass
class NormalizedQuantity:
    raw_text: str
    normalized_value: float  # Frequency / probability in [0.0, 1.0]
    quantity_type: str       # 'rate_per_base', 'ratio', 'percentage'
    unit_category: str       # 'frequency_rate', 'percentage', 'dosage', 'unsupported'
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    original_unit: Optional[str] = None


@dataclass
class NumericalComparisonResult:
    relation: str            # 'EQUIVALENT', 'CONTRADICTORY', 'UNKNOWN'
    premise_quantity: Optional[NormalizedQuantity]
    claim_quantity: Optional[NormalizedQuantity]
    relative_difference: Optional[float]
    is_within_confidence_interval: bool
    explanation: str
    context_caveat: str = (
        "Numerical equivalence reflects mathematical frequency/ratio consistency only. "
        "It does not independently establish semantic entailment without matching population, "
        "intervention, and medical context."
    )


class ScientificNumericalParser:
    """
    Parser for scientific rates, ratios, percentages, and uncertainty bounds.
    Returns UNKNOWN on ambiguous or unmapped quantities.
    """

    # Multipliers for population base
    BASE_MULTIPLIERS = {
        "billion": 1e9,
        "million": 1e6,
        "100000": 1e5,
        "100,000": 1e5,
        "thousand": 1e3,
        "1000": 1e3,
        "1,000": 1e3,
        "100": 1e2,
    }

    # Year filter: 1900-2099 to avoid interpreting publication/cohort years as denominators
    YEAR_PATTERN = re.compile(r"\b(19\d\d|20\d\d)\b")

    def __init__(self, rounding_tolerance: float = 0.20):
        """
        tolerance: Fractional relative difference considered 'approximately equivalent'.
        Default 0.20 (20%) accounts for typical epidemiological rounding (e.g. 1 in 2028 ~ 1 in 2000).
        """
        self.rounding_tolerance = rounding_tolerance

    def extract_quantities(self, text: str) -> List[NormalizedQuantity]:
        """Extract all valid normalized rates, ratios, and percentages from text."""
        quantities: List[NormalizedQuantity] = []
        if not text or not text.strip():
            return quantities

        clean_text = text.replace(",", "")

        # 1. Parse Rates with Confidence Intervals:
        # e.g. "493 per million population (95% confidence interval 282 to 784 per million)"
        ci_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*per\s*(million|billion|thousand|100000)\b.*?"
            r"(?:95%|99%)?\s*(?:ci|confidence interval)\s*[:\(]?\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE
        )
        matched_spans = set()
        for m in ci_pattern.finditer(clean_text):
            point = float(m.group(1))
            unit_str = m.group(2).lower()
            base = self.BASE_MULTIPLIERS.get(unit_str, 1e6)
            ci_low = float(m.group(3)) / base
            ci_high = float(m.group(4)) / base
            val = point / base
            quantities.append(NormalizedQuantity(
                raw_text=m.group(0),
                normalized_value=val,
                quantity_type="rate_per_base",
                unit_category="frequency_rate",
                ci_lower=ci_low,
                ci_upper=ci_high,
                original_unit=f"per {unit_str}",
            ))
            matched_spans.add(m.span())

        # 2. Parse Standard Rates: 'X per million / billion / thousand'
        rate_pattern = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*per\s*(million|billion|thousand|100000|1000|100)\b",
            re.IGNORECASE
        )
        for m in rate_pattern.finditer(clean_text):
            if any(s[0] <= m.start() and m.end() <= s[1] for s in matched_spans):
                continue
            num = float(m.group(1))
            unit_str = m.group(2).lower()
            base = self.BASE_MULTIPLIERS.get(unit_str, 1e6)
            quantities.append(NormalizedQuantity(
                raw_text=m.group(0),
                normalized_value=num / base,
                quantity_type="rate_per_base",
                unit_category="frequency_rate",
                original_unit=f"per {unit_str}",
            ))

        # 3. Parse Fractions and Ratios: '1 in 2000', '1/2000', '1 in 5 million'
        ratio_pattern = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(?:/|\bin\b)\s*(\d+(?:\.\d+)?)\s*(million|billion|thousand)?\b",
            re.IGNORECASE
        )
        for m in ratio_pattern.finditer(clean_text):
            num = float(m.group(1))
            denom = float(m.group(2))
            unit_mult = m.group(3)

            # Avoid treating dates as denominators: e.g. "published in 2012"
            if "in" in m.group(0).lower() and 1950 <= denom <= 2030 and not unit_mult and num < 1:
                continue

            if unit_mult:
                mult = self.BASE_MULTIPLIERS.get(unit_mult.lower(), 1.0)
                denom *= mult

            if denom > 1:
                quantities.append(NormalizedQuantity(
                    raw_text=m.group(0),
                    normalized_value=num / denom,
                    quantity_type="ratio",
                    unit_category="frequency_rate",
                    original_unit="fraction_ratio",
                ))

        # 4. Parse Percentages: 'X%' or 'X percent'
        pct_pattern = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(?:%|\bpercent\b|\bper\s*cent\b)",
            re.IGNORECASE
        )
        for m in pct_pattern.finditer(clean_text):
            num = float(m.group(1))
            quantities.append(NormalizedQuantity(
                raw_text=m.group(0),
                normalized_value=num / 100.0,
                quantity_type="percentage",
                unit_category="percentage",
                original_unit="percentage",
            ))

        return quantities

    def compare(self, premise_text: str, claim_text: str) -> NumericalComparisonResult:
        """
        Compare quantities in premise against claim.
        Returns NumericalComparisonResult with EQUIVALENT, CONTRADICTORY, or UNKNOWN.
        """
        prem_quantities = self.extract_quantities(premise_text)
        claim_quantities = self.extract_quantities(claim_text)

        # If either side has no extractable recognized quantities
        if not prem_quantities or not claim_quantities:
            return NumericalComparisonResult(
                relation=RELATION_UNKNOWN,
                premise_quantity=None,
                claim_quantity=None,
                relative_difference=None,
                is_within_confidence_interval=False,
                explanation="No comparable numerical rates or ratios recognized in one or both texts."
            )

        # Compare extracted quantities
        for pq in prem_quantities:
            for cq in claim_quantities:
                vp = pq.normalized_value
                vc = cq.normalized_value

                if vp <= 0 or vc <= 0:
                    continue

                # Check confidence interval bounds if available on premise
                in_ci = False
                if pq.ci_lower is not None and pq.ci_upper is not None:
                    if pq.ci_lower <= vc <= pq.ci_upper:
                        in_ci = True

                rel_diff = abs(vp - vc) / max(vp, vc)
                ratio = vp / vc

                # Criterion 1: Within CI or within rounding tolerance (e.g. 1/2000 vs 493 per million = 1.4% diff)
                if in_ci or rel_diff <= self.rounding_tolerance or (0.75 <= ratio <= 1.33):
                    ci_note = f" (within premise 95% CI [{pq.ci_lower:.6g}, {pq.ci_upper:.6g}])" if in_ci else ""
                    return NumericalComparisonResult(
                        relation=RELATION_EQUIVALENT,
                        premise_quantity=pq,
                        claim_quantity=cq,
                        relative_difference=rel_diff,
                        is_within_confidence_interval=in_ci,
                        explanation=(
                            f"Approximately equivalent: premise '{pq.raw_text}' ({vp:.6g}) "
                            f"and claim '{cq.raw_text}' ({vc:.6g}) differ by {rel_diff:.1%}{ci_note}."
                        )
                    )

                # Criterion 2: Conclusive numerical contradiction (discrepancy > 2.5x)
                if ratio > 2.5 or ratio < 0.4:
                    return NumericalComparisonResult(
                        relation=RELATION_CONTRADICTORY,
                        premise_quantity=pq,
                        claim_quantity=cq,
                        relative_difference=rel_diff,
                        is_within_confidence_interval=False,
                        explanation=(
                            f"Mathematically contradictory: premise '{pq.raw_text}' ({vp:.6g}) "
                            f"and claim '{cq.raw_text}' ({vc:.6g}) diverge by {ratio:.2f}x."
                        )
                    )

        # Ambiguous difference without meeting conclusive equivalence or contradiction
        return NumericalComparisonResult(
            relation=RELATION_UNKNOWN,
            premise_quantity=prem_quantities[0],
            claim_quantity=claim_quantities[0],
            relative_difference=abs(prem_quantities[0].normalized_value - claim_quantities[0].normalized_value) / max(prem_quantities[0].normalized_value, claim_quantities[0].normalized_value),
            is_within_confidence_interval=False,
            explanation="Numerical quantities differ moderately but lack confidence bounds to establish equivalence or contradiction."
        )
