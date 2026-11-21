"""
Scientific Entity Grounding Filter for ResearchMap
Phase 2 Isolated Experimental Implementation

Prevents irrelevant evidence from different organisms, diseases, interventions,
populations, or scientific contexts from being passed to the NLI cross-encoder.
"""

import re
from typing import Dict, List, Set, Any, Optional, Tuple


class GroundingResult:
    """Structured audit result for an entity grounding evaluation."""

    def __init__(
        self,
        status: str,  # "ACCEPTED", "REJECTED", "FLAGGED_UNCERTAIN"
        is_accepted: bool,
        rejection_reasons: List[str],
        matches: List[str],
        uncertainties: List[str],
        claim_entities: Dict[str, List[str]],
        evidence_entities: Dict[str, List[str]],
        sentence_entities: Dict[str, List[str]],
    ):
        self.status = status
        self.is_accepted = is_accepted
        self.rejection_reasons = rejection_reasons
        self.matches = matches
        self.uncertainties = uncertainties
        self.claim_entities = claim_entities
        self.evidence_entities = evidence_entities
        self.sentence_entities = sentence_entities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "is_accepted": self.is_accepted,
            "rejection_reasons": self.rejection_reasons,
            "matches": self.matches,
            "uncertainties": self.uncertainties,
            "claim_entities": self.claim_entities,
            "evidence_entities": self.evidence_entities,
            "sentence_entities": self.sentence_entities,
        }


class ScientificEntityGroundingFilter:
    """
    Isolated experimental filter comparing claim and evidence entities across four core categories:
    1. Organisms and model systems (taxonomic compatibility)
    2. Diseases and pathological conditions (clinical/pathological alignment)
    3. Interventions, pharmaceuticals, and biochemicals (mechanism/agent alignment)
    4. Populations and developmental stages (demographic/developmental scope)

    Key Principles:
    - Definite Mismatch: Rejected when evidence explicitly studies a conflicting, mutually exclusive entity.
    - Contextual Grounding: Evaluates document title and full abstract so evidence sentences with implicit subjects are NOT discarded.
    - Uncertainty Preservation: Preserves potentially relevant evidence when extraction is ambiguous or unmentioned.
    - Detailed Audit Logging: Records reasons for every accepted, rejected, or flagged sentence.
    """

    ORGANISM_TAXONOMY = {
        "saccharomyces_cerevisiae": {
            "name": "Saccharomyces cerevisiae (Yeast)",
            "group": "fungi",
            "patterns": [
                r"\b(?:s\.\s*cerevisiae|saccharomyces\s+cerevisiae|baker'?s\s+yeast|budding\s+yeast)\b",
                r"\byeast\b",
            ],
            "synonyms": ["saccharomyces cerevisiae", "yeast", "budding yeast", "baker's yeast"],
            "conflicts_with_groups": ["mammal", "rodent", "human", "bacteria", "virus", "insect", "nematode", "fish", "plant"],
        },
        "escherichia_coli": {
            "name": "Escherichia coli",
            "group": "bacteria",
            "patterns": [r"\b(?:e\.\s*coli|escherichia\s+coli)\b"],
            "synonyms": ["escherichia coli", "e. coli"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "virus", "plant", "insect", "nematode", "fish"],
        },
        "staphylococcus_aureus": {
            "name": "Staphylococcus aureus",
            "group": "bacteria",
            "patterns": [r"\b(?:s\.\s*aureus|staphylococcus\s+aureus)\b"],
            "synonyms": ["staphylococcus aureus", "s. aureus"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "virus", "plant", "insect", "nematode", "fish"],
        },
        "mycobacterium_tuberculosis": {
            "name": "Mycobacterium tuberculosis",
            "group": "bacteria",
            "patterns": [r"\b(?:m\.\s*tuberculosis|mycobacterium\s+tuberculosis|tubercle\s+bacillus)\b"],
            "synonyms": ["mycobacterium tuberculosis", "m. tuberculosis"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "virus", "plant", "insect", "nematode", "fish"],
        },
        "clostridium_difficile": {
            "name": "Clostridium difficile",
            "group": "bacteria",
            "patterns": [r"\b(?:c\.\s*difficile|clostridium\s+difficile)\b"],
            "synonyms": ["clostridium difficile", "c. difficile"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "virus", "plant", "insect", "nematode", "fish"],
        },
        "methanosphaera_stadtmanae": {
            "name": "Methanosphaera stadtmanae",
            "group": "archaea",
            "patterns": [r"\b(?:m\.\s*stadtmanae|methanosphaera\s+stadtmanae)\b"],
            "synonyms": ["methanosphaera stadtmanae", "m. stadtmanae"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "plant"],
        },
        "caenorhabditis_elegans": {
            "name": "Caenorhabditis elegans",
            "group": "nematode",
            "patterns": [r"\b(?:c\.\s*elegans|caenorhabditis\s+elegans|nematodes?|roundworms?)\b"],
            "synonyms": ["caenorhabditis elegans", "c. elegans", "nematode"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "bacteria", "virus", "plant", "fish"],
        },
        "drosophila_melanogaster": {
            "name": "Drosophila melanogaster",
            "group": "insect",
            "patterns": [r"\b(?:drosophila(?:\s+melanogaster)?|fruit\s+flies|fruit\s+fly)\b"],
            "synonyms": ["drosophila", "fruit fly"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "bacteria", "virus", "plant", "fish"],
        },
        "danio_rerio": {
            "name": "Danio rerio (Zebrafish)",
            "group": "fish",
            "patterns": [r"\b(?:danio\s+rerio|zebrafish)\b"],
            "synonyms": ["danio rerio", "zebrafish"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "bacteria", "virus", "plant", "insect", "nematode"],
        },
        "mus_musculus": {
            "name": "Mus musculus (Mouse)",
            "group": "rodent",
            "patterns": [r"\b(?:mice|mouse|murine|mus\s+musculus)\b"],
            "synonyms": ["mouse", "mice", "murine", "mus musculus"],
            "conflicts_with_groups": ["fungi", "bacteria", "virus", "plant", "insect", "nematode", "fish"],
        },
        "rattus_norvegicus": {
            "name": "Rattus norvegicus (Rat)",
            "group": "rodent",
            "patterns": [r"\b(?:rats?|rattus(?:\s+norvegicus)?|fischer\s+344)\b"],
            "synonyms": ["rat", "rats", "rattus norvegicus"],
            "conflicts_with_groups": ["fungi", "bacteria", "virus", "plant", "insect", "nematode", "fish"],
        },
        "homo_sapiens": {
            "name": "Homo sapiens (Human)",
            "group": "human",
            "patterns": [
                r"\b(?:humans?|homo\s+sapiens|patients?|men|women|physicians?|clinical\s+subjects?|participants?)\b"
            ],
            "synonyms": ["human", "humans", "patient", "patients", "homo sapiens"],
            "conflicts_with_groups": ["fungi", "bacteria", "plant", "insect", "nematode", "fish"],
        },
        "bacteria_general": {
            "name": "Bacteria (General)",
            "group": "bacteria",
            "patterns": [r"\b(?:bacteria|bacterium|bacterial|prokaryot(?:es?|ic))\b"],
            "synonyms": ["bacteria", "bacterial"],
            "conflicts_with_groups": ["fungi", "virus", "plant"],
        },
        "virus_general": {
            "name": "Virus (General)",
            "group": "virus",
            "patterns": [r"\b(?:vir(?:us(?:es)?|al)|influenza|retrovir(?:us|al)|oncolytic\s+virus)\b"],
            "synonyms": ["virus", "viral"],
            "conflicts_with_groups": ["fungi", "bacteria", "plant"],
        },
        "plant_general": {
            "name": "Plants (General)",
            "group": "plant",
            "patterns": [r"\b(?:plants?|botanical|vegetation)\b"],
            "synonyms": ["plant", "botanical"],
            "conflicts_with_groups": ["fungi", "mammal", "rodent", "human", "bacteria", "virus", "fish", "insect"],
        },
    }

    DISEASE_TAXONOMY = {
        "breast_cancer": {
            "name": "Breast Cancer",
            "group": "cancer",
            "patterns": [r"\bbreast\s+(?:cancer|carcinoma|neoplasm|tumor|tumour)s?\b"],
            "conflicts_with": ["prostate_cancer", "lung_cancer", "colorectal_cancer", "ovarian_cancer", "liver_cancer"],
        },
        "prostate_cancer": {
            "name": "Prostate Cancer",
            "group": "cancer",
            "patterns": [r"\bprostate\s+(?:cancer|carcinoma|neoplasm|tumor|tumour)s?\b"],
            "conflicts_with": ["breast_cancer", "lung_cancer", "colorectal_cancer", "ovarian_cancer", "liver_cancer"],
        },
        "lung_cancer": {
            "name": "Lung Cancer",
            "group": "cancer",
            "patterns": [
                r"\blung\s+(?:cancer|carcinoma|neoplasm|neoplastic|tumor|tumour)s?\b",
                r"\bnon-small\s+cell\s+lung\s+cancer\b",
                r"\bnsclc\b",
                r"\bmetastatic\s+lung\s+carcinoma\b",
            ],
            "conflicts_with": ["prostate_cancer", "breast_cancer", "colorectal_cancer", "ovarian_cancer"],
        },
        "liver_cancer": {
            "name": "Liver Cancer / Hepatocellular Carcinoma",
            "group": "cancer",
            "patterns": [r"\b(?:hepatocellular\s+carcinoma|liver\s+cancer|hcc|hepatic\s+carcinoma)\b"],
            "conflicts_with": ["prostate_cancer", "breast_cancer", "colorectal_cancer", "lung_cancer"],
        },
        "ovarian_cancer": {
            "name": "Ovarian Cancer",
            "group": "cancer",
            "patterns": [r"\bovarian\s+(?:cancer|carcinoma|neoplasm|tumor|tumour)s?\b"],
            "conflicts_with": ["prostate_cancer", "breast_cancer", "lung_cancer", "colorectal_cancer"],
        },
        "colorectal_cancer": {
            "name": "Colorectal Cancer",
            "group": "cancer",
            "patterns": [r"\b(?:colorectal|colon|rectal)\s+(?:cancer|carcinoma|neoplasm|tumor|tumour)s?\b"],
            "conflicts_with": ["prostate_cancer", "breast_cancer", "lung_cancer", "ovarian_cancer"],
        },
        "lymphoma": {
            "name": "Lymphoma / Hodgkin's Disease",
            "group": "cancer",
            "patterns": [r"\b(?:hodgkin(?:'s)?\s+)?lymphoma\b", r"\bfollicular\s+lymphoma\b"],
            "conflicts_with": ["prostate_cancer", "breast_cancer", "lung_cancer"],
        },
        "leukemia": {
            "name": "Leukemia",
            "group": "cancer",
            "patterns": [r"\bleuk[ae]mia\b"],
            "conflicts_with": ["prostate_cancer", "breast_cancer", "lung_cancer"],
        },
        "cancer_general": {
            "name": "Cancer (General Neoplasm)",
            "group": "cancer",
            "patterns": [r"\b(?:cancer|carcinoma|neoplasm|neoplasms|neoplastic|tumors?|tumours?|malignan(?:cy|cies))\b"],
            "conflicts_with": [],  # Compatible with all cancer subtypes
        },
        "obesity": {
            "name": "Obesity and Gestational Weight Gain",
            "group": "metabolic",
            "patterns": [r"\b(?:obesity|obese|gestational\s+weight\s+gain|overweight)\b"],
            "conflicts_with": ["anxiety", "panic", "alzheimers", "epilepsy", "asthma"],
        },
        "type_2_diabetes": {
            "name": "Type 2 Diabetes",
            "group": "metabolic",
            "patterns": [r"\b(?:type\s+2\s+diabetes|t2d|diabetes\s+mellitus|diabetic)\b"],
            "conflicts_with": ["anxiety", "panic", "asthma"],
        },
        "macrosomia": {
            "name": "Macrosomia",
            "group": "metabolic",
            "patterns": [r"\bmacrosomia\b"],
            "conflicts_with": ["anxiety", "asthma", "alzheimers"],
        },
        "alzheimers": {
            "name": "Alzheimer's Disease",
            "group": "neurodegenerative",
            "patterns": [r"\balzheimer(?:'s)?(?:\s+disease)?\b", r"\bad\b"],
            "conflicts_with": ["obesity", "asthma", "prostate_cancer", "epilepsy", "macrosomia"],
        },
        "parkinsons": {
            "name": "Parkinson's Disease",
            "group": "neurodegenerative",
            "patterns": [r"\bparkinson(?:'s)?(?:\s+disease)?\b", r"\bpd\b"],
            "conflicts_with": ["obesity", "asthma", "prostate_cancer", "epilepsy"],
        },
        "asthma": {
            "name": "Asthma",
            "group": "respiratory",
            "patterns": [r"\b(?:childhood\s+)?asthma\b"],
            "conflicts_with": ["alzheimers", "prostate_cancer", "breast_cancer", "epilepsy", "obesity"],
        },
        "anxiety": {
            "name": "Anxiety",
            "group": "psychiatric",
            "patterns": [r"\banxiety\b"],
            "conflicts_with": ["obesity", "asthma", "epilepsy"],
        },
        "panic": {
            "name": "Panic Disorder",
            "group": "psychiatric",
            "patterns": [r"\bpanic(?:prone)?\b"],
            "conflicts_with": ["obesity", "asthma", "epilepsy"],
        },
        "epilepsy": {
            "name": "Epilepsy and Status Epilepticus",
            "group": "neurological",
            "patterns": [r"\b(?:status\s+epilepticus|epilep(?:sy|tic)|seizures?)\b"],
            "conflicts_with": ["asthma", "obesity", "anxiety", "alzheimers"],
        },
        "stroke": {
            "name": "Stroke and Cerebral Ischemia",
            "group": "cardiovascular",
            "patterns": [r"\b(?:strokes?|cerebral\s+ischemia)\b"],
            "conflicts_with": ["asthma", "anxiety"],
        },
    }

    INTERVENTION_TAXONOMY = {
        "vitamin_e": {
            "name": "Vitamin E / alpha-Tocopherol",
            "patterns": [
                r"\b(?:vitamin\s+e|alpha-tocopher(?:ol|yl(?:\s+acetate)?)|α-tocopher(?:ol|yl(?:\s+acetate)?))\b"
            ],
            "synonyms": ["vitamin e", "alpha-tocopherol", "alpha-tocopheryl acetate"],
            "conflicts_with": ["valproate", "propranolol", "methionine", "dark_chocolate"],
        },
        "vitamin_c": {
            "name": "Vitamin C / Ascorbic Acid",
            "patterns": [r"\b(?:vitamin\s+c|ascorbic\s+acid|ascorbate)\b"],
            "synonyms": ["vitamin c", "ascorbic acid"],
            "conflicts_with": ["valproate", "propranolol", "methionine"],
        },
        "vitamin_d": {
            "name": "Vitamin D / Cholecalciferol",
            "patterns": [r"\b(?:vitamin\s+d3?|cholecalciferol)\b"],
            "synonyms": ["vitamin d", "cholecalciferol"],
            "conflicts_with": ["valproate", "propranolol", "methionine"],
        },
        "vitamin_b6": {
            "name": "Vitamin B6 / Pyridoxine",
            "patterns": [r"\b(?:vitamin\s+b6|pyridoxine)\b"],
            "synonyms": ["vitamin b6", "pyridoxine"],
            "conflicts_with": ["valproate", "propranolol"],
        },
        "folate": {
            "name": "Folate / Folic Acid",
            "patterns": [r"\b(?:folate|folic\s+acid|vitamin\s+b9)\b"],
            "synonyms": ["folate", "folic acid"],
            "conflicts_with": ["valproate", "propranolol"],
        },
        "glucose": {
            "name": "Glucose / Glucose Restriction",
            "patterns": [r"\b(?:glucose(?:\s+restriction)?|dextrose)\b"],
            "synonyms": ["glucose", "glucose restriction"],
            "conflicts_with": ["methionine", "valproate", "propranolol"],
        },
        "methionine": {
            "name": "Methionine",
            "patterns": [r"\b(?:methionine|l-methionine)\b"],
            "synonyms": ["methionine", "l-methionine"],
            "conflicts_with": ["glucose", "valproate", "propranolol", "vitamin_e"],
        },
        "propranolol": {
            "name": "Propranolol",
            "patterns": [r"\bpropranolol\b"],
            "synonyms": ["propranolol"],
            "conflicts_with": ["vitamin_e", "vitamin_c", "glucose", "methionine"],
        },
        "valproate": {
            "name": "Valproate",
            "patterns": [r"\b(?:valproate|valproic\s+acid)\b"],
            "synonyms": ["valproate", "valproic acid"],
            "conflicts_with": ["vitamin_e", "glucose", "methionine", "vitamin_c"],
        },
        "antibiotics": {
            "name": "Antibiotics",
            "patterns": [r"\b(?:antibiotics?|antibacterial)\b"],
            "synonyms": ["antibiotics", "antibacterial"],
            "conflicts_with": ["sunscreen", "dark_chocolate"],
        },
        "sunscreen": {
            "name": "Sunscreen",
            "patterns": [r"\bsunscreen\b"],
            "synonyms": ["sunscreen"],
            "conflicts_with": ["antibiotics", "dark_chocolate", "valproate"],
        },
        "dark_chocolate": {
            "name": "Dark Chocolate",
            "patterns": [r"\b(?:dark\s+chocolate|cocoa)\b"],
            "synonyms": ["dark chocolate", "cocoa"],
            "conflicts_with": ["antibiotics", "sunscreen", "valproate", "vitamin_e"],
        },
        "crispr_cas9": {
            "name": "CRISPR-Cas9",
            "patterns": [r"\b(?:crispr(?:-cas9)?|cas9)\b"],
            "synonyms": ["crispr", "crispr-cas9", "cas9"],
            "conflicts_with": ["dark_chocolate", "sunscreen"],
        },
    }

    POPULATION_TAXONOMY = {
        "pediatric": {
            "name": "Pediatric / Children / Infancy",
            "patterns": [r"\b(?:pediatric|children|childhood|infan(?:cy|ts?)|newborns?|neonat(?:al|es?))\b"],
            "conflicts_with": ["geriatric"],
        },
        "pregnancy": {
            "name": "Pregnancy / Maternal",
            "patterns": [r"\b(?:pregnan(?:cy|t)|maternal|gestational|first\s+trimester)\b"],
            "conflicts_with": [],
        },
        "geriatric": {
            "name": "Geriatric / Elderly",
            "patterns": [r"\b(?:elderly|geriatric|aging|aged\s+adults?)\b"],
            "conflicts_with": ["pediatric"],
        },
    }

    def __init__(self):
        self._organism_regexes = {
            k: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for k, data in self.ORGANISM_TAXONOMY.items()
        }
        self._disease_regexes = {
            k: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for k, data in self.DISEASE_TAXONOMY.items()
        }
        self._intervention_regexes = {
            k: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for k, data in self.INTERVENTION_TAXONOMY.items()
        }
        self._population_regexes = {
            k: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for k, data in self.POPULATION_TAXONOMY.items()
        }

    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract all recognized entities from text across all categories."""
        if not text:
            return {"organisms": set(), "diseases": set(), "interventions": set(), "populations": set()}

        res = {
            "organisms": set(),
            "diseases": set(),
            "interventions": set(),
            "populations": set(),
        }

        for k, regex_list in self._organism_regexes.items():
            if any(r.search(text) for r in regex_list):
                res["organisms"].add(k)

        for k, regex_list in self._disease_regexes.items():
            if any(r.search(text) for r in regex_list):
                res["diseases"].add(k)

        for k, regex_list in self._intervention_regexes.items():
            if any(r.search(text) for r in regex_list):
                res["interventions"].add(k)

        for k, regex_list in self._population_regexes.items():
            if any(r.search(text) for r in regex_list):
                res["populations"].add(k)

        return res

    def evaluate_grounding(
        self,
        claim_text: str,
        sentence_text: str,
        doc_title: str = "",
        doc_abstract: Optional[List[str]] = None,
    ) -> GroundingResult:
        """
        Evaluate whether an evidence sentence is scientifically grounded to the claim.
        Uses document title and full abstract as contextual prior.
        """
        claim_ents = self.extract_entities(claim_text)
        sent_ents = self.extract_entities(sentence_text)

        doc_full_text = f"{doc_title} {' '.join(doc_abstract or [])}".strip()
        doc_ents = self.extract_entities(doc_full_text)

        # Context entities combine document-level context with sentence-level entities
        context_ents = {
            cat: sent_ents[cat] | doc_ents[cat]
            for cat in ["organisms", "diseases", "interventions", "populations"]
        }

        rejection_reasons = []
        matches = []
        uncertainties = []

        # 1. Organism Evaluation
        claim_orgs = claim_ents["organisms"]
        context_orgs = context_ents["organisms"]

        if claim_orgs:
            common_orgs = claim_orgs & context_orgs
            if common_orgs:
                matches.append(f"Organism match: {sorted(list(common_orgs))}")
            else:
                conflicts = []
                for c_org in claim_orgs:
                    c_group = self.ORGANISM_TAXONOMY[c_org]["group"]
                    c_conflicts = self.ORGANISM_TAXONOMY[c_org]["conflicts_with_groups"]
                    for ev_org in context_orgs:
                        ev_group = self.ORGANISM_TAXONOMY[ev_org]["group"]
                        if ev_group in c_conflicts:
                            conflicts.append((c_org, ev_org))
                if conflicts:
                    rejection_reasons.append(
                        f"Definite organism mismatch: claim requires {sorted(list(claim_orgs))}, "
                        f"evidence context studies conflicting organism(s) {[c[1] for c in conflicts]}"
                    )
                else:
                    uncertainties.append(f"Claim organism {sorted(list(claim_orgs))} not explicitly found in evidence.")

        # 2. Disease Evaluation
        claim_diseases = claim_ents["diseases"]
        context_diseases = context_ents["diseases"]

        if claim_diseases:
            # Check for specific conflicting subtypes first
            conflicts = []
            for c_dis in claim_diseases:
                c_conflicts = self.DISEASE_TAXONOMY[c_dis].get("conflicts_with", [])
                for ev_dis in context_diseases:
                    if ev_dis in c_conflicts and c_dis not in context_diseases:
                        conflicts.append((c_dis, ev_dis))

            if conflicts:
                rejection_reasons.append(
                    f"Definite disease mismatch: claim specifies {sorted(list(claim_diseases))}, "
                    f"evidence context studies conflicting disease(s) {[c[1] for c in conflicts]}"
                )
            else:
                # Check for exact matches excluding pure general cancer if specific exists
                specific_claim_dis = claim_diseases - {"cancer_general"}
                specific_context_dis = context_diseases - {"cancer_general"}

                if specific_claim_dis and specific_context_dis:
                    common_specific = specific_claim_dis & specific_context_dis
                    if common_specific:
                        matches.append(f"Disease match: {sorted(list(common_specific))}")
                    else:
                        uncertainties.append(f"Claim disease {sorted(list(specific_claim_dis))} not matched to {sorted(list(specific_context_dis))}")
                elif "cancer_general" in claim_diseases or "cancer_general" in context_diseases:
                    matches.append("Disease compatible: generic cancer alignment")
                else:
                    uncertainties.append(f"Claim disease {sorted(list(claim_diseases))} not explicitly found.")

        # 3. Intervention Evaluation
        claim_interventions = claim_ents["interventions"]
        context_interventions = context_ents["interventions"]

        if claim_interventions:
            common_int = claim_interventions & context_interventions
            if common_int:
                matches.append(f"Intervention match: {sorted(list(common_int))}")
            else:
                conflicts = []
                for c_int in claim_interventions:
                    c_conflicts = self.INTERVENTION_TAXONOMY[c_int]["conflicts_with"]
                    for ev_int in context_interventions:
                        if ev_int in c_conflicts:
                            conflicts.append((c_int, ev_int))
                if conflicts:
                    rejection_reasons.append(
                        f"Definite intervention mismatch: claim specifies {sorted(list(claim_interventions))}, "
                        f"evidence context studies conflicting intervention(s) {[c[1] for c in conflicts]}"
                    )
                else:
                    uncertainties.append(f"Claim intervention {sorted(list(claim_interventions))} not explicitly found.")

        # 4. Population Evaluation
        claim_pop = claim_ents["populations"]
        context_pop = context_ents["populations"]

        if claim_pop:
            common_pop = claim_pop & context_pop
            if common_pop:
                matches.append(f"Population match: {sorted(list(common_pop))}")
            else:
                conflicts = []
                for c_p in claim_pop:
                    c_conflicts = self.POPULATION_TAXONOMY[c_p]["conflicts_with"]
                    for ev_p in context_pop:
                        if ev_p in c_conflicts:
                            conflicts.append((c_p, ev_p))
                if conflicts:
                    rejection_reasons.append(
                        f"Definite population mismatch: claim specifies {sorted(list(claim_pop))}, "
                        f"evidence context studies conflicting population(s) {[c[1] for c in conflicts]}"
                    )
                else:
                    uncertainties.append(f"Claim population {sorted(list(claim_pop))} not explicitly found.")

        # Final Grounding Status
        if rejection_reasons:
            status = "REJECTED"
            is_accepted = False
        elif uncertainties and not matches:
            status = "FLAGGED_UNCERTAIN"
            is_accepted = True  # Rule: preserve when uncertain
        else:
            status = "ACCEPTED"
            is_accepted = True

        return GroundingResult(
            status=status,
            is_accepted=is_accepted,
            rejection_reasons=rejection_reasons,
            matches=matches,
            uncertainties=uncertainties,
            claim_entities={k: sorted(list(v)) for k, v in claim_ents.items() if v},
            evidence_entities={k: sorted(list(v)) for k, v in context_ents.items() if v},
            sentence_entities={k: sorted(list(v)) for k, v in sent_ents.items() if v},
        )

    def filter_candidate_sentences(
        self,
        claim_text: str,
        candidate_sentences: List[Dict[str, Any]],
        corpus: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter a list of retrieved candidate sentences against the claim.
        Returns:
            (accepted_sentences, rejected_sentences_with_audit)
        """
        accepted = []
        rejected = []

        for s in candidate_sentences:
            doc_id = s["doc_id"]
            doc_data = corpus.get(doc_id, {})
            title = doc_data.get("title", "")
            abstract = doc_data.get("abstract", [])

            audit = self.evaluate_grounding(
                claim_text=claim_text,
                sentence_text=s["text"],
                doc_title=title,
                doc_abstract=abstract,
            )

            annotated_s = dict(s)
            annotated_s["grounding"] = audit.to_dict()

            if audit.is_accepted:
                accepted.append(annotated_s)
            else:
                rejected.append(annotated_s)

        return accepted, rejected
