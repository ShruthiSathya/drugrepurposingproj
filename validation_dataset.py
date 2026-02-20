"""
Curated Validation Dataset — v3 (n=33)
=======================================
This file defines the ground-truth test cases for evaluating the drug
repurposing pipeline.

CHANGE LOG
----------
v1 (n=31): Initial dataset.
v2 (n=34): Added 3 new cases (sildenafil/PAH, metformin/PCOS, colchicine/pericarditis).
v3 (n=33): CHANGE — tocilizumab/cytokine release syndrome moved to OUT_OF_SCOPE.
           Reason: CRS is a syndrome, not a disease with a stable EFO ontology
           entry. OpenTargets does not return consistent gene-level data for CRS
           across API versions, making scoring non-deterministic. The case is
           documented in OUT_OF_SCOPE_CASES below for transparency.

           FIX NOTE: A previous README and validation_results.json header
           incorrectly reported n=34. The correct count is n=33. See:
           - README.md (updated)
           - validation_results.json header: n_test_cases = 33

API COMPATIBILITY NOTE
----------------------
All disease names in this dataset were verified against the OpenTargets
EFO ontology (efo.obolibrary.org) as of July 2025. If the API returns
no results for a disease name, first check DISEASE_ALIASES in data_fetcher.py
before editing here.
"""

from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATASET_VERSION = "v3"
N_TEST_CASES    = 33   # CORRECTED from 34 — tocilizumab/CRS moved to OUT_OF_SCOPE


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _case(
    drug: str,
    disease: str,
    status: str,
    mechanism: str,
    min_score: float,
    expected_rank_top_n: int,
    sources: List[str],
    notes: str = "",
) -> Dict:
    """
    Build a test-case dict.

    Parameters
    ----------
    drug : str
        Drug common name (must match ChEMBL preferred name or DGIdb name).
    disease : str
        Disease name (must match OpenTargets EFO canonical name).
    status : str
        'TRUE_POSITIVE'  — known repurposing, should score >= min_score
        'TRUE_NEGATIVE'  — not repurposed, should score < 0.20
        'FALSE_POSITIVE_RISK' — no known repurposing but may score high (control)
    mechanism : str
        Biological rationale for why this drug should (or should not) score.
    min_score : float
        For TRUE_POSITIVE cases: minimum expected score for pipeline to pass.
        For TRUE_NEGATIVE cases: maximum expected score (should be below this).
    expected_rank_top_n : int
        For TRUE_POSITIVE: drug should appear in top N candidates.
        For TRUE_NEGATIVE: drug should NOT appear in top N candidates.
        Set to 0 if rank expectation is not applicable.
    sources : list of str
        PMIDs or DOIs supporting the ground truth.
    notes : str
        Additional context, edge cases, or caveats.
    """
    return {
        "drug":              drug,
        "disease":           disease,
        "status":            status,
        "mechanism":         mechanism,
        "min_score":         min_score,
        "expected_rank_top_n": expected_rank_top_n,
        "sources":           sources,
        "notes":             notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main test suite — n=33
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_CASES: List[Dict] = [

    # ── Neurology / Psychiatry ────────────────────────────────────────────────
    _case(
        drug="sildenafil",
        disease="pulmonary arterial hypertension",
        status="TRUE_POSITIVE",
        mechanism="PDE5 inhibition → cGMP elevation → vasodilation; shared VEGF/NO pathway genes",
        min_score=0.55,
        expected_rank_top_n=10,
        sources=["PMID:15951850", "PMID:17065669"],
        notes="FDA-approved for PAH. Strong shared pathway signal expected.",
    ),
    _case(
        drug="metformin",
        disease="type 2 diabetes mellitus",
        status="TRUE_POSITIVE",
        mechanism="AMPK activation → glucose uptake; primary indication, used as positive control",
        min_score=0.60,
        expected_rank_top_n=5,
        sources=["PMID:9742976"],
        notes="Primary indication — highest expected score. Confirms scoring floor.",
    ),
    _case(
        drug="metformin",
        disease="polycystic ovary syndrome",
        status="TRUE_POSITIVE",
        mechanism="Insulin sensitisation; shared INSR/AMPK pathway genes with PCOS",
        min_score=0.40,
        expected_rank_top_n=15,
        sources=["PMID:12849456", "PMID:23044908"],
        notes="Off-label use well-supported by RCTs. Added in v2.",
    ),
    _case(
        drug="thalidomide",
        disease="multiple myeloma",
        status="TRUE_POSITIVE",
        mechanism="Anti-angiogenic; TNF-alpha suppression; shared VEGF/IKZF pathway",
        min_score=0.45,
        expected_rank_top_n=20,
        sources=["PMID:10432031"],
        notes="FDA-approved after repurposing. Classic case.",
    ),
    _case(
        drug="imatinib",
        disease="chronic myelogenous leukemia",
        status="TRUE_POSITIVE",
        mechanism="BCR-ABL1 kinase inhibition; primary indication — positive control",
        min_score=0.65,
        expected_rank_top_n=5,
        sources=["PMID:11287977"],
        notes="Paradigmatic targeted therapy. Strongest expected shared gene signal.",
    ),
    _case(
        drug="imatinib",
        disease="gastrointestinal stromal tumor",
        status="TRUE_POSITIVE",
        mechanism="KIT/PDGFRA inhibition; shared kinase pathway with CML imatinib targets",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:11547741"],
        notes="FDA-approved repurposing from CML.",
    ),
    _case(
        drug="rituximab",
        disease="rheumatoid arthritis",
        status="TRUE_POSITIVE",
        mechanism="CD20+ B-cell depletion; shared inflammatory pathway genes (TNF, IL6)",
        min_score=0.40,
        expected_rank_top_n=20,
        sources=["PMID:15022347"],
        notes="FDA-approved for RA after initial NHL indication.",
    ),
    _case(
        drug="finasteride",
        disease="benign prostatic hyperplasia",
        status="TRUE_POSITIVE",
        mechanism="5-alpha-reductase inhibition; shared SRD5A1/SRD5A2 pathway",
        min_score=0.50,
        expected_rank_top_n=10,
        sources=["PMID:1279543"],
        notes="Primary indication — positive control.",
    ),
    _case(
        drug="colchicine",
        disease="gout",
        status="TRUE_POSITIVE",
        mechanism="Microtubule polymerisation inhibition; primary indication",
        min_score=0.55,
        expected_rank_top_n=10,
        sources=["PMID:19252078"],
        notes="Primary indication — positive control.",
    ),
    _case(
        drug="colchicine",
        disease="pericarditis",
        status="TRUE_POSITIVE",
        mechanism="NLRP3 inflammasome inhibition; shared IL1B pathway genes",
        min_score=0.35,
        expected_rank_top_n=20,
        sources=["PMID:23992557"],
        notes="FDA-approved for recurrent pericarditis. Added in v2.",
    ),
    _case(
        drug="aspirin",
        disease="colorectal cancer",
        status="TRUE_POSITIVE",
        mechanism="COX-2/PTGS2 inhibition → prostaglandin reduction → anti-proliferative",
        min_score=0.30,
        expected_rank_top_n=25,
        sources=["PMID:21562262", "PMID:26222032"],
        notes="Strong epidemiological and trial evidence. Score may be moderate "
              "because CRC gene associations are broad.",
    ),
    _case(
        drug="propranolol",
        disease="infantile hemangioma",
        status="TRUE_POSITIVE",
        mechanism="Beta-adrenergic blockade → anti-angiogenic; VEGF pathway shared",
        min_score=0.30,
        expected_rank_top_n=30,
        sources=["PMID:18940929"],
        notes="FDA breakthrough therapy. Score may be moderate — disease genes "
              "sparse in OpenTargets for rare pediatric conditions.",
    ),
    _case(
        drug="sirolimus",
        disease="tuberous sclerosis",
        status="TRUE_POSITIVE",
        mechanism="mTOR pathway inhibition; TSC1/TSC2 direct targets of mTOR signaling",
        min_score=0.50,
        expected_rank_top_n=15,
        sources=["PMID:23561269"],
        notes="Strong mechanistic signal expected via mTOR pathway overlap.",
    ),
    _case(
        drug="hydroxychloroquine",
        disease="systemic lupus erythematosus",
        status="TRUE_POSITIVE",
        mechanism="TLR9 inhibition, autophagy modulation; shared interferonopathy pathway",
        min_score=0.40,
        expected_rank_top_n=15,
        sources=["PMID:14693966"],
        notes="Standard of care for SLE.",
    ),
    _case(
        drug="lenalidomide",
        disease="multiple myeloma",
        status="TRUE_POSITIVE",
        mechanism="IKZF1/IKZF3 degradation via CRBN; immunomodulatory",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:24382584"],
        notes="Thalidomide analogue. Approved for MM after MDS.",
    ),
    _case(
        drug="letrozole",
        disease="breast cancer",
        status="TRUE_POSITIVE",
        mechanism="Aromatase (CYP19A1) inhibition → estrogen reduction",
        min_score=0.50,
        expected_rank_top_n=10,
        sources=["PMID:9470415"],
        notes="Primary indication for hormone-receptor-positive breast cancer.",
    ),
    _case(
        drug="gabapentin",
        disease="epilepsy",
        status="TRUE_POSITIVE",
        mechanism="CACNA2D1 subunit inhibition → reduced calcium influx → neuronal stabilisation",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:8602533"],
        notes="DGIdb CACNA2D1 data may be missing; pipeline falls back to "
              "KNOWN_BIOLOGIC_TARGETS for this drug (documented in data_fetcher.py). "
              "Test verifies that fallback does not silently fail.",
    ),
    _case(
        drug="olaparib",
        disease="ovarian cancer",
        status="TRUE_POSITIVE",
        mechanism="PARP1/2 inhibition → synthetic lethality in BRCA1/2-mutant cells",
        min_score=0.55,
        expected_rank_top_n=10,
        sources=["PMID:24882576"],
        notes="Strong BRCA1/BRCA2 shared gene signal expected.",
    ),
    _case(
        drug="nivolumab",
        disease="non-small cell lung carcinoma",
        status="TRUE_POSITIVE",
        mechanism="PD-1 checkpoint blockade; shared PDCD1 / CD274 pathway",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:25359623"],
        notes="First indication was melanoma; NSCLC is a repurposing.",
    ),
    _case(
        drug="pembrolizumab",
        disease="melanoma",
        status="TRUE_POSITIVE",
        mechanism="PD-1 checkpoint blockade; shared PDCD1 / CD274 / TMB pathway",
        min_score=0.50,
        expected_rank_top_n=10,
        sources=["PMID:25265492"],
        notes="Primary indication — positive control for checkpoint inhibitor scoring.",
    ),

    # ── Rare / Orphan diseases ────────────────────────────────────────────────
    _case(
        drug="eculizumab",
        disease="paroxysmal nocturnal hemoglobinuria",
        status="TRUE_POSITIVE",
        mechanism="C5 complement inhibition; shared complement pathway (C5, CD59) genes",
        min_score=0.40,
        expected_rank_top_n=20,
        sources=["PMID:18768943"],
        notes="Rare disease — gene count in OpenTargets may be low. "
              "Score may be at lower bound.",
    ),
    _case(
        drug="ivacaftor",
        disease="cystic fibrosis",
        status="TRUE_POSITIVE",
        mechanism="CFTR potentiation (G551D gating mutation); direct CFTR shared gene",
        min_score=0.50,
        expected_rank_top_n=10,
        sources=["PMID:23989293"],
        notes="Strong single-gene signal (CFTR). Should score well.",
    ),
    _case(
        drug="nusinersen",
        disease="spinal muscular atrophy",
        status="TRUE_POSITIVE",
        mechanism="Antisense oligonucleotide promoting SMN2 exon 7 inclusion",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:28056917"],
        notes="Strong SMN1/SMN2 gene signal expected.",
    ),

    # ── Metabolic / Other ─────────────────────────────────────────────────────
    _case(
        drug="ezetimibe",
        disease="hypercholesterolemia",
        status="TRUE_POSITIVE",
        mechanism="NPC1L1 inhibition → reduced cholesterol absorption",
        min_score=0.50,
        expected_rank_top_n=10,
        sources=["PMID:14523140"],
        notes="Primary indication.",
    ),
    _case(
        drug="empagliflozin",
        disease="heart failure",
        status="TRUE_POSITIVE",
        mechanism="SGLT2 inhibition → diuresis, cardiac preload reduction; "
                  "shared NPPA/NPPB pathway genes",
        min_score=0.35,
        expected_rank_top_n=20,
        sources=["PMID:33093160"],
        notes="Repurposed from T2DM. Added in v2 as recent high-impact repurposing.",
    ),

    # ── Negative controls ─────────────────────────────────────────────────────
    _case(
        drug="insulin glargine",
        disease="multiple sclerosis",
        status="TRUE_NEGATIVE",
        mechanism="No known mechanistic link. MS is autoimmune/neurological; "
                  "insulin targets INSR in metabolic context.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low. Flags if scoring is too permissive.",
    ),
    _case(
        drug="warfarin",
        disease="Parkinson disease",
        status="TRUE_NEGATIVE",
        mechanism="Warfarin is an anticoagulant targeting VKORC1/CYP2C9; "
                  "no mechanistic connection to PD neurodegeneration.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low.",
    ),
    _case(
        drug="omeprazole",
        disease="schizophrenia",
        status="TRUE_NEGATIVE",
        mechanism="Omeprazole is a PPI (H+/K+ ATPase inhibitor); "
                  "no known dopamine/glutamate pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low.",
    ),
    _case(
        drug="atorvastatin",
        disease="hemophilia A",
        status="TRUE_NEGATIVE",
        mechanism="Atorvastatin targets HMGCR in cholesterol metabolism; "
                  "hemophilia A is a monogenic coagulation factor VIII deficiency.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low. Tests against rare monogenic disease.",
    ),
    _case(
        drug="lisinopril",
        disease="anorexia nervosa",
        status="TRUE_NEGATIVE",
        mechanism="ACE inhibitor; no known eating disorder pathway connection.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low.",
    ),
    _case(
        drug="levothyroxine",
        disease="inflammatory bowel disease",
        status="TRUE_NEGATIVE",
        mechanism="Thyroid hormone replacement. No IBD mechanistic link.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low.",
    ),
    _case(
        drug="amoxicillin",
        disease="hypertrophic cardiomyopathy",
        status="TRUE_NEGATIVE",
        mechanism="Beta-lactam antibiotic. HCM is a genetic sarcomere disorder; "
                  "no pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low.",
    ),
    _case(
        drug="donepezil",
        disease="amyotrophic lateral sclerosis",
        status="TRUE_NEGATIVE",
        mechanism="AChE inhibitor; ALS involves motor neuron degeneration via "
                  "TDP-43/SOD1 pathways, not cholinergic deficit.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="ALS and AD share some neurodegeneration genes. This tests "
              "whether the pipeline over-generalises across neurological diseases.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# OUT_OF_SCOPE cases — excluded from test suite with explanation
# ─────────────────────────────────────────────────────────────────────────────
OUT_OF_SCOPE_CASES: List[Dict] = [
    {
        "drug":    "tocilizumab",
        "disease": "cytokine release syndrome",
        "reason":  (
            "CRS is a syndrome, not a discrete disease with a stable EFO ontology "
            "entry. OpenTargets returns non-deterministic gene sets for CRS across "
            "API versions, making score comparison across pipeline runs unreliable. "
            "Tocilizumab/CRS was included in v2 (n=34) but is excluded in v3 (n=33) "
            "to maintain reproducibility. The IL6 pathway rationale is valid and "
            "should be included in the Methods section discussion of IL-6 signalling."
        ),
        "removed_in_version": "v3",
        "mechanism": "IL-6 receptor blockade; IL6R/IL6ST shared genes",
        "sources":   ["PMID:32094299"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience accessors
# ─────────────────────────────────────────────────────────────────────────────
def get_positive_cases() -> List[Dict]:
    """Return TRUE_POSITIVE test cases only."""
    return [c for c in VALIDATION_CASES if c["status"] == "TRUE_POSITIVE"]


def get_negative_cases() -> List[Dict]:
    """Return TRUE_NEGATIVE test cases only."""
    return [c for c in VALIDATION_CASES if c["status"] == "TRUE_NEGATIVE"]


def get_cases_for_disease(disease_name: str) -> List[Dict]:
    """Return all test cases for a given disease (case-insensitive)."""
    name_lower = disease_name.lower()
    return [c for c in VALIDATION_CASES if name_lower in c["disease"].lower()]


def assert_n_cases():
    """Assert that the dataset has the expected number of cases. Call in CI."""
    n = len(VALIDATION_CASES)
    n_pos = len(get_positive_cases())
    n_neg = len(get_negative_cases())
    assert n == N_TEST_CASES, (
        f"Dataset has {n} cases but N_TEST_CASES={N_TEST_CASES}. "
        f"Update N_TEST_CASES or add/remove cases."
    )
    print(f"Dataset OK: {n} total ({n_pos} positive, {n_neg} negative)")
    return n, n_pos, n_neg


if __name__ == "__main__":
    n, n_pos, n_neg = assert_n_cases()
    print(f"\nValidation Dataset {DATASET_VERSION}")
    print(f"  Total cases: {n}")
    print(f"  Positive:    {n_pos}")
    print(f"  Negative:    {n_neg}")
    print(f"\nOut-of-scope cases: {len(OUT_OF_SCOPE_CASES)}")
    for case in OUT_OF_SCOPE_CASES:
        print(f"  - {case['drug']} / {case['disease']} (removed {case['removed_in_version']})")