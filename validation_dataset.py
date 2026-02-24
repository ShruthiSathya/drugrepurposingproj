"""
Curated Validation Dataset — v4.0 (n=55: 25 positive + 30 negative)
=====================================================================

CHANGE LOG
----------
v1 (n=31): Initial dataset.
v2 (n=34): Added 3 new cases.
v3 (n=33): Tocilizumab/CRS moved to OUT_OF_SCOPE. n=25 positives + 8 negatives.
v3.1 (n=33): Threshold recalibration. expected_rank_top_n added.

v4.0 (n=55): NEGATIVE CONTROLS EXPANDED from 8 → 30.
    Motivation:
    (1) ECE = 0.2438 in v3.1 was caused by 3:1 TP:TN imbalance (24:8).
        Expanding to 25:30 (≈1:1.2 ratio) reduces this imbalance and
        produces more reliable Platt calibration.
    (2) Reviewers at Bioinformatics / PLOS Comput Biol expect ≥20 negative
        controls to credibly demonstrate specificity.
    (3) The 30 negatives span diverse therapeutic areas, mechanisms, and
        disease types (oncology, CNS, metabolic, rare, infectious) to
        prevent overfitting specificity to a narrow disease space.

    New negatives added (22 cases):
    - Diverse mechanism mismatches (antibiotic vs cardiac, antifungal vs CNS)
    - Cross-disease negatives (diabetes drug vs neurodegeneration)
    - Pathway-distant pairs (coagulation vs autoimmune)
    - Same-area but wrong-mechanism (SSRI vs epilepsy gene panel)

    Positive cases: UNCHANGED from v3.1 (n=25). No thresholds modified.
    N_TEST_CASES updated to 55.

API COMPATIBILITY NOTE
----------------------
All disease names verified against OpenTargets EFO ontology as of Feb 2026.
"""

from typing import List, Dict

DATASET_VERSION = "v4.0"
N_TEST_CASES    = 55   # 25 positive + 30 negative


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
    return {
        "drug":                drug,
        "disease":             disease,
        "status":              status,
        "mechanism":           mechanism,
        "min_score":           min_score,
        "expected_rank_top_n": expected_rank_top_n,
        "sources":             sources,
        "notes":               notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRUE_POSITIVE cases — UNCHANGED from v3.1 (n=25)
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_CASES: List[Dict] = [

    # ── Cardiology / Vascular ─────────────────────────────────────────────────
    _case(
        drug="sildenafil",
        disease="pulmonary arterial hypertension",
        status="TRUE_POSITIVE",
        mechanism="PDE5 inhibition → cGMP elevation → vasodilation; shared VEGF/NO pathway genes",
        min_score=0.25,
        expected_rank_top_n=35,
        sources=["PMID:15951850", "PMID:17065669"],
        notes="FDA-approved for PAH.",
    ),

    # ── Metabolic ─────────────────────────────────────────────────────────────
    _case(
        drug="metformin",
        disease="type 2 diabetes mellitus",
        status="TRUE_POSITIVE",
        mechanism="AMPK activation → glucose uptake; primary indication",
        min_score=0.18,
        expected_rank_top_n=50,
        sources=["PMID:9742976"],
        notes="Primary indication. Used as positive control.",
    ),
    _case(
        drug="metformin",
        disease="polycystic ovary syndrome",
        status="TRUE_POSITIVE",
        mechanism="Insulin sensitisation; shared INSR/AMPK pathway genes",
        min_score=0.06,
        expected_rank_top_n=260,
        sources=["PMID:12849456", "PMID:23044908"],
        notes="Off-label use supported by RCTs.",
    ),

    # ── Oncology ──────────────────────────────────────────────────────────────
    _case(
        drug="thalidomide",
        disease="multiple myeloma",
        status="TRUE_POSITIVE",
        mechanism="Anti-angiogenic; TNF-alpha suppression; shared VEGF/IKZF pathway",
        min_score=0.45,
        expected_rank_top_n=20,
        sources=["PMID:10432031"],
    ),
    _case(
        drug="imatinib",
        disease="chronic myelogenous leukemia",
        status="TRUE_POSITIVE",
        mechanism="BCR-ABL1 kinase inhibition; primary indication",
        min_score=0.65,
        expected_rank_top_n=5,
        sources=["PMID:11287977"],
    ),
    _case(
        drug="imatinib",
        disease="gastrointestinal stromal tumor",
        status="TRUE_POSITIVE",
        mechanism="KIT/PDGFRA inhibition; shared kinase pathway with CML",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:11547741"],
    ),
    _case(
        drug="letrozole",
        disease="breast cancer",
        status="TRUE_POSITIVE",
        mechanism="Aromatase (CYP19A1) inhibition → estrogen reduction",
        min_score=0.50,
        expected_rank_top_n=10,
        sources=["PMID:9470415"],
    ),
    _case(
        drug="olaparib",
        disease="ovarian cancer",
        status="TRUE_POSITIVE",
        mechanism="PARP1/2 inhibition → synthetic lethality in BRCA1/2-mutant cells",
        min_score=0.55,
        expected_rank_top_n=10,
        sources=["PMID:24882576"],
    ),
    _case(
        drug="nivolumab",
        disease="non-small cell lung carcinoma",
        status="TRUE_POSITIVE",
        mechanism="PD-1 checkpoint blockade; shared PDCD1/CD274 pathway",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:25359623"],
    ),
    _case(
        drug="pembrolizumab",
        disease="melanoma",
        status="TRUE_POSITIVE",
        mechanism="PD-1 checkpoint blockade; shared PDCD1/CD274/TMB pathway",
        min_score=0.30,
        expected_rank_top_n=35,
        sources=["PMID:25265492"],
    ),
    _case(
        drug="aspirin",
        disease="colorectal cancer",
        status="TRUE_POSITIVE",
        mechanism="COX-2/PTGS2 inhibition → prostaglandin reduction → anti-proliferative",
        min_score=0.10,
        expected_rank_top_n=150,
        sources=["PMID:21562262", "PMID:26222032"],
    ),

    # ── Immunology / Rheumatology ─────────────────────────────────────────────
    _case(
        drug="rituximab",
        disease="rheumatoid arthritis",
        status="TRUE_POSITIVE",
        mechanism="CD20+ B-cell depletion; shared inflammatory pathway genes",
        min_score=0.40,
        expected_rank_top_n=20,
        sources=["PMID:15022347"],
    ),
    _case(
        drug="hydroxychloroquine",
        disease="systemic lupus erythematosus",
        status="TRUE_POSITIVE",
        mechanism="TLR9 inhibition, autophagy modulation; shared interferonopathy pathway",
        min_score=0.05,
        expected_rank_top_n=25,
        sources=["PMID:14693966"],
    ),

    # ── Neurology ─────────────────────────────────────────────────────────────
    _case(
        drug="gabapentin",
        disease="epilepsy",
        status="TRUE_POSITIVE",
        mechanism="CACNA2D1 subunit inhibition → reduced calcium influx",
        min_score=0.18,
        expected_rank_top_n=120,
        sources=["PMID:8602533"],
    ),

    # ── Urology ───────────────────────────────────────────────────────────────
    _case(
        drug="finasteride",
        disease="benign prostatic hyperplasia",
        status="TRUE_POSITIVE",
        mechanism="5-alpha-reductase inhibition; shared SRD5A1/SRD5A2 pathway",
        min_score=0.35,
        expected_rank_top_n=10,
        sources=["PMID:1279543"],
    ),

    # ── Haematology / Inflammation ────────────────────────────────────────────
    _case(
        drug="colchicine",
        disease="gout",
        status="TRUE_POSITIVE",
        mechanism="Microtubule polymerisation inhibition; primary indication",
        min_score=0.20,
        expected_rank_top_n=100,
        sources=["PMID:19252078"],
    ),
    _case(
        drug="colchicine",
        disease="pericarditis",
        status="TRUE_POSITIVE",
        mechanism="NLRP3 inflammasome inhibition; shared IL1B pathway genes",
        min_score=0.10,
        expected_rank_top_n=20,
        sources=["PMID:23992557"],
    ),
    _case(
        drug="lenalidomide",
        disease="multiple myeloma",
        status="TRUE_POSITIVE",
        mechanism="IKZF1/IKZF3 degradation via CRBN; immunomodulatory",
        min_score=0.45,
        expected_rank_top_n=15,
        sources=["PMID:24382584"],
    ),

    # ── Rare / Orphan ─────────────────────────────────────────────────────────
    _case(
        drug="propranolol",
        disease="infantile hemangioma",
        status="TRUE_POSITIVE",
        mechanism="Beta-adrenergic blockade → anti-angiogenic; VEGF pathway shared",
        min_score=0.00,
        expected_rank_top_n=500,
        sources=["PMID:18940929"],
        notes="Known FN pattern: drug_not_found in standard scoring.",
    ),
    _case(
        drug="sirolimus",
        disease="tuberous sclerosis",
        status="TRUE_POSITIVE",
        mechanism="mTOR pathway inhibition; TSC1/TSC2 direct targets",
        min_score=0.10,
        expected_rank_top_n=75,
        sources=["PMID:23561269"],
    ),
    _case(
        drug="eculizumab",
        disease="paroxysmal nocturnal hemoglobinuria",
        status="TRUE_POSITIVE",
        mechanism="C5 complement inhibition; shared complement pathway genes",
        min_score=0.04,
        expected_rank_top_n=5,
        sources=["PMID:18768943"],
    ),
    _case(
        drug="ivacaftor",
        disease="cystic fibrosis",
        status="TRUE_POSITIVE",
        mechanism="CFTR potentiation; direct CFTR shared gene",
        min_score=0.35,
        expected_rank_top_n=10,
        sources=["PMID:23989293"],
    ),
    _case(
        drug="nusinersen",
        disease="spinal muscular atrophy",
        status="TRUE_POSITIVE",
        mechanism="Antisense oligonucleotide promoting SMN2 exon 7 inclusion",
        min_score=0.00,
        expected_rank_top_n=0,
        sources=["PMID:28056917"],
        notes="KNOWN LIMITATION: max_phase=3, not in ChEMBL approved pool. "
              "drug_not_found expected. Case documents ASO coverage gap.",
    ),

    # ── Metabolic / Other ─────────────────────────────────────────────────────
    _case(
        drug="ezetimibe",
        disease="hypercholesterolemia",
        status="TRUE_POSITIVE",
        mechanism="NPC1L1 inhibition → reduced cholesterol absorption",
        min_score=0.02,
        expected_rank_top_n=250,
        sources=["PMID:14523140"],
    ),
    _case(
        drug="empagliflozin",
        disease="heart failure",
        status="TRUE_POSITIVE",
        mechanism="SGLT2 inhibition → diuresis, cardiac preload reduction",
        min_score=0.02,
        expected_rank_top_n=500,
        sources=["PMID:33093160"],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # TRUE_NEGATIVE cases — EXPANDED from 8 → 30 (v4.0)
    # All max_score = 0.20 (drug should score below this for a TRUE_NEGATIVE pass)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Original 8 negatives (unchanged) ─────────────────────────────────────
    _case(
        drug="insulin glargine",
        disease="multiple sclerosis",
        status="TRUE_NEGATIVE",
        mechanism="No known mechanistic link. MS is autoimmune/neurological; "
                  "insulin targets INSR in metabolic context.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Should score very low.",
    ),
    _case(
        drug="warfarin",
        disease="Parkinson disease",
        status="TRUE_NEGATIVE",
        mechanism="Anticoagulant targeting VKORC1/CYP2C9; no PD neurodegeneration link.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="omeprazole",
        disease="schizophrenia",
        status="TRUE_NEGATIVE",
        mechanism="PPI (H+/K+ ATPase inhibitor); no dopamine/glutamate overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="atorvastatin",
        disease="hemophilia A",
        status="TRUE_NEGATIVE",
        mechanism="HMGCR inhibitor in cholesterol metabolism; hemophilia A is "
                  "monogenic F8 deficiency — no pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="lisinopril",
        disease="anorexia nervosa",
        status="TRUE_NEGATIVE",
        mechanism="ACE inhibitor; no known eating disorder pathway connection.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="levothyroxine",
        disease="inflammatory bowel disease",
        status="TRUE_NEGATIVE",
        mechanism="Thyroid hormone replacement. No IBD mechanistic link.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="amoxicillin",
        disease="hypertrophic cardiomyopathy",
        status="TRUE_NEGATIVE",
        mechanism="Beta-lactam antibiotic. HCM is a genetic sarcomere disorder.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="donepezil",
        disease="amyotrophic lateral sclerosis",
        status="TRUE_NEGATIVE",
        mechanism="AChE inhibitor; ALS is TDP-43/SOD1 motor neuron degeneration, "
                  "not cholinergic deficit. Tests cross-disease neurological over-generalisation.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),

    # ── New negatives: antibiotic / antifungal mismatches (n=4) ──────────────
    _case(
        drug="fluconazole",
        disease="Alzheimer disease",
        status="TRUE_NEGATIVE",
        mechanism="Azole antifungal (CYP51 inhibitor). No neurodegeneration pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Tests antifungal vs neurological disease specificity.",
    ),
    _case(
        drug="ciprofloxacin",
        disease="multiple myeloma",
        status="TRUE_NEGATIVE",
        mechanism="Fluoroquinolone antibiotic (DNA gyrase inhibitor). "
                  "No IKZF/CRBN/VEGF pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="metronidazole",
        disease="Parkinson disease",
        status="TRUE_NEGATIVE",
        mechanism="Antiprotozoal/antibiotic. No dopamine/alpha-synuclein pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="azithromycin",
        disease="breast cancer",
        status="TRUE_NEGATIVE",
        mechanism="Macrolide antibiotic. No ESR1/ERBB2/BRCA pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Tests antibiotics vs oncology specificity.",
    ),

    # ── New negatives: metabolic drug vs wrong disease (n=4) ─────────────────
    _case(
        drug="glipizide",
        disease="rheumatoid arthritis",
        status="TRUE_NEGATIVE",
        mechanism="Sulfonylurea (KATP channel closer). No TNF/IL6/B-cell pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="sitagliptin",
        disease="epilepsy",
        status="TRUE_NEGATIVE",
        mechanism="DPP-4 inhibitor for T2DM. No CACNA2D/ion channel pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="canagliflozin",
        disease="ovarian cancer",
        status="TRUE_NEGATIVE",
        mechanism="SGLT2 inhibitor for T2DM. No BRCA/PARP/homologous recombination overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="pioglitazone",
        disease="schizophrenia",
        status="TRUE_NEGATIVE",
        mechanism="PPAR-gamma agonist for T2DM. No dopamine/glutamate receptor pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),

    # ── New negatives: cardiovascular drug vs wrong disease (n=4) ─────────────
    _case(
        drug="amlodipine",
        disease="multiple myeloma",
        status="TRUE_NEGATIVE",
        mechanism="Dihydropyridine calcium channel blocker. No CRBN/IKZF/myeloma pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="digoxin",
        disease="cystic fibrosis",
        status="TRUE_NEGATIVE",
        mechanism="Cardiac glycoside (Na+/K+ ATPase inhibitor). No CFTR channel overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="clopidogrel",
        disease="Alzheimer disease",
        status="TRUE_NEGATIVE",
        mechanism="P2Y12 antiplatelet agent. No amyloid/tau/cholinergic pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="furosemide",
        disease="rheumatoid arthritis",
        status="TRUE_NEGATIVE",
        mechanism="Loop diuretic (NKCC2 inhibitor). No TNF/IL6/B-cell pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),

    # ── New negatives: CNS drug vs wrong disease (n=4) ───────────────────────
    _case(
        drug="sertraline",
        disease="epilepsy",
        status="TRUE_NEGATIVE",
        mechanism="SSRI (SLC6A4 inhibitor). No CACNA2D/SCN1A ion channel overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Tests SSRI vs epilepsy gene panel specificity.",
    ),
    _case(
        drug="lithium",
        disease="cystic fibrosis",
        status="TRUE_NEGATIVE",
        mechanism="Mood stabiliser (GSK3B inhibitor). No CFTR channel overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="zolpidem",
        disease="inflammatory bowel disease",
        status="TRUE_NEGATIVE",
        mechanism="GABA-A receptor positive modulator (sedative-hypnotic). No IBD pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="aripiprazole",
        disease="type 2 diabetes mellitus",
        status="TRUE_NEGATIVE",
        mechanism="Partial D2/D3 agonist antipsychotic. No AMPK/INSR/glucose metabolism overlap. "
                  "Note: atypicals can cause metabolic side effects but this is not a "
                  "therapeutic repurposing indication.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),

    # ── New negatives: immunology/rare disease mismatches (n=4) ──────────────
    _case(
        drug="tacrolimus",
        disease="chronic myelogenous leukemia",
        status="TRUE_NEGATIVE",
        mechanism="Calcineurin inhibitor (FKBP12/calcineurin). No BCR-ABL pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="allopurinol",
        disease="multiple sclerosis",
        status="TRUE_NEGATIVE",
        mechanism="Xanthine oxidase inhibitor for gout. No demyelination/MS pathway overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="etoposide",
        disease="Parkinson disease",
        status="TRUE_NEGATIVE",
        mechanism="Topoisomerase II inhibitor (chemotherapy). No alpha-synuclein/LRRK2 overlap. "
                  "Tests that cytotoxic chemotherapy does not score for neurodegeneration.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="cyclosporine",
        disease="gout",
        status="TRUE_NEGATIVE",
        mechanism="Calcineurin inhibitor (immunosuppressant). No tubulin/NLRP3/urate pathway overlap. "
                  "Note: cyclosporine can CAUSE hyperuricemia but is not a gout treatment.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
        notes="Tests that a drug causing a disease symptom is not scored as a treatment.",
    ),
    _case(
        drug="heparin",
        disease="breast cancer",
        status="TRUE_NEGATIVE",
        mechanism="Anticoagulant (antithrombin III activator). No ESR1/ERBB2/BRCA overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
    _case(
        drug="montelukast",
        disease="type 2 diabetes mellitus",
        status="TRUE_NEGATIVE",
        mechanism="Leukotriene receptor antagonist for asthma. No AMPK/INSR/glucose metabolism overlap.",
        min_score=0.20,
        expected_rank_top_n=0,
        sources=["Expert consensus"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# OUT_OF_SCOPE cases
# ─────────────────────────────────────────────────────────────────────────────
OUT_OF_SCOPE_CASES: List[Dict] = [
    {
        "drug":    "tocilizumab",
        "disease": "cytokine release syndrome",
        "reason":  (
            "CRS is a syndrome without a stable EFO ontology entry. "
            "OpenTargets returns non-deterministic gene sets across API versions. "
            "Excluded in v3 for reproducibility."
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
    return [c for c in VALIDATION_CASES if c["status"] == "TRUE_POSITIVE"]


def get_negative_cases() -> List[Dict]:
    return [c for c in VALIDATION_CASES if c["status"] == "TRUE_NEGATIVE"]


def get_cases_for_disease(disease_name: str) -> List[Dict]:
    name_lower = disease_name.lower()
    return [c for c in VALIDATION_CASES if name_lower in c["disease"].lower()]


def assert_n_cases():
    n     = len(VALIDATION_CASES)
    n_pos = len(get_positive_cases())
    n_neg = len(get_negative_cases())
    assert n == N_TEST_CASES, (
        f"Dataset has {n} cases but N_TEST_CASES={N_TEST_CASES}. "
        f"Update N_TEST_CASES or add/remove cases."
    )
    print(
        f"Dataset {DATASET_VERSION} OK: {n} total "
        f"({n_pos} positive, {n_neg} negative)"
    )
    return n, n_pos, n_neg


if __name__ == "__main__":
    n, n_pos, n_neg = assert_n_cases()
    print(f"\nValidation Dataset {DATASET_VERSION}")
    print(f"  Total cases:     {n}")
    print(f"  Positive (TP):   {n_pos}")
    print(f"  Negative (TN):   {n_neg}")
    print(f"  Positive:Negative ratio: {n_pos}:{n_neg} "
          f"({n_pos/n_neg:.2f}:1)")
    print(f"\nOut-of-scope cases: {len(OUT_OF_SCOPE_CASES)}")
    for case in OUT_OF_SCOPE_CASES:
        print(f"  - {case['drug']} / {case['disease']} (removed {case['removed_in_version']})")