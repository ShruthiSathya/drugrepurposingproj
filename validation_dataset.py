#!/usr/bin/env python3
"""
EXPANDED Validation Dataset v2
================================
Expanded from 7 → 42 test cases, sourced from:
  1. DrugCentral (Ursu et al. 2019, Nucleic Acids Res)  
  2. Rephetio / Hetionet (Himmelstein et al. 2017, eLife)
  3. FDA drug label repurposing approvals
  4. Systematic review: Pushpakom et al. 2019, Nat Rev Drug Discov

Split:
  TUNING_SET         (8 cases)  — calibration only, never reported
  KNOWN_REPURPOSING_CASES (34 cases) — TEST SET, headline metrics
  NEGATIVE_CONTROLS  (15 cases) — specificity measurement

Categories:
  mechanism_congruent  — strong gene/pathway overlap expected
  empirical            — discovered clinically, indirect mechanism
  literature_supported — PubMed co-occurrence signal present
"""

# ─────────────────────────────────────────────────────────────────────────────
# TUNING SET (calibration only — NEVER reported)
# ─────────────────────────────────────────────────────────────────────────────
TUNING_SET = [
    {
        "drug_name":           "Thalidomide",
        "original_indication": "Morning sickness / sedative",
        "repurposed_for":      "multiple myeloma",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "CRBN-mediated protein degradation, immunomodulation",
        "reference":           "PMID: 16829528",
        "expected_score_range": (0.35, 0.75),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Amantadine",
        "original_indication": "Influenza A",
        "repurposed_for":      "Parkinson disease",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "NMDA antagonism, dopamine release",
        "reference":           "PMID: 4142340",
        "expected_score_range": (0.35, 0.85),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Raloxifene",
        "original_indication": "Osteoporosis",
        "repurposed_for":      "breast carcinoma",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Selective estrogen receptor modulation",
        "reference":           "PMID: 16880444",
        "expected_score_range": (0.40, 0.85),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Finasteride",
        "original_indication": "Benign prostatic hyperplasia",
        "repurposed_for":      "alopecia",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "5-alpha reductase inhibition",
        "reference":           "PMID: 9429826",
        "expected_score_range": (0.50, 0.90),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Celecoxib",
        "original_indication": "Osteoarthritis / pain",
        "repurposed_for":      "colorectal cancer prevention",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "COX-2 inhibition reduces polyp formation",
        "reference":           "PMID: 11157017",
        "expected_score_range": (0.25, 0.65),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Tamoxifen",
        "original_indication": "Breast cancer treatment",
        "repurposed_for":      "bipolar disorder",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "PKC inhibition (anti-manic effect)",
        "reference":           "PMID: 18079950",
        "expected_score_range": (0.20, 0.55),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Valproic acid",
        "original_indication": "Epilepsy",
        "repurposed_for":      "bipolar disorder",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "HDAC inhibition, mood stabilisation",
        "reference":           "PMID: 12493080",
        "expected_score_range": (0.25, 0.70),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Dexamethasone",
        "original_indication": "Inflammation / immunosuppression",
        "repurposed_for":      "multiple myeloma",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "glucocorticoid receptor-mediated apoptosis in plasma cells",
        "reference":           "PMID: 11352963",
        "expected_score_range": (0.25, 0.65),
        "source":              "DrugCentral",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TEST SET — 34 cases (report headline metrics from this ONLY)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_REPURPOSING_CASES = [

    # ══ MECHANISM-CONGRUENT (n=18) ══════════════════════════════════════════

    # PAH / Cardiology
    {
        "drug_name":           "Sildenafil",
        "original_indication": "Angina / erectile dysfunction",
        "repurposed_for":      "pulmonary arterial hypertension",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "PDE5 inhibition → cGMP↑ → pulmonary vasodilation",
        "reference":           "PMID: 16115318",
        "expected_score_range": (0.40, 0.85),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Imatinib",
        "original_indication": "Chronic myeloid leukaemia",
        "repurposed_for":      "pulmonary arterial hypertension",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "PDGFR/ABL kinase inhibition in vascular remodelling",
        "reference":           "PMID: 16478904",
        "expected_score_range": (0.30, 0.70),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Bosentan",
        "original_indication": "Pulmonary arterial hypertension",
        "repurposed_for":      "systemic sclerosis",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Endothelin receptor antagonism reduces fibrotic vasculopathy",
        "reference":           "PMID: 15016484",
        "expected_score_range": (0.25, 0.65),
        "source":              "DrugCentral",
    },

    # Oncology
    {
        "drug_name":           "Rituximab",
        "original_indication": "Non-Hodgkin lymphoma",
        "repurposed_for":      "rheumatoid arthritis",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "CD20/MS4A1 targeting, B-cell depletion",
        "reference":           "PMID: 16339094",
        "expected_score_range": (0.40, 0.90),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Trastuzumab",
        "original_indication": "HER2+ breast cancer",
        "repurposed_for":      "HER2+ gastric cancer",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "ERBB2 (HER2) receptor blockade",
        "reference":           "PMID: 20728210",
        "expected_score_range": (0.35, 0.80),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Bevacizumab",
        "original_indication": "Colorectal cancer",
        "repurposed_for":      "non-small cell lung carcinoma",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "VEGF-A neutralisation, anti-angiogenesis",
        "reference":           "PMID: 16467544",
        "expected_score_range": (0.30, 0.70),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Gefitinib",
        "original_indication": "NSCLC (second-line)",
        "repurposed_for":      "head and neck squamous cell carcinoma",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "EGFR kinase inhibition",
        "reference":           "PMID: 16505417",
        "expected_score_range": (0.25, 0.65),
        "source":              "Rephetio",
    },

    # Metabolic / hormonal
    {
        "drug_name":           "Metformin",
        "original_indication": "Type 2 diabetes",
        "repurposed_for":      "polycystic ovary syndrome",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "AMPK activation, insulin sensitisation",
        "reference":           "PMID: 17050835",
        "expected_score_range": (0.35, 0.80),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Pioglitazone",
        "original_indication": "Type 2 diabetes",
        "repurposed_for":      "non-alcoholic steatohepatitis",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "PPARgamma activation, hepatic insulin sensitisation",
        "reference":           "PMID: 16738015",
        "expected_score_range": (0.25, 0.65),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Spironolactone",
        "original_indication": "Heart failure / hypertension",
        "repurposed_for":      "polycystic ovary syndrome",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Androgen receptor blockade, anti-androgenic",
        "reference":           "PMID: 24575146",
        "expected_score_range": (0.25, 0.65),
        "source":              "Rephetio",
    },

    # Immunology / inflammation
    {
        "drug_name":           "Tocilizumab",
        "original_indication": "Rheumatoid arthritis",
        "repurposed_for":      "cytokine release syndrome",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "IL-6 receptor blockade",
        "reference":           "PMID: 31085063",
        "expected_score_range": (0.30, 0.70),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Abatacept",
        "original_indication": "Rheumatoid arthritis",
        "repurposed_for":      "juvenile idiopathic arthritis",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "CTLA4-Ig T-cell co-stimulation blockade",
        "reference":           "PMID: 18509108",
        "expected_score_range": (0.25, 0.65),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Hydroxychloroquine",
        "original_indication": "Malaria",
        "repurposed_for":      "systemic lupus erythematosus",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "TLR7/9 signalling inhibition, lysosomal pH disruption",
        "reference":           "PMID: 16462997",
        "expected_score_range": (0.25, 0.65),
        "source":              "DrugCentral",
    },

    # Neurology
    {
        "drug_name":           "Memantine",
        "original_indication": "Moderate-to-severe Alzheimer disease",
        "repurposed_for":      "neuropathic pain",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "NMDA receptor antagonism blocks central sensitisation",
        "reference":           "PMID: 14570916",
        "expected_score_range": (0.25, 0.65),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Gabapentin",
        "original_indication": "Epilepsy",
        "repurposed_for":      "neuropathic pain",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Alpha-2-delta subunit of voltage-gated calcium channels",
        "reference":           "PMID: 10333316",
        "expected_score_range": (0.20, 0.60),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Donepezil",
        "original_indication": "Alzheimer disease",
        "repurposed_for":      "vascular dementia",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Acetylcholinesterase inhibition improves cholinergic tone",
        "reference":           "PMID: 15033282",
        "expected_score_range": (0.25, 0.65),
        "source":              "DrugCentral",
    },

    # Hair / dermatology
    {
        "drug_name":           "Minoxidil",
        "original_indication": "Hypertension",
        "repurposed_for":      "alopecia",
        "fda_approved":        True,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Potassium channel opening (KCNJ8/ABCC9), vasodilation",
        "reference":           "PMID: 2458842",
        "expected_score_range": (0.15, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Dutasteride",
        "original_indication": "Benign prostatic hyperplasia",
        "repurposed_for":      "alopecia",
        "fda_approved":        False,
        "category":            "mechanism_congruent",
        "shared_mechanism":    "Dual 5-alpha reductase (SRD5A1+SRD5A2) inhibition",
        "reference":           "PMID: 17034374",
        "expected_score_range": (0.40, 0.85),
        "source":              "DrugCentral",
    },

    # ══ EMPIRICAL (n=10) — discovered clinically / serendipitous ════════════

    {
        "drug_name":           "Aspirin",
        "original_indication": "Pain / fever / inflammation",
        "repurposed_for":      "coronary artery disease",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "COX / platelet aggregation inhibition",
        "reference":           "PMID: 7386825",
        "expected_score_range": (0.20, 0.70),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Propranolol",
        "original_indication": "Hypertension / angina",
        "repurposed_for":      "essential tremor",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Beta-adrenergic blockade (ADRB1/ADRB2)",
        "reference":           "PMID: 6370753",
        "expected_score_range": (0.10, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Bupropion",
        "original_indication": "Depression",
        "repurposed_for":      "smoking cessation",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Nicotinic receptor antagonism / dopamine reuptake inhibition",
        "reference":           "PMID: 9289242",
        "expected_score_range": (0.15, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Duloxetine",
        "original_indication": "Depression / GAD",
        "repurposed_for":      "fibromyalgia",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "SNRI, central pain sensitisation modulation",
        "reference":           "PMID: 17196756",
        "expected_score_range": (0.15, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Pregabalin",
        "original_indication": "Epilepsy",
        "repurposed_for":      "fibromyalgia",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Voltage-gated calcium channel subunit binding",
        "reference":           "PMID: 16387521",
        "expected_score_range": (0.15, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Sildenafil",
        "original_indication": "Angina / erectile dysfunction",
        "repurposed_for":      "Raynaud phenomenon",
        "fda_approved":        False,
        "category":            "empirical",
        "shared_mechanism":    "PDE5 inhibition → peripheral vasodilation",
        "reference":           "PMID: 16849424",
        "expected_score_range": (0.15, 0.55),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Methotrexate",
        "original_indication": "Cancer chemotherapy",
        "repurposed_for":      "rheumatoid arthritis",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Folate antagonism, anti-inflammatory at low dose",
        "reference":           "PMID: 3309002",
        "expected_score_range": (0.15, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Clonidine",
        "original_indication": "Hypertension",
        "repurposed_for":      "attention deficit hyperactivity disorder",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Alpha-2 adrenoreceptor agonism in prefrontal cortex",
        "reference":           "PMID: 8913238",
        "expected_score_range": (0.10, 0.50),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Naltrexone",
        "original_indication": "Opioid use disorder",
        "repurposed_for":      "alcohol use disorder",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Mu-opioid receptor blockade reduces craving",
        "reference":           "PMID: 7745002",
        "expected_score_range": (0.10, 0.50),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Topiramate",
        "original_indication": "Epilepsy",
        "repurposed_for":      "migraine prevention",
        "fda_approved":        True,
        "category":            "empirical",
        "shared_mechanism":    "Glutamate/AMPA blockade, cortical spreading suppression",
        "reference":           "PMID: 15243308",
        "expected_score_range": (0.10, 0.50),
        "source":              "DrugCentral",
    },

    # ══ LITERATURE-SUPPORTED (n=6) — PubMed signal present ═════════════════

    {
        "drug_name":           "Metformin",
        "original_indication": "Type 2 diabetes",
        "repurposed_for":      "colorectal cancer",
        "fda_approved":        False,
        "category":            "literature_supported",
        "shared_mechanism":    "AMPK/mTOR pathway, cancer cell energy depletion",
        "reference":           "PMID: 23255621",
        "expected_score_range": (0.20, 0.60),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Atorvastatin",
        "original_indication": "Hypercholesterolaemia",
        "repurposed_for":      "coronary artery disease",
        "fda_approved":        True,
        "category":            "literature_supported",
        "shared_mechanism":    "HMGCR inhibition, pleiotropic anti-inflammatory effects",
        "reference":           "PMID: 8560330",
        "expected_score_range": (0.25, 0.65),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Doxycycline",
        "original_indication": "Bacterial infection",
        "repurposed_for":      "rosacea",
        "fda_approved":        True,
        "category":            "literature_supported",
        "shared_mechanism":    "MMP inhibition, anti-inflammatory at sub-antimicrobial dose",
        "reference":           "PMID: 18492077",
        "expected_score_range": (0.10, 0.45),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Colchicine",
        "original_indication": "Gout",
        "repurposed_for":      "pericarditis",
        "fda_approved":        True,
        "category":            "literature_supported",
        "shared_mechanism":    "Microtubule depolymerisation, neutrophil inhibition",
        "reference":           "PMID: 24694394",
        "expected_score_range": (0.15, 0.55),
        "source":              "DrugCentral",
    },
    {
        "drug_name":           "Azithromycin",
        "original_indication": "Bacterial infection",
        "repurposed_for":      "diffuse panbronchiolitis",
        "fda_approved":        False,
        "category":            "literature_supported",
        "shared_mechanism":    "Immunomodulatory / anti-inflammatory at low dose",
        "reference":           "PMID: 9517165",
        "expected_score_range": (0.10, 0.45),
        "source":              "Rephetio",
    },
    {
        "drug_name":           "Losartan",
        "original_indication": "Hypertension",
        "repurposed_for":      "Marfan syndrome",
        "fda_approved":        False,
        "category":            "literature_supported",
        "shared_mechanism":    "AT1R blockade reduces TGF-beta signalling in aorta",
        "reference":           "PMID: 18650507",
        "expected_score_range": (0.15, 0.55),
        "source":              "Rephetio",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE CONTROLS (n=15)
# ─────────────────────────────────────────────────────────────────────────────
NEGATIVE_CONTROLS = [
    # Original 5
    {
        "drug_name":             "Aspirin",
        "disease":               "Alzheimer disease",
        "expected_score_range":  (0.0, 0.30),
        "reason":                "No established shared mechanism or pathway",
        "source":                "Curated",
    },
    {
        "drug_name":             "Metformin",
        "disease":               "acute myeloid leukemia",
        "expected_score_range":  (0.0, 0.35),
        "reason":                "Different pathophysiology; metabolic genes peripherally linked",
        "note":                  "Marginal metabolic overlap is acknowledged — score cap set to 0.35",
        "source":                "Curated",
    },
    {
        "drug_name":             "Ibuprofen",
        "disease":               "Parkinson disease",
        "expected_score_range":  (0.0, 0.25),
        "reason":                "NSAID with no dopaminergic/NMDA mechanism",
        "source":                "Curated",
    },
    {
        "drug_name":             "Warfarin",
        "disease":               "multiple myeloma",
        "expected_score_range":  (0.0, 0.25),
        "reason":                "Anticoagulant; no relevant oncological pathway",
        "source":                "Curated",
    },
    {
        "drug_name":             "Omeprazole",
        "disease":               "Parkinson disease",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "Proton-pump inhibitor; no neurological mechanism",
        "source":                "Curated",
    },
    # New 10 (from Rephetio negative pairs)
    {
        "drug_name":             "Metformin",
        "disease":               "schizophrenia",
        "expected_score_range":  (0.0, 0.25),
        "reason":                "Antidiabetic; no dopaminergic or serotonergic pathway",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Atorvastatin",
        "disease":               "Parkinson disease",
        "expected_score_range":  (0.0, 0.25),
        "reason":                "Cholesterol-lowering; no dopamine synthesis connection",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Lisinopril",
        "disease":               "multiple myeloma",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "ACE inhibitor for hypertension; no plasma cell mechanism",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Levothyroxine",
        "disease":               "rheumatoid arthritis",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "Thyroid hormone replacement; not an immunomodulator",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Amlodipine",
        "disease":               "breast carcinoma",
        "expected_score_range":  (0.0, 0.25),
        "reason":                "Calcium channel blocker; no direct anti-tumour mechanism",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Omeprazole",
        "disease":               "rheumatoid arthritis",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "Proton pump inhibitor; no joint inflammation pathway",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Warfarin",
        "disease":               "Alzheimer disease",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "Anticoagulant; no amyloid or tau pathway link",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Albuterol",
        "disease":               "multiple myeloma",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "Beta-2 agonist bronchodilator; no plasma cell pathway",
        "source":                "Curated",
    },
    {
        "drug_name":             "Metoprolol",
        "disease":               "breast carcinoma",
        "expected_score_range":  (0.0, 0.25),
        "reason":                "Beta-1 selective blocker; no direct oncological mechanism",
        "source":                "Rephetio",
    },
    {
        "drug_name":             "Pantoprazole",
        "disease":               "Parkinson disease",
        "expected_score_range":  (0.0, 0.20),
        "reason":                "Proton pump inhibitor; no neurodegeneration mechanism",
        "source":                "Curated",
    },
]


def get_validation_metrics_target() -> dict:
    """
    Minimum performance thresholds for publication.
    Thresholds set against full TEST SET (KNOWN_REPURPOSING_CASES).
    """
    return {
        "sensitivity": 0.65,
        "specificity": 0.75,
        "precision":   0.65,
        "mechanism_congruent_sensitivity": 0.75,
        "empirical_sensitivity":           0.40,
        "literature_supported_sensitivity": 0.50,
        "explanation": (
            "Empirical and literature-supported cases are expected to score "
            "lower because their repurposing mechanism is indirect or post-hoc. "
            "The algorithm is primarily gene-overlap-driven."
        ),
    }


def get_test_diseases() -> dict:
    diseases = set()
    for c in KNOWN_REPURPOSING_CASES:
        diseases.add(c["repurposed_for"])
    for c in NEGATIVE_CONTROLS:
        diseases.add(c["disease"])
    return {d: d for d in sorted(diseases)}


def get_all_test_cases() -> list:
    return TUNING_SET + KNOWN_REPURPOSING_CASES