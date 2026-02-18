#!/usr/bin/env python3
"""
Validation Dataset - Known Drug Repurposing Successes
======================================================
This module contains well-documented successful drug repurposing cases
for validating our algorithm's predictive accuracy.

References:
-----------
1. Pushpakom et al. (2019) "Drug repurposing: progress, challenges and recommendations"
   Nature Reviews Drug Discovery. PMID: 30310233
2. Novac (2013) "Challenges and opportunities of drug repositioning" 
   Trends in Pharmacological Sciences. PMID: 23769625
"""

KNOWN_REPURPOSING_CASES = [
    {
        "drug_name": "Sildenafil",
        "original_indication": "Angina / Hypertension",
        "repurposed_for": "Pulmonary Arterial Hypertension",
        "success_year": 2005,
        "fda_approved": True,
        "shared_mechanism": "PDE5 inhibition",
        "reference": "PMID: 16115318",
        "expected_score_range": (0.6, 0.85),  # Based on strong mechanism overlap
    },
    {
        "drug_name": "Thalidomide",
        "original_indication": "Morning sickness",
        "repurposed_for": "Multiple Myeloma",
        "success_year": 2006,
        "fda_approved": True,
        "shared_mechanism": "Immunomodulation, angiogenesis inhibition",
        "reference": "PMID: 16829528",
        "expected_score_range": (0.5, 0.75),
    },
    {
        "drug_name": "Metformin",
        "original_indication": "Type 2 Diabetes",
        "repurposed_for": "Polycystic Ovary Syndrome",
        "success_year": 2006,
        "fda_approved": False,  # Off-label
        "shared_mechanism": "Insulin sensitization",
        "reference": "PMID: 17050835",
        "expected_score_range": (0.6, 0.8),
    },
    {
        "drug_name": "Aspirin",
        "original_indication": "Pain, fever, inflammation",
        "repurposed_for": "Cardiovascular Disease Prevention",
        "success_year": 1980,
        "fda_approved": True,
        "shared_mechanism": "COX inhibition, platelet aggregation",
        "reference": "PMID: 7386825",
        "expected_score_range": (0.5, 0.75),
    },
    {
        "drug_name": "Rituximab",
        "original_indication": "Non-Hodgkin Lymphoma",
        "repurposed_for": "Rheumatoid Arthritis",
        "success_year": 2006,
        "fda_approved": True,
        "shared_mechanism": "CD20 targeting, B-cell depletion",
        "reference": "PMID: 16339094",
        "expected_score_range": (0.7, 0.9),
    },
    {
        "drug_name": "Minoxidil",
        "original_indication": "Hypertension",
        "repurposed_for": "Alopecia (Hair Loss)",
        "success_year": 1988,
        "fda_approved": True,
        "shared_mechanism": "Vasodilation, potassium channel opening",
        "reference": "PMID: 2458842",
        "expected_score_range": (0.3, 0.6),  # Lower - different mechanism
    },
    {
        "drug_name": "Propranolol",
        "original_indication": "Hypertension, Angina",
        "repurposed_for": "Essential Tremor",
        "success_year": 1984,
        "fda_approved": True,
        "shared_mechanism": "Beta-adrenergic blockade",
        "reference": "PMID: 6370753",
        "expected_score_range": (0.5, 0.75),
    },
    {
        "drug_name": "Amantadine",
        "original_indication": "Influenza A",
        "repurposed_for": "Parkinson Disease",
        "success_year": 1973,
        "fda_approved": True,
        "shared_mechanism": "Dopamine release, NMDA antagonism",
        "reference": "PMID: 4142340",
        "expected_score_range": (0.6, 0.85),
    },
    {
        "drug_name": "Raloxifene",
        "original_indication": "Osteoporosis",
        "repurposed_for": "Breast Cancer Prevention",
        "success_year": 2007,
        "fda_approved": True,
        "shared_mechanism": "Selective estrogen receptor modulation",
        "reference": "PMID: 16880444",
        "expected_score_range": (0.65, 0.85),
    },
    {
        "drug_name": "Finasteride",
        "original_indication": "Benign Prostatic Hyperplasia",
        "repurposed_for": "Male Pattern Baldness",
        "success_year": 1997,
        "fda_approved": True,
        "shared_mechanism": "5-alpha reductase inhibition",
        "reference": "PMID: 9429826",
        "expected_score_range": (0.7, 0.9),  # High - same mechanism
    },
]

NEGATIVE_CONTROLS = [
    # These are drug-disease pairs that SHOULD NOT match
    {
        "drug_name": "Aspirin",
        "disease": "Alzheimer Disease",
        "expected_score_range": (0.0, 0.3),
        "reason": "No shared mechanism or pathway",
    },
    {
        "drug_name": "Metformin",
        "disease": "Acute Myeloid Leukemia",
        "expected_score_range": (0.0, 0.35),
        "reason": "Different pathophysiology",
    },
    {
        "drug_name": "Ibuprofen",
        "disease": "Parkinson Disease",
        "expected_score_range": (0.0, 0.25),
        "reason": "No relevant mechanism",
    },
]


def get_validation_metrics_target():
    """
    Target performance metrics for a scientifically valid algorithm.
    
    Based on:
    - Brown & Patel (2017) "Computational approaches to drug repositioning"
    - Industry benchmarks for drug discovery tools
    """
    return {
        "sensitivity": 0.70,  # 70% of true positives identified
        "specificity": 0.75,  # 75% of true negatives identified
        "precision": 0.65,    # 65% of predictions are correct
        "auc_roc": 0.75,      # Area under ROC curve
        "explanation": """
        These targets are realistic for early-stage computational drug repurposing.
        Perfect accuracy (>95%) would be unrealistic given biological complexity.
        
        For publication, we need:
        - Sensitivity > 65% (catch most true opportunities)
        - Specificity > 70% (avoid too many false positives)
        - Statistical significance (p < 0.05) vs random baseline
        """
    }


def get_test_diseases():
    """
    Return disease names that should be used for validation.
    These are standardized names from OpenTargets.
    """
    return {
        "Pulmonary Arterial Hypertension": "pulmonary hypertension",
        "Multiple Myeloma": "multiple myeloma",
        "Polycystic Ovary Syndrome": "polycystic ovary syndrome",
        "Cardiovascular Disease Prevention": "cardiovascular disease",
        "Rheumatoid Arthritis": "rheumatoid arthritis",
        "Alopecia (Hair Loss)": "alopecia",
        "Essential Tremor": "essential tremor",
        "Parkinson Disease": "Parkinson disease",
        "Breast Cancer Prevention": "breast carcinoma",
        "Male Pattern Baldness": "androgenetic alopecia",
    }