

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FN investigation categories and remediation notes
# ─────────────────────────────────────────────────────────────────────────────

# Drug names known to depend on the KNOWN_BIOLOGIC_TARGETS / KNOWN_SMALL_MOLECULE_TARGETS
# fallback in data_fetcher.py. These cases may fail if the API returns zero targets
# AND the fallback lookup also fails (e.g. name normalisation mismatch).
KNOWN_FALLBACK_DRUGS = {
    "gabapentin":        "Uses KNOWN_BIOLOGIC_TARGETS fallback for CACNA2D1",
    "propranolol":       "May need fallback — beta-adrenergic targets sparse in DGIdb",
    "hydroxychloroquine":"TLR9/autophagy targets sparse in DGIdb for this drug name",
    "thalidomide":       "Mechanism is anti-angiogenic; may not match standard DGIdb targets",
}

# Diseases known to have sparse or inconsistent OpenTargets gene associations
SPARSE_DISEASE_GENES = {
    "infantile hemangioma":         "Rare pediatric condition; limited OT associations",
    "paroxysmal nocturnal hemoglobinuria": "Rare; complement pathway genes may be sparse",
    "cytokine release syndrome":    "Syndrome (not disease); non-deterministic OT results",
    "tuberous sclerosis":           "mTOR genes present but disease called 'tuberous sclerosis complex'",
}

FN_REMEDIATION = {
    "score_below_threshold": (
        "Score computed but below expected minimum. "
        "Investigate: (1) Are drug targets present in DGIdb/fallback? "
        "(2) Does OpenTargets return the expected disease genes? "
        "(3) Is the mechanism correctly captured by the pathway mapper?"
    ),
    "drug_not_found": (
        "Drug not in candidate list at all. Likely causes: "
        "(1) Drug name in validation_dataset.py does not match ChEMBL preferred name. "
        "(2) Drug has no targets from DGIdb or fallback. "
        "(3) Drug filtered out before scoring. "
        "Check: data_fetcher.py logs for this drug name."
    ),
    "disease_not_found": (
        "Disease not found in OpenTargets. "
        "Check DISEASE_ALIASES in data_fetcher.py. "
        "May need to add disease name variant."
    ),
    "analysis_failed": (
        "Pipeline raised an exception for this case. "
        "Check run_validation.py logs for traceback."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def categorise_fn(case: Dict) -> str:
    """Categorise why a false negative failed."""
    status = case.get("status", "")
    if status == "analysis_failed":
        return "analysis_failed"
    if status == "false_negative" and case.get("rank") is not None:
        return "score_below_threshold"
    if status == "false_negative" and case.get("rank") is None:
        return "drug_not_found"
    if "not_found" in status:
        return "disease_not_found"
    return "score_below_threshold"


def analyse_false_negatives(
    input_path:  str = "validation_results.json",
    output_path: str = "fn_analysis.json",
) -> Dict:
    """
    Load validation results and produce detailed FN analysis.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")

    with open(p) as f:
        data = json.load(f)

    # FIX 1: Accept both key formats (consistent with score_calibration.py)
    positive_cases: Optional[List[Dict]] = (
        data.get("test_cases") or data.get("positive_results")
    )
    if positive_cases is None:
        raise KeyError(
            "Could not find positive case list. Expected 'test_cases' or 'positive_results'"
        )

    # Identify false negatives
    fn_cases = [
        c for c in positive_cases
        if not c.get("pass", True) or c.get("status") in ("false_negative", "analysis_failed")
    ]

    logger.info(f"Total positive cases: {len(positive_cases)}")
    logger.info(f"False negatives:      {len(fn_cases)}")

    if not fn_cases:
        logger.info("No false negatives found. All positive cases passed.")
        result = {
            "summary": {
                "n_positive_cases": len(positive_cases),
                "n_false_negatives": 0,
                "fn_rate": 0.0,
            },
            "false_negative_cases": [],
            "diagnostics": {
                "fallback_dependency_note": (
                    "No false negatives. No fallback dependency issues to report."
                ),
            },
        }
        Path(output_path).write_text(json.dumps(result, indent=2))
        return result

    fn_rate = len(fn_cases) / len(positive_cases) if positive_cases else 0.0

    # Per-case analysis
    detailed_cases = []
    fallback_risk_cases = []

    for case in fn_cases:
        drug    = case["drug"]
        disease = case["disease"]
        score   = float(case.get("raw_score", case.get("score", 0.0)))
        rank    = case.get("rank")
        min_score = case.get("min_score_threshold", "unknown")
        category  = categorise_fn(case)

        detail = {
            "drug":              drug,
            "disease":           disease,
            "raw_score":         score,
            "rank":              rank,
            "min_score_expected":min_score,
            "category":          category,
            "remediation":       FN_REMEDIATION.get(category, "Unknown failure mode"),
            "mechanism":         case.get("mechanism", ""),
            "sources":           case.get("sources", []),
            "notes":             case.get("notes", ""),
        }

        # Flag drugs that rely on fallback targets
        if drug.lower() in KNOWN_FALLBACK_DRUGS:
            detail["fallback_dependency"] = KNOWN_FALLBACK_DRUGS[drug.lower()]
            detail["fallback_investigation"] = (
                "Check data_fetcher.py logs for 'Using fallback targets' message "
                "for this drug. If fallback was NOT triggered, the drug had targets "
                "from DGIdb but they did not produce pathway overlap. "
                "If fallback WAS triggered, verify the hardcoded target list includes "
                "the expected gene."
            )
            fallback_risk_cases.append(drug)

        # Flag diseases with sparse OT data
        disease_lower = disease.lower()
        for sparse_disease, sparse_note in SPARSE_DISEASE_GENES.items():
            if sparse_disease in disease_lower:
                detail["sparse_disease_warning"] = sparse_note
                break

        detailed_cases.append(detail)
        logger.info(
            f"  FN: {drug} / {disease} — "
            f"score={score:.3f}, rank={rank}, category={category}"
        )

    # Diagnostics: Fallback usage counter
    # data_fetcher.py is expected to log:
    #   "Using fallback targets for {drug_name}: {target_list}"
    # and increment an internal counter. The counter value is NOT captured in
    # validation_results.json currently — this is a known gap flagged for
    # future improvement.
    # TODO: Add fallback_usage_count to validation_results.json metadata
    #       so this script can surface exact counts per drug.
    fallback_diagnostics = {
        "drugs_at_risk_of_fallback_issues": fallback_risk_cases,
        "fallback_counter_note": (
            "data_fetcher.py logs a 'Using fallback targets' message each time "
            "the KNOWN_BIOLOGIC_TARGETS / KNOWN_SMALL_MOLECULE_TARGETS fallback "
            "is triggered. To count these for the paper Methods section: "
            "grep the run log for 'Using fallback targets' and count occurrences. "
            "TODO: Add _fallback_usage_count to pipeline diagnostics output so "
            "this script can surface the count automatically."
        ),
        "methods_disclosure_required": (
            "The paper Methods section must disclose: "
            "'When all live API sources (ChEMBL, DGIdb) returned zero targets for "
            "an approved drug, the pipeline fell back to a curated last-resort target "
            "list (KNOWN_BIOLOGIC_TARGETS, KNOWN_SMALL_MOLECULE_TARGETS) derived from "
            "primary pharmacological literature (FDA product labels, peer-reviewed "
            "pharmacology references). This fallback was triggered for [N] drugs "
            "in the validation set and [M] drugs in the full analysis. "
            "The full fallback list is available in data_fetcher.py and as "
            "Supplementary Table S{X}.'"
        ),
    }

    # Category breakdown
    category_counts: Dict[str, int] = {}
    for case in detailed_cases:
        cat = case["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    result = {
        "summary": {
            "n_positive_cases":  len(positive_cases),
            "n_false_negatives": len(fn_cases),
            "fn_rate":           round(fn_rate, 4),
            "fn_rate_percent":   round(fn_rate * 100, 2),
            "category_breakdown":category_counts,
        },
        "false_negative_cases": detailed_cases,
        "diagnostics": {
            "fallback_dependency": fallback_diagnostics,
            "sparse_disease_genes": dict(SPARSE_DISEASE_GENES),
            "recommended_investigations": [
                "1. Check data_fetcher.py logs for each FN drug — did it log "
                "   'Using fallback targets'? If not, was it in DGIdb?",
                "2. For score_below_threshold cases: check which genes/pathways "
                "   were returned by OpenTargets for each disease.",
                "3. For drug_not_found cases: verify drug name in ChEMBL matches "
                "   validation_dataset.py (try 'imatinib mesylate' vs 'imatinib').",
                "4. For disease_not_found cases: add alias to DISEASE_ALIASES in "
                "   data_fetcher.py.",
                "5. After fixes: re-run validate_all.sh and confirm FN count decreases.",
            ],
        },
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"\nFN analysis written to: {out_path.resolve()}")

    logger.info("\nSUMMARY")
    logger.info(f"  False negative rate: {fn_rate:.1%}")
    logger.info(f"  Categories: {category_counts}")
    if fallback_risk_cases:
        logger.info(f"  Fallback-risk drugs: {fallback_risk_cases}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse false negative cases from validation results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="validation_results.json",
        help="Path to validation_results.json (default: validation_results.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fn_analysis.json",
        help="Output path for fn_analysis.json (default: fn_analysis.json)",
    )
    args = parser.parse_args()

    try:
        analyse_false_negatives(
            input_path=args.input,
            output_path=args.output,
        )
    except Exception as e:
        logger.error(f"FN analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()