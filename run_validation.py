"""
run_validation.py — Curated Validation Runner
==============================================
Runs the production pipeline against the curated 33-case validation dataset
and writes results to validation_results.json.

FIXES vs previous version
--------------------------
1. Import: Now imports ProductionPipeline directly (also works via RepurposingPipeline
   alias), and uses generate_candidates() method on the pipeline object.
   Old code imported 'RepurposingPipeline' which did not exist as a class name
   in production_pipeline.py — would cause ImportError at runtime.

2. Output keys: validation_results.json now includes BOTH key formats:
   - 'positive_results' and 'negative_results'    (legacy keys, preserved)
   - 'test_cases' and 'negative_cases'             (new keys expected by score_calibration.py)
   This dual-key approach means both run_validation.py and score_calibration.py
   work without changes to either side, and validate_all.sh step 2 succeeds.

3. n_test_cases: Header now reports 33 (corrected from 34). Tocilizumab/CRS
   was moved to OUT_OF_SCOPE in validation_dataset.py v3.

Usage
-----
    python run_validation.py [--min-score 0.0] [--output validation_results.json]
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Import the class by its canonical name.
# Both ProductionPipeline and RepurposingPipeline (alias) are defined in the
# module; using the canonical name avoids confusion.
# ─────────────────────────────────────────────────────────────────────────────
from backend.pipeline.production_pipeline import ProductionPipeline
from backend.pipeline.calibration import calibrate_score
from validation_dataset import (
    VALIDATION_CASES,
    DATASET_VERSION,
    N_TEST_CASES,
    get_positive_cases,
    get_negative_cases,
    OUT_OF_SCOPE_CASES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_single_validation_case(
    pipeline: ProductionPipeline,
    drugs_data: List[Dict],
    case: Dict,
) -> Dict:
    """
    Run one validation case: fetch disease data, score all drugs, return result.
    Uses generate_candidates() so results are consistent with RepoDB benchmark.
    """
    drug_name    = case["drug"]
    disease_name = case["disease"]

    logger.info(f"  Testing: {drug_name} vs {disease_name} ...")

    # Fetch disease data
    disease_data = await pipeline.data_fetcher.fetch_disease_data(disease_name)

    if not disease_data:
        logger.warning(f"    Disease not found in OpenTargets: {disease_name}")
        return {
            "drug":              drug_name,
            "disease":           disease_name,
            "status":            "analysis_failed",
            "reason":            "Disease not found in OpenTargets",
            "raw_score":         0.0,
            "calibrated_score":  0.0,
            "rank":              None,
            "expected_status":   case["status"],
            "pass":              False,
            "notes":             case.get("notes", ""),
        }

    # Score all drugs
    candidates = await pipeline.generate_candidates(
        disease_data=disease_data,
        drugs_data=drugs_data,
        min_score=0.0,
        fetch_pubmed=False,   # PubMed disabled for speed in validation run
    )

    # Find this drug in results
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
    drug_lower = drug_name.lower()
    found_candidate = None
    found_rank = None

    for rank, cand in enumerate(candidates_sorted, 1):
        if cand["name"].lower() == drug_lower:
            found_candidate = cand
            found_rank = rank
            break

    if found_candidate is None:
        result_status = "false_negative" if case["status"] == "TRUE_POSITIVE" else "true_negative_not_found"
        raw_score  = 0.0
        cal_score  = calibrate_score(0.0)
        passed     = case["status"] == "TRUE_NEGATIVE"
        logger.info(f"    {drug_name} not in candidates — status: {result_status}")
    else:
        raw_score = found_candidate["score"]
        cal_score = calibrate_score(raw_score)

        if case["status"] == "TRUE_POSITIVE":
            passed = raw_score >= case["min_score"]
            result_status = "found" if passed else "false_negative"
            if not passed:
                logger.warning(
                    f"    FAIL: {drug_name} score {raw_score:.3f} < "
                    f"expected {case['min_score']}"
                )
        else:  # TRUE_NEGATIVE
            passed = raw_score < case["min_score"]
            result_status = (
                "true_negative_low_score" if passed else "false_positive"
            )
            if not passed:
                logger.warning(
                    f"    FAIL (FP risk): {drug_name} scored {raw_score:.3f} "
                    f"(threshold {case['min_score']})"
                )

    result = {
        "drug":                  drug_name,
        "disease":               disease_name,
        "status":                result_status,
        "expected_status":       case["status"],
        "raw_score":             round(raw_score, 4),
        "calibrated_score":      round(cal_score, 4),
        "rank":                  found_rank,
        "expected_rank_top_n":   case["expected_rank_top_n"],
        "rank_pass":             (
            found_rank is not None and case["expected_rank_top_n"] > 0
            and found_rank <= case["expected_rank_top_n"]
        ) if case["expected_rank_top_n"] > 0 else None,
        "pass":                  passed,
        "min_score_threshold":   case["min_score"],
        "mechanism":             case["mechanism"],
        "sources":               case["sources"],
        "notes":                 case.get("notes", ""),
    }

    logger.info(
        f"    {'PASS' if passed else 'FAIL'} — "
        f"raw={raw_score:.3f} cal={cal_score:.3f} "
        f"rank={found_rank} status={result_status}"
    )

    return result


async def run_all_validations(
    min_score:   float = 0.0,
    output_path: str   = "validation_results.json",
) -> Dict:
    """
    Run all validation cases and write results JSON.
    """
    pipeline  = ProductionPipeline()
    start_utc = datetime.now(timezone.utc)

    logger.info("=" * 70)
    logger.info(f"VALIDATION RUN — Dataset {DATASET_VERSION} — {N_TEST_CASES} cases")
    logger.info("=" * 70)

    # Fetch drugs once (shared across all disease queries for speed)
    logger.info("\nFetching approved drugs (shared across all test cases)...")
    drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
    logger.info(f"Using {len(drugs_data)} drugs for all tests\n")

    positive_cases = get_positive_cases()
    negative_cases = get_negative_cases()

    logger.info(f"Running {len(positive_cases)} TRUE_POSITIVE cases...")
    positive_results = []
    for case in positive_cases:
        result = await run_single_validation_case(pipeline, drugs_data, case)
        positive_results.append(result)

    logger.info(f"\nRunning {len(negative_cases)} TRUE_NEGATIVE cases...")
    negative_results = []
    for case in negative_cases:
        result = await run_single_validation_case(pipeline, drugs_data, case)
        negative_results.append(result)

    await pipeline.close()

    # Compute metrics
    n_pos = len(positive_results)
    n_neg = len(negative_results)
    tp = sum(1 for r in positive_results if r["pass"])
    fn = n_pos - tp
    tn = sum(1 for r in negative_results if r["pass"])
    fp = n_neg - tn

    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0 else 0.0
    )

    end_utc = datetime.now(timezone.utc)
    elapsed = (end_utc - start_utc).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  True Positives:  {tp}/{n_pos} ({sensitivity:.1%})")
    logger.info(f"  True Negatives:  {tn}/{n_neg} ({specificity:.1%})")
    logger.info(f"  Sensitivity:     {sensitivity:.3f}")
    logger.info(f"  Specificity:     {specificity:.3f}")
    logger.info(f"  Precision:       {precision:.3f}")
    logger.info(f"  F1:              {f1:.3f}")
    logger.info(f"  Elapsed:         {elapsed:.1f}s")

    # ─── OUTPUT JSON ──────────────────────────────────────────────────────────
    # FIX 2: Include BOTH key formats to satisfy score_calibration.py input
    # expectations ('test_cases' / 'negative_cases') AND any legacy code that
    # reads 'positive_results' / 'negative_results'.
    # ──────────────────────────────────────────────────────────────────────────
    output = {
        "header": {
            "dataset_version":    DATASET_VERSION,
            "n_test_cases":       N_TEST_CASES,   # FIX 3: 33, not 34
            "n_positive_cases":   n_pos,
            "n_negative_cases":   n_neg,
            "run_timestamp_utc":  start_utc.isoformat(),
            "elapsed_seconds":    round(elapsed, 2),
            "min_score_used":     min_score,
        },
        "metrics": {
            "tp":            tp,
            "fn":            fn,
            "tn":            tn,
            "fp":            fp,
            "sensitivity":   round(sensitivity, 4),
            "specificity":   round(specificity, 4),
            "precision":     round(precision, 4),
            "f1":            round(f1, 4),
        },
        # --- Primary output keys (legacy, preserved) ---
        "positive_results": positive_results,
        "negative_results": negative_results,
        # --- Alias keys expected by score_calibration.py ---
        # score_calibration.py reads 'test_cases' and 'negative_cases'.
        # These point to the same lists as above — no duplication of logic.
        "test_cases":     positive_results,
        "negative_cases": negative_results,
        # --- Out-of-scope documentation ---
        "out_of_scope_cases": [
            {
                "drug":    c["drug"],
                "disease": c["disease"],
                "reason":  c["reason"],
                "version_removed": c["removed_in_version"],
            }
            for c in OUT_OF_SCOPE_CASES
        ],
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults written to: {out_path.resolve()}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run curated validation suite against the drug repurposing pipeline"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum raw score to include in candidate list (default 0.0 = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output JSON file path (default: validation_results.json)",
    )
    args = parser.parse_args()

    try:
        result = asyncio.run(
            run_all_validations(
                min_score=args.min_score,
                output_path=args.output,
            )
        )
        metrics = result["metrics"]
        print("\nFINAL METRICS")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  F1:          {metrics['f1']:.3f}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()