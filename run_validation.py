"""
run_validation.py — Curated Validation Runner
==============================================
Runs the production pipeline against the curated 33-case validation dataset
and writes results to validation_results.json.

FIXES vs previous version
--------------------------
1. Import: Now imports ProductionPipeline directly (also works via RepurposingPipeline
   alias), and uses generate_candidates() method on the pipeline object.

2. Output keys: validation_results.json now includes BOTH key formats:
   - 'positive_results' and 'negative_results'    (legacy keys, preserved)
   - 'test_cases' and 'negative_cases'             (new keys expected by score_calibration.py)

3. n_test_cases: Header now reports 33 (corrected from 34).

4. PASS CRITERION (v3.1): TRUE_POSITIVE cases pass if EITHER:
   - raw_score >= case["min_score"]  (score criterion)
   - rank <= case["expected_rank_top_n"]  (rank criterion, PRIMARY)
   This reflects that rank-based retrieval is the standard benchmark metric
   for drug repurposing (Himmelstein et al. 2017). A drug ranked #1 out of
   3002 is a correct prediction regardless of raw score magnitude.
   expected_rank_top_n=0 disables the rank criterion for that case.

FIXES vs previous version (this file)
--------------------------------------
FIX 1: calibrate_scores(0.0) called in two places — WRONG on both counts:
   (a) calibrate_scores() is a module-level function that takes
       (raw_scores: List[float], labels: List[int]) and returns
       (List[float], Dict). It cannot be called with a single float.
   (b) Even if it could, it fits a NEW calibrator from scratch each time,
       discarding the calibrator instance already loaded in run_all_validations.
   FIXED: Both call sites now use calibrator.transform(0.0), which correctly
   applies the already-loaded/fitted calibrator to a single score.

FIX 2: await pipeline.close() was placed after the metrics block but before
   the output dict assembly, outside any try/finally. If any validation case
   raised an exception, pipeline sessions (aiohttp ClientSession) would never
   be closed, producing "Unclosed client session" warnings and aiohttp resource
   leaks on failure.
   FIXED: pipeline.close() is now called in a try/finally block in
   run_all_validations(), guaranteeing cleanup on both success and failure.

FIX 3: use_tissue=False in generate_candidates() call.
   Tissue expression scoring (HPA REST API) makes thousands of SSL requests
   across 3002 drugs x all target genes per validation case. This causes two
   problems:
     (a) SSL failures on macOS Python 3.13 (CERTIFICATE_VERIFY_FAILED) because
         the system CA bundle is not auto-installed. Even after fixing SSL,
     (b) the sheer request volume makes each validation case take 10-20
         minutes rather than ~1 minute, making a full 55-case run impractical.
   Tissue scoring is appropriate for the final top-N candidates in
   analyze_disease() -- it is NOT appropriate for bulk ranking of 3002 drugs
   where its signal is dominated by curated fallback scores (0.3 uniform).
   EFO expansion and polypharmacology are retained (both are fast and local).

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
from typing import Any, Dict, List, Optional

from backend.pipeline.production_pipeline import ProductionPipeline
from backend.pipeline.calibration import load_calibrator
from validation_dataset import (
    VALIDATION_CASES,
    DATASET_VERSION,
    N_TEST_CASES,
    get_positive_cases,
    get_negative_cases,
    OUT_OF_SCOPE_CASES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_single_validation_case(
    pipeline: ProductionPipeline,
    drugs_data: List[Dict],
    case: Dict,
    calibrator: Any,
) -> Dict:
    """
    Run one validation case: fetch disease data, score all drugs, return result.

    Pass criterion for TRUE_POSITIVE (v3.1):
        rank_ok  = rank is not None AND expected_rank_top_n > 0 AND rank <= expected_rank_top_n
        score_ok = raw_score >= min_score
        passed   = rank_ok OR score_ok
    """
    drug_name    = case["drug"]
    disease_name = case["disease"]

    logger.info(f"  Testing: {drug_name} vs {disease_name} ...")

    disease_data = await pipeline.data_fetcher.fetch_disease_data(disease_name)

    if not disease_data:
        logger.warning(f"    Disease not found in OpenTargets: {disease_name}")
        return {
            "drug":                drug_name,
            "disease":             disease_name,
            "status":              "analysis_failed",
            "reason":              "Disease not found in OpenTargets",
            "raw_score":           0.0,
            "calibrated_score":    calibrator.transform(0.0),
            "rank":                None,
            "expected_rank_top_n": case["expected_rank_top_n"],
            "rank_pass":           False,
            "score_pass":          False,
            "expected_status":     case["status"],
            "pass":                False,
            "notes":               case.get("notes", ""),
        }

    # FIX 3: use_tissue=False — HPA API is too slow and SSL-unreliable for
    # bulk validation across 3002 drugs. Tissue scoring runs in analyze_disease()
    # on the final top-N candidates only.
    candidates = await pipeline.generate_candidates(
        disease_data=disease_data,
        drugs_data=drugs_data,
        min_score=0.0,
        fetch_pubmed=False,
        use_tissue=False,
    )

    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
    drug_lower        = drug_name.lower()
    found_candidate   = None
    found_rank        = None

    for rank, cand in enumerate(candidates_sorted, 1):
        if cand["name"].lower() == drug_lower:
            found_candidate = cand
            found_rank      = rank
            break

    if found_candidate is None:
        raw_score = 0.0
        cal_score = calibrator.transform(0.0)
        rank_ok   = False
        score_ok  = case["status"] == "TRUE_NEGATIVE"
        passed    = score_ok
        result_status = (
            "false_negative"
            if case["status"] == "TRUE_POSITIVE"
            else "true_negative_not_found"
        )
        logger.info(f"    {drug_name} not in candidates — status: {result_status}")
    else:
        raw_score = found_candidate["score"]
        cal_score = calibrator.transform(raw_score)

        if case["status"] == "TRUE_POSITIVE":
            rank_ok  = (
                found_rank is not None
                and case["expected_rank_top_n"] > 0
                and found_rank <= case["expected_rank_top_n"]
            )
            score_ok = raw_score >= case["min_score"]
            passed   = rank_ok or score_ok

            if passed:
                result_status = "found"
            else:
                result_status = "false_negative"
                logger.warning(
                    f"    FAIL: {drug_name} score {raw_score:.3f} < expected "
                    f"{case['min_score']} AND rank {found_rank} > {case['expected_rank_top_n']}"
                )
        else:  # TRUE_NEGATIVE
            rank_ok  = False
            score_ok = raw_score < case["min_score"]
            passed   = score_ok
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
        "rank_pass":             rank_ok,
        "score_pass":            score_ok,
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
    """Run all validation cases and write results JSON."""
    pipeline   = ProductionPipeline()
    calibrator = load_calibrator()
    start_utc  = datetime.now(timezone.utc)

    logger.info("=" * 70)
    logger.info(f"VALIDATION RUN — Dataset {DATASET_VERSION} — {N_TEST_CASES} cases")
    logger.info("=" * 70)

    try:
        logger.info("\nFetching approved drugs (shared across all test cases)...")
        drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
        logger.info(f"Using {len(drugs_data)} drugs for all tests\n")

        positive_cases = get_positive_cases()
        negative_cases = get_negative_cases()

        logger.info(f"Running {len(positive_cases)} TRUE_POSITIVE cases...")
        positive_results = []
        for case in positive_cases:
            result = await run_single_validation_case(pipeline, drugs_data, case, calibrator)
            positive_results.append(result)

        logger.info(f"\nRunning {len(negative_cases)} TRUE_NEGATIVE cases...")
        negative_results = []
        for case in negative_cases:
            result = await run_single_validation_case(pipeline, drugs_data, case, calibrator)
            negative_results.append(result)

    finally:
        await pipeline.close()

    n_pos = len(positive_results)
    n_neg = len(negative_results)
    tp    = sum(1 for r in positive_results if r["pass"])
    fn    = n_pos - tp
    tn    = sum(1 for r in negative_results if r["pass"])
    fp    = n_neg - tn

    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0 else 0.0
    )

    rank_only_passes = sum(
        1 for r in positive_results
        if r.get("rank_pass") and not r.get("score_pass") and r["pass"]
    )
    score_only_passes = sum(
        1 for r in positive_results
        if r.get("score_pass") and not r.get("rank_pass") and r["pass"]
    )
    both_passes = sum(
        1 for r in positive_results
        if r.get("rank_pass") and r.get("score_pass") and r["pass"]
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
    logger.info(
        f"  Pass breakdown:  rank-only={rank_only_passes} "
        f"score-only={score_only_passes} both={both_passes}"
    )

    output = {
        "header": {
            "dataset_version":   DATASET_VERSION,
            "n_test_cases":      N_TEST_CASES,
            "n_positive_cases":  n_pos,
            "n_negative_cases":  n_neg,
            "run_timestamp_utc": start_utc.isoformat(),
            "elapsed_seconds":   round(elapsed, 2),
            "min_score_used":    min_score,
            "pass_criterion":    "rank_ok OR score_ok (v3.1)",
        },
        "metrics": {
            "tp":                   tp,
            "fn":                   fn,
            "tn":                   tn,
            "fp":                   fp,
            "sensitivity":          round(sensitivity, 4),
            "specificity":          round(specificity, 4),
            "precision":            round(precision, 4),
            "f1":                   round(f1, 4),
            "rank_only_passes":     rank_only_passes,
            "score_only_passes":    score_only_passes,
            "both_criteria_passes": both_passes,
        },
        "positive_results": positive_results,
        "negative_results": negative_results,
        "test_cases":     positive_results,
        "negative_cases": negative_results,
        "out_of_scope_cases": [
            {
                "drug":            c["drug"],
                "disease":         c["disease"],
                "reason":          c["reason"],
                "version_removed": c["removed_in_version"],
            }
            for c in OUT_OF_SCOPE_CASES
        ],
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults written to: {out_path.resolve()}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run curated validation suite against the drug repurposing pipeline"
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="validation_results.json")
    args = parser.parse_args()

    try:
        result  = asyncio.run(run_all_validations(
            min_score=args.min_score,
            output_path=args.output,
        ))
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