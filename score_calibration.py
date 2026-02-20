"""
score_calibration.py — Post-Run Calibration Analysis
======================================================
Reads validation_results.json and computes/verifies Platt calibration
parameters, Cohen's d effect size, and Expected Calibration Error.

This script is Step 2 of validate_all.sh.

FIXES vs previous version
--------------------------
1. Key flexibility: Now accepts BOTH key formats from validation_results.json:
   - 'test_cases' / 'negative_cases'        (expected by this script)
   - 'positive_results' / 'negative_results' (legacy keys from run_validation.py)
   This means validate_all.sh Step 2 no longer silently fails even if
   run_validation.py uses the legacy key names.

2. Cohen's d guard: Skips Cohen's d calculation when either group has fewer
   than 5 samples (n<5 makes d unreliable). Reports a warning and sets d=None
   rather than returning a misleading number.

3. Status string matching: Uses substring matching for true_negative detection
   ('true_negative' in status) which correctly matches both:
   - 'true_negative_low_score'
   - 'true_negative_not_found'
   This was already partially correct but is now explicit and tested.

Usage
-----
    python score_calibration.py [--input validation_results.json]
                                [--output calibration_results.json]
"""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.pipeline.calibration import ScoreCalibrator, _PLATT_A, _PLATT_B


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_results(path: str) -> Dict:
    """Load validation_results.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Validation results not found: {p.resolve()}")
    with open(p) as f:
        return json.load(f)


def _extract_cases(data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract positive and negative case lists from validation_results.json.

    FIX 1: Accept both key formats.
    run_validation.py outputs BOTH formats but older versions only output
    'positive_results' / 'negative_results'. This function handles either.

    Priority: 'test_cases' > 'positive_results'
              'negative_cases' > 'negative_results'
    """
    positive_cases: Optional[List[Dict]] = (
        data.get("test_cases")
        or data.get("positive_results")
    )
    negative_cases: Optional[List[Dict]] = (
        data.get("negative_cases")
        or data.get("negative_results")
    )

    if positive_cases is None:
        raise KeyError(
            "validation_results.json must contain 'test_cases' or 'positive_results'. "
            "Found keys: " + str(list(data.keys()))
        )
    if negative_cases is None:
        raise KeyError(
            "validation_results.json must contain 'negative_cases' or 'negative_results'. "
            "Found keys: " + str(list(data.keys()))
        )

    print(f"Loaded {len(positive_cases)} positive cases and {len(negative_cases)} negative cases")
    return positive_cases, negative_cases


def _get_score(case: Dict) -> float:
    """Get the raw score from a case dict (try both key names)."""
    return float(case.get("raw_score", case.get("score", 0.0)))


def _is_true_positive(case: Dict) -> bool:
    """Check if a case was correctly identified (status 'found' or expected TRUE_POSITIVE)."""
    status = case.get("status", "")
    return status == "found" or case.get("expected_status", "") == "TRUE_POSITIVE"


def _is_true_negative(case: Dict) -> bool:
    """
    Check if a case is a true negative.

    FIX 3: Uses 'true_negative' in status (substring) which matches both:
      - 'true_negative_low_score'
      - 'true_negative_not_found'
    Explicitly verified against actual status strings from run_validation.py.
    """
    status = case.get("status", "")
    return "true_negative" in status.lower()


def _cohen_d(
    group_a: List[float],
    group_b: List[float],
    min_n: int = 5,
) -> Optional[float]:
    """
    Compute Cohen's d effect size (pooled SD version).

    FIX 2: Returns None with a warning if either group has fewer than `min_n`
    samples (default 5). Unreliable effect sizes with n<5 should not be
    reported in the paper.

    Parameters
    ----------
    group_a : list of float
        Scores for group A (e.g. true positives).
    group_b : list of float
        Scores for group B (e.g. true negatives / false negatives).
    min_n : int
        Minimum required samples per group. Default 5.

    Returns
    -------
    float or None
        Cohen's d, or None if either group is too small.
    """
    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < min_n or n_b < min_n:
        print(
            f"  WARNING: Cohen's d skipped — insufficient samples "
            f"(group_a n={n_a}, group_b n={n_b}, min required={min_n}). "
            f"Effect size is unreliable at small n. Report as N/A in paper."
        )
        return None

    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a  = statistics.variance(group_a)
    var_b  = statistics.variance(group_b)

    # Pooled standard deviation
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_sd == 0:
        print("  WARNING: Pooled SD is 0 — all scores identical. Cohen's d undefined.")
        return None

    return (mean_a - mean_b) / pooled_sd


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_analysis(
    input_path:  str = "validation_results.json",
    output_path: str = "calibration_results.json",
) -> Dict:
    """
    Load validation results and compute calibration statistics.
    Writes calibration_results.json.
    """
    print("=" * 70)
    print("SCORE CALIBRATION ANALYSIS")
    print("=" * 70)

    data = _load_results(input_path)
    positive_cases, negative_cases = _extract_cases(data)

    # Collect scores by outcome
    tp_scores:  List[float] = []  # True positives (found, score >= threshold)
    fn_scores:  List[float] = []  # False negatives (missed)
    tn_scores:  List[float] = []  # True negatives (correctly low)
    fp_scores:  List[float] = []  # False positives (should be low but scored high)

    for case in positive_cases:
        score = _get_score(case)
        if case.get("pass", False):
            tp_scores.append(score)
        else:
            fn_scores.append(score)

    for case in negative_cases:
        score = _get_score(case)
        if _is_true_negative(case) and case.get("pass", False):
            tn_scores.append(score)
        else:
            fp_scores.append(score)

    print(f"\nScore distribution by outcome:")
    print(f"  True positives:  n={len(tp_scores)}, mean={statistics.mean(tp_scores):.3f}" if tp_scores else "  True positives:  n=0")
    print(f"  False negatives: n={len(fn_scores)}, mean={statistics.mean(fn_scores):.3f}" if fn_scores else "  False negatives: n=0")
    print(f"  True negatives:  n={len(tn_scores)}, mean={statistics.mean(tn_scores):.3f}" if tn_scores else "  True negatives:  n=0")
    print(f"  False positives: n={len(fp_scores)}, mean={statistics.mean(fp_scores):.3f}" if fp_scores else "  False positives: n=0")

    # Calibrator with current parameters
    calibrator = ScoreCalibrator(A=_PLATT_A, B=_PLATT_B)

    # All positive raw scores and labels for ECE
    all_raw_scores = [_get_score(c) for c in positive_cases + negative_cases]
    all_labels     = (
        [True] * len(positive_cases) + [False] * len(negative_cases)
    )
    ece = calibrator.expected_calibration_error(all_raw_scores, all_labels)

    # Cohen's d: TP vs FN (do positives score higher than false negatives?)
    print("\nEffect size (TP vs FN scores):")
    d_tp_fn = _cohen_d(tp_scores, fn_scores, min_n=5)
    if d_tp_fn is not None:
        print(f"  Cohen's d (TP vs FN) = {d_tp_fn:.3f}")

    # Cohen's d: TP vs FP (do positives separate from false positives?)
    print("Effect size (TP vs FP scores):")
    d_tp_fp = _cohen_d(tp_scores, fp_scores, min_n=5)
    if d_tp_fp is not None:
        print(f"  Cohen's d (TP vs FP) = {d_tp_fp:.3f}")

    # Score distribution stats
    all_tp_fn_scores = tp_scores + fn_scores  # All positive-class raw scores
    all_neg_scores   = tn_scores + fp_scores   # All negative-class raw scores

    print(f"\nCalibration metrics:")
    print(f"  Platt A:      {calibrator.A}")
    print(f"  Platt B:      +{calibrator.B}  (positive — corrected from old -0.42)")
    print(f"  ECE:          {ece}")
    print(f"  Raw threshold for calibrated {calibrator.threshold}: "
          f"{calibrator.raw_threshold():.4f}")

    # Calibration table (for paper figure)
    calibration_table = calibrator.calibration_table()

    output = {
        "platt_parameters": {
            "A": calibrator.A,
            "B": calibrator.B,
            "note": (
                "B is POSITIVE (+0.42). Previous versions incorrectly used -0.42. "
                "The positive value reflects the calibration curve showing "
                "high observed positive rate (~57.6%) even at raw score ~0.10."
            ),
        },
        "classification_threshold": {
            "calibrated": calibrator.threshold,
            "raw_equivalent": round(calibrator.raw_threshold(), 4),
        },
        "ece": ece,
        "effect_sizes": {
            "cohens_d_tp_vs_fn": round(d_tp_fn, 4) if d_tp_fn is not None else None,
            "cohens_d_tp_vs_fp": round(d_tp_fp, 4) if d_tp_fp is not None else None,
            "cohens_d_note": (
                "Cohen's d is None when group n < 5 (unreliable at small sample sizes). "
                "fp_scores is typically very small (0-1 cases); effect size vs FP "
                "should be interpreted with caution."
            ),
        },
        "score_distributions": {
            "tp": {
                "n":    len(tp_scores),
                "mean": round(statistics.mean(tp_scores), 4) if tp_scores else None,
                "sd":   round(statistics.stdev(tp_scores), 4) if len(tp_scores) > 1 else None,
                "min":  round(min(tp_scores), 4) if tp_scores else None,
                "max":  round(max(tp_scores), 4) if tp_scores else None,
            },
            "fn": {
                "n":    len(fn_scores),
                "mean": round(statistics.mean(fn_scores), 4) if fn_scores else None,
                "sd":   round(statistics.stdev(fn_scores), 4) if len(fn_scores) > 1 else None,
                "min":  round(min(fn_scores), 4) if fn_scores else None,
                "max":  round(max(fn_scores), 4) if fn_scores else None,
            },
            "tn": {
                "n":    len(tn_scores),
                "mean": round(statistics.mean(tn_scores), 4) if tn_scores else None,
                "sd":   round(statistics.stdev(tn_scores), 4) if len(tn_scores) > 1 else None,
                "min":  round(min(tn_scores), 4) if tn_scores else None,
                "max":  round(max(tn_scores), 4) if tn_scores else None,
            },
            "fp": {
                "n":    len(fp_scores),
                "mean": round(statistics.mean(fp_scores), 4) if fp_scores else None,
                "sd":   round(statistics.stdev(fp_scores), 4) if len(fp_scores) > 1 else None,
                "note": (
                    "fp_scores typically very small (n=0-1); Cohen's d vs TP unreliable. "
                    "This is expected: the validation set is designed to be biased toward "
                    "true positives to measure sensitivity."
                ),
            },
        },
        "calibration_table": calibration_table,
        "data_source": {
            "input_file": input_path,
            "n_positive_cases_loaded": len(positive_cases),
            "n_negative_cases_loaded": len(negative_cases),
        },
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nCalibration results written to: {out_path.resolve()}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute calibration statistics from validation results"
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
        default="calibration_results.json",
        help="Output path for calibration_results.json (default: calibration_results.json)",
    )
    args = parser.parse_args()

    try:
        run_calibration_analysis(
            input_path=args.input,
            output_path=args.output,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()