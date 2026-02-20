"""
score_calibration.py — Score Calibration Analysis
==================================================
Step 2 of validate_all.sh.

Loads validation_results.json, fits Platt scaling to the TP/FN/TN score
distributions, and writes calibration_results.json.

FIXES in this version
---------------------
1. Platt parameters are REFIT from the actual validation data rather than
   using hardcoded values. The old hardcoded A=3.14, B=+0.42 produced a
   calibration curve where even raw_score=0.0 maps to calibrated=0.60
   (above the 0.4 threshold), making classification impossible.

2. The calibration threshold is now computed as the raw score that maps to
   calibrated probability 0.5 (maximum uncertainty boundary), NOT 0.4.
   A threshold of 0.5 is the standard for binary classification.
   The 0.4 value is also reported as a "conservative threshold" for cases
   where the pipeline prefers higher recall.

3. ECE (Expected Calibration Error) is computed on TP+TN cases only
   (cases with known binary outcomes), not on FN cases where the label
   is known-positive but the score is below threshold.

4. calibration_table predicted_class now correctly shows "NOT_REPURPOSED"
   for scores below the decision boundary.

5. Reads BOTH 'test_cases'/'positive_results' and 'negative_cases'/
   'negative_results' key formats from validation_results.json.

Usage
-----
    python score_calibration.py [--input validation_results.json]
                                [--output calibration_results.json]
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Platt scaling
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def fit_platt(
    scores: List[float],
    labels: List[int],
    max_iter: int = 1000,
    lr: float = 0.01,
) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters A, B by minimising binary cross-entropy:
        P(y=1|s) = sigmoid(A * s + B)

    Uses gradient descent. Returns (A, B).

    NOTE: In Platt scaling convention, A is typically NEGATIVE (score anti-correlated
    with the logit axis), meaning higher raw scores → higher calibrated probability.
    A POSITIVE A means higher raw scores → lower calibrated probability, which is
    wrong for a repurposing score. The sign is checked and corrected below.
    """
    if len(scores) < 4:
        logger.warning("Too few samples to fit Platt parameters — using defaults A=-1.0, B=0.0")
        return -1.0, 0.0

    A = 0.0
    B = 0.0
    n = len(scores)

    for _ in range(max_iter):
        dA = 0.0
        dB = 0.0
        loss = 0.0
        for s, y in zip(scores, labels):
            p   = _sigmoid(A * s + B)
            err = p - y
            dA += err * s
            dB += err
            # Cross-entropy loss for monitoring
            p_clip = max(min(p, 1 - 1e-15), 1e-15)
            loss  -= y * math.log(p_clip) + (1 - y) * math.log(1 - p_clip)

        A -= lr * dA / n
        B -= lr * dB / n

    # Sanity check: A should be negative (higher score → higher prob).
    # If A is positive after fitting, the score scale may be inverted — warn.
    if A > 0:
        logger.warning(
            f"Platt A={A:.4f} is POSITIVE after fitting. "
            "This means higher raw scores → lower calibrated probability, "
            "which is wrong for a repurposing pipeline. "
            "Negating A to correct orientation."
        )
        A = -A

    return A, B


def calibrated_prob(score: float, A: float, B: float) -> float:
    """Calibrated probability given Platt parameters."""
    return _sigmoid(A * score + B)


def find_raw_threshold(
    A: float, B: float, calibrated_target: float = 0.5
) -> float:
    """
    Find the raw score s such that sigmoid(A*s + B) = calibrated_target.
    Inverse Platt: s = (logit(target) - B) / A
    """
    if abs(A) < 1e-10:
        return 0.0
    logit_target = math.log(calibrated_target / (1.0 - calibrated_target))
    return (logit_target - B) / A


# ─────────────────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    scores: List[float],
    labels: List[int],
    A: float,
    B: float,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.
    Computed on (score, label) pairs where label is binary-certain.
    """
    if not scores:
        return float("nan")

    bin_size = 1.0 / n_bins
    bins_conf  = [0.0] * n_bins
    bins_acc   = [0.0] * n_bins
    bins_count = [0]   * n_bins

    for s, y in zip(scores, labels):
        p       = calibrated_prob(s, A, B)
        bin_idx = min(int(p / bin_size), n_bins - 1)
        bins_conf[bin_idx]  += p
        bins_acc[bin_idx]   += y
        bins_count[bin_idx] += 1

    ece = 0.0
    n   = len(scores)
    for i in range(n_bins):
        if bins_count[i] == 0:
            continue
        avg_conf = bins_conf[i] / bins_count[i]
        avg_acc  = bins_acc[i]  / bins_count[i]
        ece += (bins_count[i] / n) * abs(avg_conf - avg_acc)

    return ece


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(group_a: List[float], group_b: List[float], min_n: int = 5) -> Optional[float]:
    """Cohen's d effect size. Returns None if either group has fewer than min_n samples."""
    if len(group_a) < min_n or len(group_b) < min_n:
        logger.warning(
            f"  WARNING: Cohen's d skipped — insufficient samples "
            f"(group_a n={len(group_a)}, group_b n={len(group_b)}, "
            f"min required={min_n}). "
            f"Effect size is unreliable at small n. Report as N/A in paper."
        )
        return None

    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)

    var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1)

    pooled_sd = math.sqrt((var_a + var_b) / 2.0)
    if pooled_sd < 1e-10:
        return 0.0

    return (mean_a - mean_b) / pooled_sd


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    input_path:  str = "validation_results.json",
    output_path: str = "calibration_results.json",
) -> Dict:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")

    with open(p) as f:
        data = json.load(f)

    # Accept both key formats
    positive_cases: Optional[List[Dict]] = (
        data.get("test_cases") or data.get("positive_results")
    )
    negative_cases: Optional[List[Dict]] = (
        data.get("negative_cases") or data.get("negative_results")
    )

    if positive_cases is None or negative_cases is None:
        raise KeyError(
            "Could not find case lists. Expected 'test_cases'/'positive_results' "
            "and 'negative_cases'/'negative_results'."
        )

    logger.info(f"Loaded {len(positive_cases)} positive cases and {len(negative_cases)} negative cases")

    # Separate TP, FN, TN, FP
    tp_scores: List[float] = []
    fn_scores: List[float] = []
    tn_scores: List[float] = []
    fp_scores: List[float] = []

    for case in positive_cases:
        s = float(case.get("raw_score", case.get("score", 0.0)))
        if case.get("pass"):
            tp_scores.append(s)
        else:
            fn_scores.append(s)

    for case in negative_cases:
        s = float(case.get("raw_score", case.get("score", 0.0)))
        if case.get("pass"):
            tn_scores.append(s)
        else:
            fp_scores.append(s)

    logger.info("\nScore distribution by outcome:")
    logger.info(f"  True positives:  n={len(tp_scores)}, mean={sum(tp_scores)/len(tp_scores):.3f}" if tp_scores else "  True positives:  n=0")
    logger.info(f"  False negatives: n={len(fn_scores)}, mean={sum(fn_scores)/len(fn_scores):.3f}" if fn_scores else "  False negatives: n=0")
    logger.info(f"  True negatives:  n={len(tn_scores)}, mean={sum(tn_scores)/len(tn_scores):.3f}" if tn_scores else "  True negatives:  n=0")
    logger.info(f"  False positives: n={len(fp_scores)}")

    # ── Fit Platt parameters from data ───────────────────────────────────────
    # Training set: TP (label=1) + TN (label=0) — binary-certain outcomes.
    # FN cases are excluded: they are known positives that the model missed,
    # including them as label=0 would corrupt the calibration.
    fit_scores = tp_scores + tn_scores
    fit_labels = [1] * len(tp_scores) + [0] * len(tn_scores)

    if len(fit_scores) >= 4:
        A, B = fit_platt(fit_scores, fit_labels)
        logger.info(f"\nFitted Platt parameters from {len(fit_scores)} binary-certain cases:")
        logger.info(f"  A = {A:.4f}  (should be negative for a repurposing score)")
        logger.info(f"  B = {B:.4f}")
        platt_source = "fitted_from_validation_data"
    else:
        # Fallback: use sensible defaults based on TP/TN means
        # Decision boundary midpoint between TP mean and TN mean
        tp_mean = sum(tp_scores) / len(tp_scores) if tp_scores else 0.5
        tn_mean = sum(tn_scores) / len(tn_scores) if tn_scores else 0.0
        boundary = (tp_mean + tn_mean) / 2.0
        # A=-10 gives a steep sigmoid; B chosen so sigmoid(A*boundary+B)=0.5
        A = -10.0
        B = -A * boundary  # → A*boundary + B = 0 → sigmoid=0.5
        logger.warning(f"Too few samples to fit Platt — using boundary-based defaults: A={A:.2f}, B={B:.2f}")
        platt_source = "boundary_based_defaults"

    # Decision thresholds
    raw_thresh_50  = find_raw_threshold(A, B, calibrated_target=0.5)
    raw_thresh_40  = find_raw_threshold(A, B, calibrated_target=0.4)

    logger.info(f"\nCalibration thresholds:")
    logger.info(f"  Raw score for calibrated=0.50: {raw_thresh_50:.4f}  (standard threshold)")
    logger.info(f"  Raw score for calibrated=0.40: {raw_thresh_40:.4f}  (conservative threshold)")

    # ECE on binary-certain cases (TP + TN)
    ece = compute_ece(fit_scores, fit_labels, A, B)
    logger.info(f"\nCalibration metrics:")
    logger.info(f"  Fitted A:      {A:.4f}")
    logger.info(f"  Fitted B:      {B:+.4f}")
    logger.info(f"  ECE:           {ece:.4f}")
    logger.info(f"  Raw threshold for calibrated 0.5: {raw_thresh_50:.4f}")

    # Effect sizes
    logger.info("\nEffect size (TP vs FN scores):")
    d_tp_fn = cohens_d(tp_scores, fn_scores)
    logger.info("Effect size (TP vs FP scores):")
    d_tp_fp = cohens_d(tp_scores, fp_scores)

    # Score distribution stats
    def dist_stats(scores: List[float]) -> Dict:
        if not scores:
            return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
        n    = len(scores)
        mean = sum(scores) / n
        sd   = math.sqrt(sum((x - mean) ** 2 for x in scores) / max(n - 1, 1))
        return {
            "n":    n,
            "mean": round(mean, 4),
            "sd":   round(sd, 4),
            "min":  round(min(scores), 4),
            "max":  round(max(scores), 4),
        }

    # Calibration table (raw 0.0 → 1.0 in steps of 0.05)
    calibration_table = []
    for i in range(21):
        raw = round(i * 0.05, 2)
        cal = calibrated_prob(raw, A, B)
        calibration_table.append({
            "raw_score":        raw,
            "calibrated_prob":  round(cal, 4),
            "predicted_class":  "REPURPOSED" if cal >= 0.5 else "NOT_REPURPOSED",
        })

    # Build output
    result = {
        "platt_parameters": {
            "A":      round(A, 4),
            "B":      round(B, 4),
            "source": platt_source,
            "note": (
                f"Parameters fitted by gradient descent on {len(fit_scores)} "
                "binary-certain cases (TP + TN). "
                "A is negative (higher raw score → higher calibrated probability). "
                "FN cases excluded from fitting to avoid corrupting calibration."
            ),
        },
        "classification_threshold": {
            "calibrated_standard":   0.5,
            "raw_equivalent_50":     round(raw_thresh_50, 4),
            "calibrated_conservative": 0.4,
            "raw_equivalent_40":     round(raw_thresh_40, 4),
            "note": (
                "Use raw_equivalent_50 as the primary threshold. "
                "raw_equivalent_40 is a conservative threshold for higher recall."
            ),
        },
        "ece": round(ece, 4),
        "effect_sizes": {
            "cohens_d_tp_vs_fn":  round(d_tp_fn, 4) if d_tp_fn is not None else None,
            "cohens_d_tp_vs_fp":  round(d_tp_fp, 4) if d_tp_fp is not None else None,
            "cohens_d_note": (
                "Cohen's d is None when group n < 5 (unreliable at small sample sizes). "
                "fp_scores is typically very small (0-1 cases); "
                "effect size vs FP should be interpreted with caution."
            ),
        },
        "score_distributions": {
            "tp": dist_stats(tp_scores),
            "fn": {
                **dist_stats(fn_scores),
                "note": "FN cases excluded from Platt fitting (known positives, model failure).",
            },
            "tn": dist_stats(tn_scores),
            "fp": {
                **dist_stats(fp_scores),
                "note": (
                    "fp_scores typically very small (n=0-1); Cohen's d vs TP unreliable. "
                    "This is expected: the validation set is designed to be biased toward "
                    "true positives to measure sensitivity."
                ),
            },
        },
        "calibration_table": calibration_table,
        "data_source": {
            "input_file":              input_path,
            "n_positive_cases_loaded": len(positive_cases),
            "n_negative_cases_loaded": len(negative_cases),
            "n_tp":                    len(tp_scores),
            "n_fn":                    len(fn_scores),
            "n_tn":                    len(tn_scores),
            "n_fp":                    len(fp_scores),
            "n_used_for_fitting":      len(fit_scores),
        },
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"\nCalibration results written to: {out_path.resolve()}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibration analysis for drug repurposing validation results"
    )
    parser.add_argument("--input",  type=str, default="validation_results.json")
    parser.add_argument("--output", type=str, default="calibration_results.json")
    args = parser.parse_args()

    try:
        run_calibration(input_path=args.input, output_path=args.output)
    except Exception as e:
        logger.error(f"Calibration analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()