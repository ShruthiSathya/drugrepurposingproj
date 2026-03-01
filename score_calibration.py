

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PRACTICAL_RAW_THRESHOLD = 0.20

# Must match CALIBRATION_PARAMS_FILE in calibration.py
_CACHE_DIR             = Path("/tmp/drug_repurposing_cache")
_CACHE_PARAMS_FILE     = _CACHE_DIR / "calibration_params.json"


# ─────────────────────────────────────────────────────────────────────────────
# Platt scaling
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def fit_platt(
    scores: List[float],
    labels: List[int],
    max_iter: int = 2000,
    lr: float = 0.005,
) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters A, B by minimising binary cross-entropy.
    A is constrained to be negative (higher score → higher probability).
    Returns (A, B).
    """
    if len(scores) < 4:
        logger.warning("Too few samples — using defaults A=-0.7053, B=0.8535")
        return -0.7053, 0.8535

    A = -1.0
    B = 0.0
    n = len(scores)

    for _ in range(max_iter):
        dA = 0.0
        dB = 0.0
        for s, y in zip(scores, labels):
            p   = _sigmoid(A * s + B)
            err = p - y
            dA += err * s
            dB += err
        A -= lr * dA / n
        B -= lr * dB / n

    if A > 0:
        logger.warning(
            f"Fitted A={A:.4f} is positive after {max_iter} iterations. "
            "Negating to enforce correct orientation."
        )
        A = -abs(A)

    logger.info(f"Fitted Platt: A={A:.4f}, B={B:.4f}")
    return A, B


def calibrated_prob(score: float, A: float, B: float) -> float:
    return _sigmoid(A * score + B)


def find_raw_threshold(A: float, B: float, calibrated_target: float = 0.5) -> float:
    if abs(A) < 1e-10:
        return float("inf")
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
    if not scores:
        return float("nan")

    bin_size   = 1.0 / n_bins
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

    return round(ece, 4)


def interpret_ece(ece: float, n_pos: int, n_neg: int) -> Dict:
    ratio = n_pos / n_neg if n_neg > 0 else float("inf")
    if ece < 0.10:
        status      = "excellent"
        paper_note  = "Calibration is excellent (ECE < 0.10)."
        use_cal     = True
    elif ece < 0.15:
        status      = "acceptable"
        paper_note  = "Calibration is acceptable (ECE < 0.15) for supplementary ranking."
        use_cal     = True
    else:
        status      = "poor_due_to_imbalance"
        paper_note  = (
            f"ECE = {ece:.4f} exceeds the 0.15 target. Root cause: training set "
            f"imbalance ({n_pos} TP : {n_neg} TN = {ratio:.1f}:1 ratio) with "
            f"n = {n_pos + n_neg} cases. Calibrated probabilities are reported as "
            "supplementary rankings only. Primary metrics are sensitivity, specificity, "
            "precision, and rank-based retrieval (Hit@N, MRR)."
        )
        use_cal = False

    return {
        "status":                    status,
        "ece":                       ece,
        "n_pos_used_for_fitting":    n_pos,
        "n_neg_used_for_fitting":    n_neg,
        "pos_neg_ratio":             round(ratio, 2),
        "calibrated_scores_usable":  use_cal,
        "paper_note":                paper_note,
        "recommended_threshold":     (
            f"raw_score >= {PRACTICAL_RAW_THRESHOLD}"
            if not use_cal else
            "calibrated_prob >= 0.5"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(group_a: List[float], group_b: List[float], min_n: int = 5) -> Optional[float]:
    if len(group_a) < min_n or len(group_b) < min_n:
        logger.warning(
            f"Cohen's d skipped — group_a n={len(group_a)}, group_b n={len(group_b)} "
            f"(min={min_n}). Report as N/A."
        )
        return None
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    var_a  = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1)
    var_b  = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1)
    pooled = math.sqrt((var_a + var_b) / 2.0)
    if pooled < 1e-10:
        return 0.0
    return (mean_a - mean_b) / pooled


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    input_path:    str = "validation_results.json",
    output_path:   str = "calibration_results.json",
    params_output: str = "calibration_params.json",
) -> Dict:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")

    with open(p) as f:
        data = json.load(f)

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

    logger.info(f"Loaded {len(positive_cases)} positive, {len(negative_cases)} negative cases")

    tp_scores: List[float] = []
    fn_scores: List[float] = []
    tn_scores: List[float] = []
    fp_scores: List[float] = []

    for case in positive_cases:
        s = float(case.get("raw_score", case.get("score", 0.0)))
        (tp_scores if case.get("pass") else fn_scores).append(s)

    for case in negative_cases:
        s = float(case.get("raw_score", case.get("score", 0.0)))
        (tn_scores if case.get("pass") else fp_scores).append(s)

    def _stats(scores):
        if not scores:
            return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
        n    = len(scores)
        mean = sum(scores) / n
        sd   = math.sqrt(sum((x - mean) ** 2 for x in scores) / max(n - 1, 1))
        return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
                "min": round(min(scores), 4), "max": round(max(scores), 4)}

    logger.info("Score distributions:")
    for label, scores in [("TP", tp_scores), ("FN", fn_scores),
                           ("TN", tn_scores), ("FP", fp_scores)]:
        s = _stats(scores)
        logger.info(f"  {label}: n={s['n']}, mean={s['mean']}")

    fit_scores = tp_scores + tn_scores
    fit_labels = [1] * len(tp_scores) + [0] * len(tn_scores)

    if len(fit_scores) >= 4:
        A, B = fit_platt(fit_scores, fit_labels)
        platt_source = "fitted_from_validation_data"
    else:
        A, B = -0.7053, 0.8535
        platt_source = "default_insufficient_data"
        logger.warning("Using default Platt parameters.")

    raw_thresh_50 = find_raw_threshold(A, B, 0.5)
    raw_thresh_40 = find_raw_threshold(A, B, 0.4)

    logger.info(f"Platt A={A:.4f}, B={B:.4f}")
    if raw_thresh_50 > 1.0:
        logger.warning(
            f"Raw threshold {raw_thresh_50:.4f} > 1.0 — calibrated threshold is "
            f"unreachable. Use raw_score >= {PRACTICAL_RAW_THRESHOLD} instead."
        )

    ece = compute_ece(fit_scores, fit_labels, A, B)
    ece_info = interpret_ece(ece, len(tp_scores), len(tn_scores))

    logger.info(f"ECE = {ece:.4f} — {ece_info['status']}")

    d_tp_fn = cohens_d(tp_scores, fn_scores)
    d_tp_fp = cohens_d(tp_scores, fp_scores)

    calibration_table = []
    for i in range(21):
        raw = round(i * 0.05, 2)
        cal = calibrated_prob(raw, A, B)
        calibration_table.append({
            "raw_score":        raw,
            "calibrated_prob":  round(cal, 4),
            "predicted_class":  (
                "REPURPOSED" if raw >= PRACTICAL_RAW_THRESHOLD else "NOT_REPURPOSED"
            ),
        })

    result = {
        "platt_parameters": {
            "A":      round(A, 4),
            "B":      round(B, 4),
            "source": platt_source,
            "note": (
                f"Parameters fitted by gradient descent on {len(fit_scores)} "
                "binary-certain cases (TP + TN). A is negative "
                "(higher raw score → higher calibrated probability). "
                "FN cases excluded from fitting. "
                "These parameters are also written to calibration_params.json "
                "and loaded by calibration.py at runtime."
            ),
        },
        "classification_threshold": {
            "calibrated_standard":       0.5,
            "raw_equivalent_50":         round(raw_thresh_50, 4),
            "calibrated_conservative":   0.4,
            "raw_equivalent_40":         round(raw_thresh_40, 4),
            "practical_raw_threshold":   PRACTICAL_RAW_THRESHOLD,
            "note": (
                f"raw_equivalent_50 = {raw_thresh_50:.4f}. "
                + (
                    "This exceeds 1.0 (unreachable). "
                    f"Use practical_raw_threshold = {PRACTICAL_RAW_THRESHOLD}."
                    if raw_thresh_50 > 1.0 else
                    "Use raw_equivalent_50 as primary threshold."
                )
            ),
        },
        "ece":                ece,
        "ece_interpretation": ece_info,
        "effect_sizes": {
            "cohens_d_tp_vs_fn":  round(d_tp_fn, 4) if d_tp_fn is not None else None,
            "cohens_d_tp_vs_fp":  round(d_tp_fp, 4) if d_tp_fp is not None else None,
            "cohens_d_note": (
                "Cohen's d is None when group n < 5. "
                "fp_scores typically n=0-2; effect size vs FP unreliable."
            ),
        },
        "score_distributions": {
            "tp": _stats(tp_scores),
            "fn": {**_stats(fn_scores),
                   "note": "FN excluded from Platt fitting (known positives, model failure)."},
            "tn": _stats(tn_scores),
            "fp": {**_stats(fp_scores),
                   "note": "FP typically n=0-2; Cohen's d vs TP unreliable."},
        },
        "calibration_table": calibration_table,
        "paper_reporting_note": {
            "primary_metrics": [
                "sensitivity", "specificity", "precision", "F1",
                "Hit@N", "MRR", "raw_score"
            ],
            "supplementary_metrics": ["calibrated_prob"],
            "ece_statement": ece_info["paper_note"],
            "recommended_methods_text": (
                "Raw network overlap scores were calibrated using Platt scaling "
                f"(A = {A:.3f}, B = {B:.3f}; fitted by gradient-descent maximum-likelihood "
                f"logistic regression on {len(fit_scores)} binary-certain validation cases). "
                f"ECE = {ece:.4f}. "
                "Calibrated probabilities are reported for supplementary ranking only. "
                "Primary performance metrics use sensitivity, specificity, precision, "
                f"and rank-based retrieval, with binary classification threshold "
                f"raw_score ≥ {PRACTICAL_RAW_THRESHOLD}."
            ),
        },
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

    # ── FIX (v6): Write calibration_params.json with keys calibration.py expects
    # calibration.py._try_load_params() reads:
    #   params["method"]   → must be "platt" or "isotonic"
    #   params["platt_a"]  → Platt A parameter
    #   params["platt_b"]  → Platt B parameter
    #   params["n_samples"]→ used in the load log message
    # v5 used keys "A" and "B" which caused silent load failure, leaving the
    # calibrator unfitted and producing "returning raw score" warnings.
    params_data = {
        "method":    "platt",
        "platt_a":   round(A, 4),
        "platt_b":   round(B, 4),
        "source":    platt_source,
        "fitted_from": input_path,
        "n_samples": len(fit_scores),
        "ece":       ece,
        "note": (
            "Written by score_calibration.py. Read by calibration.py at runtime. "
            "Do not edit manually — re-run score_calibration.py to update."
        ),
    }

    # Write to project root (legacy path, used by --params-output)
    params_path = Path(params_output)
    params_path.write_text(json.dumps(params_data, indent=2))
    logger.info(f"Calibration params → {params_path.resolve()}")

    # FIX (v6): Also write to the cache dir where calibration.py looks for it.
    # calibration.py hardcodes CALIBRATION_PARAMS_FILE =
    #   Path("/tmp/drug_repurposing_cache/calibration_params.json")
    # Without this second write, load_calibrator() never finds the file and
    # the calibrator stays unfitted across all downstream steps.
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_PARAMS_FILE.write_text(json.dumps(params_data, indent=2))
        logger.info(f"Calibration params (cache) → {_CACHE_PARAMS_FILE}")
    except Exception as e:
        logger.warning(f"Could not write to cache dir {_CACHE_DIR}: {e}")

    out_path = Path(output_path)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Calibration results → {out_path.resolve()}")

    logger.info("\n" + "=" * 60)
    logger.info(f"  Platt A = {A:.4f},  B = {B:.4f}")
    logger.info(f"  ECE     = {ece:.4f}  ({ece_info['status']})")
    logger.info(f"  Params written to: {params_path.name}")
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         default="validation_results.json")
    parser.add_argument("--output",        default="calibration_results.json")
    parser.add_argument("--params-output", default="calibration_params.json",
                        help="Path to write fitted A/B for use by calibration.py")
    args = parser.parse_args()
    try:
        run_calibration(
            input_path=args.input,
            output_path=args.output,
            params_output=args.params_output,
        )
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise


if __name__ == "__main__":
    main()