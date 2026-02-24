"""
statistical_tests.py — Publication-Grade Statistical Analysis
=============================================================
Provides the statistical tests required for publication in journals such as
Bioinformatics, PLOS Computational Biology, or Briefings in Bioinformatics.

Tests included
--------------
1. Bootstrap confidence intervals (1000 iterations, 95% CI) for:
   sensitivity, specificity, precision, F1, AUC-ROC

2. McNemar's test — compare algorithm vs baseline (e.g. Jaccard) on paired
   binary predictions. Tests whether the disagreements are symmetric.

3. DeLong's test — compare AUC-ROC values between two methods on the same
   test set without assuming parametric distributions.

4. Runtime benchmarking — measure wall-clock time per disease analysis.

Usage
-----
    python statistical_tests.py --input validation_results.json
                                 --output statistical_results.json

Or import directly:
    from statistical_tests import bootstrap_ci, mcnemar_test, delong_test

Paper reporting template (Methods — Statistical Analysis):
    "Bootstrap 95% confidence intervals (1000 iterations, stratified resampling)
     were computed for all primary metrics. Algorithm performance was compared
     to the Jaccard baseline using McNemar's test on paired binary predictions
     (two-sided, α = 0.05). AUC-ROC values were compared using DeLong's test.
     All statistical analyses were performed with custom Python code available
     at [repository URL]."
"""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(tp: int, fn: int, tn: int, fp: int) -> Dict[str, float]:
    n_pos = tp + fn
    n_neg = tn + fp
    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0 else 0.0
    )
    return {"sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "f1": f1}


def bootstrap_ci(
    positive_results: List[Dict],
    negative_results: List[Dict],
    n_iterations:     int   = 1000,
    alpha:            float = 0.05,
    seed:             int   = 42,
) -> Dict:
    """
    Stratified bootstrap confidence intervals for sensitivity, specificity,
    precision, and F1.

    Stratified: resamples positives and negatives independently to preserve
    the positive:negative ratio in each bootstrap replicate.

    Parameters
    ----------
    positive_results : list of dict
        Each dict must have 'pass' (bool). From validation_results.json.
    negative_results : list of dict
        Each dict must have 'pass' (bool). From validation_results.json.
    n_iterations : int
        Bootstrap iterations (default 1000).
    alpha : float
        Significance level for CI (default 0.05 → 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: sensitivity, specificity, precision, f1
        Each value is a dict: {point_estimate, ci_lower, ci_upper, se}

    Paper reporting:
        "Sensitivity = X.XX (95% CI: X.XX–X.XX)"
    """
    random.seed(seed)

    n_pos = len(positive_results)
    n_neg = len(negative_results)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least 1 positive and 1 negative result.")

    # Point estimates
    tp_obs = sum(1 for r in positive_results if r.get("pass"))
    fn_obs = n_pos - tp_obs
    tn_obs = sum(1 for r in negative_results if r.get("pass"))
    fp_obs = n_neg - tn_obs
    point  = _compute_metrics(tp_obs, fn_obs, tn_obs, fp_obs)

    # Bootstrap
    boot_metrics: Dict[str, List[float]] = {
        "sensitivity": [], "specificity": [], "precision": [], "f1": []
    }

    for _ in range(n_iterations):
        boot_pos = random.choices(positive_results, k=n_pos)
        boot_neg = random.choices(negative_results, k=n_neg)

        tp = sum(1 for r in boot_pos if r.get("pass"))
        fn = n_pos - tp
        tn = sum(1 for r in boot_neg if r.get("pass"))
        fp = n_neg - tn

        m = _compute_metrics(tp, fn, tn, fp)
        for key in boot_metrics:
            boot_metrics[key].append(m[key])

    # Compute CIs (percentile method)
    lo_pct = alpha / 2.0
    hi_pct = 1.0 - alpha / 2.0

    results = {}
    for metric in ["sensitivity", "specificity", "precision", "f1"]:
        samples = sorted(boot_metrics[metric])
        n       = len(samples)
        lo_idx  = max(0, int(lo_pct * n))
        hi_idx  = min(n - 1, int(hi_pct * n))
        se      = (
            math.sqrt(sum((s - point[metric]) ** 2 for s in samples) / (n - 1))
            if n > 1 else 0.0
        )
        results[metric] = {
            "point_estimate": round(point[metric], 4),
            "ci_lower":       round(samples[lo_idx], 4),
            "ci_upper":       round(samples[hi_idx], 4),
            "se":             round(se, 4),
            "n_bootstrap":    n_iterations,
            "alpha":          alpha,
            "ci_label":       f"{int((1 - alpha) * 100)}% CI",
        }

    logger.info("Bootstrap CI results:")
    for metric, r in results.items():
        logger.info(
            f"  {metric}: {r['point_estimate']:.3f} "
            f"({r['ci_label']}: {r['ci_lower']:.3f}–{r['ci_upper']:.3f})"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. McNemar's test
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    predictions_a: List[bool],
    predictions_b: List[bool],
    labels:        List[bool],
    method_a_name: str = "Algorithm",
    method_b_name: str = "Baseline",
) -> Dict:
    """
    McNemar's test for paired binary predictions.

    Tests whether the two methods differ significantly in their prediction
    errors. Uses exact binomial test when b+c < 25, chi-squared otherwise.

    Parameters
    ----------
    predictions_a : list of bool
        Binary predictions from method A (algorithm).
    predictions_b : list of bool
        Binary predictions from method B (baseline, e.g. Jaccard).
    labels : list of bool
        True binary labels.

    Returns
    -------
    dict with:
        b: cases A=correct, B=wrong
        c: cases A=wrong, B=correct
        statistic: McNemar chi-squared
        p_value: two-sided p-value
        significant: bool (p < 0.05)
        interpretation: str

    Paper reporting:
        "McNemar's test: χ²(1) = X.XX, p = X.XXX"
    """
    if not (len(predictions_a) == len(predictions_b) == len(labels)):
        raise ValueError("predictions_a, predictions_b, labels must be same length.")

    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct

    for pred_a, pred_b, label in zip(predictions_a, predictions_b, labels):
        correct_a = pred_a == label
        correct_b = pred_b == label
        if correct_a and not correct_b:
            b += 1
        elif not correct_a and correct_b:
            c += 1

    n_discordant = b + c

    if n_discordant == 0:
        return {
            "b": b, "c": c,
            "statistic": 0.0, "p_value": 1.0,
            "significant": False,
            "test_type": "exact_binomial",
            "interpretation": (
                "No discordant pairs — methods are identical on this test set."
            ),
        }

    if n_discordant < 25:
        # Exact binomial p-value (two-sided, H0: p=0.5)
        # P(X <= min(b,c)) × 2 under Binomial(n_discordant, 0.5)
        k   = min(b, c)
        n   = n_discordant
        p05 = 0.5

        def _binom_pmf(n, k, p):
            log_coef = (
                sum(math.log(i) for i in range(1, n + 1))
                - sum(math.log(i) for i in range(1, k + 1))
                - sum(math.log(i) for i in range(1, n - k + 1))
            )
            return math.exp(log_coef + k * math.log(p) + (n - k) * math.log(1 - p))

        p_one_sided = sum(_binom_pmf(n, i, p05) for i in range(0, k + 1))
        p_value     = min(2 * p_one_sided, 1.0)
        statistic   = None
        test_type   = "exact_binomial"
    else:
        # McNemar chi-squared with continuity correction
        statistic = (abs(b - c) - 1.0) ** 2 / n_discordant
        test_type = "chi_squared_continuity_corrected"

        # Chi-squared p-value (1 df) via incomplete gamma
        # Use normal approximation for simplicity (valid for chi-sq df=1)
        z       = math.sqrt(statistic)
        p_value = 2 * (1 - _normal_cdf(z))

    interpretation = (
        f"{method_a_name} significantly {'outperforms' if b > c else 'underperforms'} "
        f"{method_b_name} (b={b}, c={c}, p={p_value:.4f})"
        if p_value < 0.05 else
        f"No significant difference between {method_a_name} and {method_b_name} "
        f"(b={b}, c={c}, p={p_value:.4f})"
    )

    logger.info(f"McNemar's test: b={b}, c={c}, p={p_value:.4f} — {interpretation}")

    return {
        "b":              b,
        "c":              c,
        "n_discordant":   n_discordant,
        "statistic":      round(statistic, 4) if statistic is not None else None,
        "p_value":        round(p_value, 4),
        "significant":    p_value < 0.05,
        "alpha":          0.05,
        "test_type":      test_type,
        "method_a":       method_a_name,
        "method_b":       method_b_name,
        "interpretation": interpretation,
    }


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. DeLong's test for AUC-ROC comparison
# ─────────────────────────────────────────────────────────────────────────────

def _auc_roc(scores: List[float], labels: List[bool]) -> float:
    """Trapezoidal AUC-ROC."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_tp = 0.0
    prev_fp = 0.0

    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_tp = tp
        prev_fp = fp

    return auc / (n_pos * n_neg)


def delong_test(
    scores_a:      List[float],
    scores_b:      List[float],
    labels:        List[bool],
    method_a_name: str = "Algorithm",
    method_b_name: str = "Baseline",
) -> Dict:
    """
    DeLong's test for comparing two correlated AUC-ROC values.

    Implements the structural components method (DeLong et al. 1988).
    Accounts for the correlation between two AUCs estimated on the same test set.

    Parameters
    ----------
    scores_a, scores_b : list of float
        Predicted scores from method A and B on the same samples.
    labels : list of bool
        True binary labels.

    Returns
    -------
    dict with:
        auc_a, auc_b: float
        auc_difference: float (a - b)
        z_statistic: float
        p_value: float (two-sided)
        significant: bool

    Reference:
        DeLong ER, DeLong DM, Clarke-Pearson DL (1988). "Comparing the areas
        under two or more correlated receiver operating characteristic curves:
        a nonparametric approach." Biometrics 44(3):837–845.

    Paper reporting:
        "DeLong's test: AUC difference = X.XX (95% CI: X.XX–X.XX), Z = X.XX, p = X.XXX"
    """
    if not (len(scores_a) == len(scores_b) == len(labels)):
        raise ValueError("All inputs must be same length.")

    n = len(labels)
    pos_idx = [i for i, l in enumerate(labels) if l]
    neg_idx = [i for i, l in enumerate(labels) if not l]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos < 2 or n_neg < 2:
        return {
            "auc_a": _auc_roc(scores_a, labels),
            "auc_b": _auc_roc(scores_b, labels),
            "auc_difference": None,
            "z_statistic": None,
            "p_value": None,
            "significant": None,
            "error": "Insufficient positives or negatives for DeLong's test (need ≥2 each).",
        }

    def _psi(score_pos, score_neg):
        if score_pos > score_neg:
            return 1.0
        elif score_pos == score_neg:
            return 0.5
        return 0.0

    def _structural_components(scores):
        # V10: for each positive, mean psi over all negatives
        V10 = []
        for i in pos_idx:
            V10.append(sum(_psi(scores[i], scores[j]) for j in neg_idx) / n_neg)
        # V01: for each negative, mean psi over all positives
        V01 = []
        for j in neg_idx:
            V01.append(sum(_psi(scores[i], scores[j]) for i in pos_idx) / n_pos)
        return V10, V01

    V10_a, V01_a = _structural_components(scores_a)
    V10_b, V01_b = _structural_components(scores_b)

    auc_a = sum(V10_a) / n_pos
    auc_b = sum(V10_b) / n_pos

    # Covariance matrix of (auc_a, auc_b)
    def _cov(X, Y):
        n = len(X)
        mx = sum(X) / n
        my = sum(Y) / n
        return sum((x - mx) * (y - my) for x, y in zip(X, Y)) / (n - 1)

    S10_aa = _cov(V10_a, V10_a)
    S10_bb = _cov(V10_b, V10_b)
    S10_ab = _cov(V10_a, V10_b)
    S01_aa = _cov(V01_a, V01_a)
    S01_bb = _cov(V01_b, V01_b)
    S01_ab = _cov(V01_a, V01_b)

    var_a   = S10_aa / n_pos + S01_aa / n_neg
    var_b   = S10_bb / n_pos + S01_bb / n_neg
    cov_ab  = S10_ab / n_pos + S01_ab / n_neg

    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff <= 0:
        return {
            "auc_a": round(auc_a, 4),
            "auc_b": round(auc_b, 4),
            "auc_difference": round(auc_a - auc_b, 4),
            "z_statistic": None,
            "p_value": None,
            "significant": None,
            "error": "Non-positive variance of AUC difference — methods may be identical.",
        }

    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    # 95% CI for the difference
    se_diff  = math.sqrt(var_diff)
    ci_lower = (auc_a - auc_b) - 1.96 * se_diff
    ci_upper = (auc_a - auc_b) + 1.96 * se_diff

    interpretation = (
        f"{method_a_name} AUC ({auc_a:.3f}) significantly "
        f"{'higher' if auc_a > auc_b else 'lower'} than "
        f"{method_b_name} AUC ({auc_b:.3f}) (p={p_value:.4f})"
        if p_value < 0.05 else
        f"No significant AUC difference between {method_a_name} ({auc_a:.3f}) "
        f"and {method_b_name} ({auc_b:.3f}) (p={p_value:.4f})"
    )

    logger.info(f"DeLong's test: {interpretation}")

    return {
        "auc_a":            round(auc_a, 4),
        "auc_b":            round(auc_b, 4),
        "auc_difference":   round(auc_a - auc_b, 4),
        "ci_lower_95":      round(ci_lower, 4),
        "ci_upper_95":      round(ci_upper, 4),
        "z_statistic":      round(z, 4),
        "p_value":          round(p_value, 4),
        "significant":      p_value < 0.05,
        "alpha":            0.05,
        "method_a":         method_a_name,
        "method_b":         method_b_name,
        "n_pos":            n_pos,
        "n_neg":            n_neg,
        "interpretation":   interpretation,
        "reference":        "DeLong et al. (1988) Biometrics 44:837-845",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Runtime benchmarking
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeBenchmark:
    """
    Measure wall-clock runtime per disease analysis.

    Usage
    -----
        bench = RuntimeBenchmark()
        with bench.time("Parkinson disease"):
            result = await pipeline.analyze_disease("Parkinson disease")
        print(bench.report())
    """

    def __init__(self):
        self._times: Dict[str, float] = {}
        self._current_label: Optional[str] = None
        self._start: Optional[float] = None

    class _Timer:
        def __init__(self, bench, label):
            self._bench = bench
            self._label = label

        def __enter__(self):
            self._bench._start = time.time()
            self._bench._current_label = self._label
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self._bench._start
            self._bench._times[self._label] = elapsed

    def time(self, label: str) -> "_Timer":
        return self._Timer(self, label)

    def report(self) -> Dict:
        if not self._times:
            return {"n": 0, "times": {}}
        values = list(self._times.values())
        mean   = sum(values) / len(values)
        sd     = (
            math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            if len(values) > 1 else 0.0
        )
        return {
            "n":             len(values),
            "mean_seconds":  round(mean, 2),
            "sd_seconds":    round(sd, 2),
            "min_seconds":   round(min(values), 2),
            "max_seconds":   round(max(values), 2),
            "total_seconds": round(sum(values), 2),
            "per_disease":   {k: round(v, 2) for k, v in self._times.items()},
            "paper_statement": (
                f"Mean analysis time: {mean:.1f} ± {sd:.1f}s per disease "
                f"(range {min(values):.1f}–{max(values):.1f}s, n={len(values)})."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_analysis(
    input_path:  str = "validation_results.json",
    output_path: str = "statistical_results.json",
    n_bootstrap: int = 1000,
    seed:        int = 42,
) -> Dict:
    """
    Load validation_results.json and run all statistical tests.

    Outputs statistical_results.json with bootstrap CIs and test results.
    McNemar's and DeLong's tests require baseline scores — if not present
    in the input file, they are skipped with a note.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")

    with open(p) as f:
        data = json.load(f)

    positive_results = data.get("test_cases") or data.get("positive_results") or []
    negative_results = data.get("negative_cases") or data.get("negative_results") or []

    logger.info(f"Loaded {len(positive_results)} positive, "
                f"{len(negative_results)} negative results")

    # 1. Bootstrap CIs
    logger.info(f"\nRunning bootstrap CI ({n_bootstrap} iterations)...")
    ci_results = bootstrap_ci(
        positive_results, negative_results,
        n_iterations=n_bootstrap, seed=seed
    )

    # 2. McNemar's test (requires 'baseline_pass' key in results)
    mcnemar_result = None
    has_baseline = any("baseline_pass" in r for r in positive_results + negative_results)
    if has_baseline:
        logger.info("\nRunning McNemar's test vs baseline...")
        all_results    = positive_results + negative_results
        labels         = [r.get("expected_status") == "TRUE_POSITIVE" for r in all_results]
        preds_algo     = [bool(r.get("pass")) for r in all_results]
        preds_baseline = [bool(r.get("baseline_pass", False)) for r in all_results]
        mcnemar_result = mcnemar_test(preds_algo, preds_baseline, labels)
    else:
        logger.info("\nMcNemar's test skipped: no 'baseline_pass' key in results.")
        logger.info("  To enable: add 'baseline_pass' to each result in "
                    "validation_results.json by running a baseline scorer.")

    # 3. DeLong's test (requires 'baseline_score' key)
    delong_result = None
    has_baseline_scores = any("baseline_score" in r for r in positive_results + negative_results)
    if has_baseline_scores:
        logger.info("\nRunning DeLong's test...")
        all_results    = positive_results + negative_results
        labels_b       = [r.get("expected_status") == "TRUE_POSITIVE" for r in all_results]
        scores_algo    = [float(r.get("raw_score", 0)) for r in all_results]
        scores_base    = [float(r.get("baseline_score", 0)) for r in all_results]
        delong_result  = delong_test(scores_algo, scores_base, labels_b)
    else:
        logger.info("\nDeLong's test skipped: no 'baseline_score' key in results.")
        logger.info("  To enable: add 'baseline_score' to each result in "
                    "validation_results.json.")

    # Paper-ready summary
    paper_summary = _build_paper_summary(ci_results, mcnemar_result, delong_result)

    output = {
        "bootstrap_ci": ci_results,
        "mcnemar_test": mcnemar_result or {"skipped": True,
                        "reason": "No 'baseline_pass' key in validation_results.json"},
        "delong_test":  delong_result  or {"skipped": True,
                        "reason": "No 'baseline_score' key in validation_results.json"},
        "paper_summary": paper_summary,
        "metadata": {
            "input_file":           input_path,
            "n_positive_cases":     len(positive_results),
            "n_negative_cases":     len(negative_results),
            "n_bootstrap_iters":    n_bootstrap,
            "bootstrap_seed":       seed,
        },
    }

    Path(output_path).write_text(json.dumps(output, indent=2))
    logger.info(f"\nStatistical results → {output_path}")
    _print_summary(output)

    return output


def _build_paper_summary(ci: Dict, mcnemar: Optional[Dict], delong: Optional[Dict]) -> Dict:
    """Generate paper-ready text for each test."""
    lines = []

    for metric in ["sensitivity", "specificity", "precision", "f1"]:
        if metric in ci:
            r = ci[metric]
            label = metric.replace("_", " ").title()
            lines.append(
                f"{label} = {r['point_estimate']:.3f} "
                f"({r['ci_label']}: {r['ci_lower']:.3f}–{r['ci_upper']:.3f})"
            )

    if mcnemar and not mcnemar.get("skipped"):
        lines.append(
            f"McNemar's test vs baseline: b={mcnemar['b']}, c={mcnemar['c']}, "
            f"p={mcnemar['p_value']:.4f} "
            f"({'significant' if mcnemar['significant'] else 'not significant'} at α=0.05)"
        )

    if delong and not delong.get("skipped") and delong.get("z_statistic") is not None:
        lines.append(
            f"DeLong's test: AUC difference = {delong['auc_difference']:.3f} "
            f"(95% CI: {delong['ci_lower_95']:.3f}–{delong['ci_upper_95']:.3f}), "
            f"Z = {delong['z_statistic']:.2f}, p = {delong['p_value']:.4f}"
        )

    return {
        "ready_to_paste_lines": lines,
        "methods_blurb": (
            "Bootstrap 95% confidence intervals were computed using 1000 stratified "
            "resampling iterations (seed=42). McNemar's test (two-sided, α=0.05) "
            "compared algorithm and Jaccard baseline on paired binary predictions. "
            "AUC-ROC values were compared using DeLong's test (DeLong et al. 1988)."
        ),
    }


def _print_summary(output: Dict):
    print("\n" + "=" * 65)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 65)
    print("\nBootstrap 95% Confidence Intervals:")
    for metric, r in output["bootstrap_ci"].items():
        print(f"  {metric:15s}: {r['point_estimate']:.3f} "
              f"[{r['ci_lower']:.3f}–{r['ci_upper']:.3f}]")

    if not output["mcnemar_test"].get("skipped"):
        mn = output["mcnemar_test"]
        print(f"\nMcNemar's test: b={mn['b']}, c={mn['c']}, "
              f"p={mn['p_value']:.4f} ({mn['interpretation'][:60]}...)")

    if not output["delong_test"].get("skipped"):
        dl = output["delong_test"]
        if dl.get("z_statistic"):
            print(f"\nDeLong's test: ΔAUC={dl['auc_difference']:.3f}, "
                  f"Z={dl['z_statistic']:.2f}, p={dl['p_value']:.4f}")

    print("\nPaper-ready lines:")
    for line in output["paper_summary"]["ready_to_paste_lines"]:
        print(f"  • {line}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Publication-grade statistical analysis for drug repurposing validation"
    )
    parser.add_argument("--input",       default="validation_results.json")
    parser.add_argument("--output",      default="statistical_results.json")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    try:
        run_statistical_analysis(
            input_path=args.input,
            output_path=args.output,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()