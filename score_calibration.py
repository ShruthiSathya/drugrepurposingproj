#!/usr/bin/env python3
"""
score_calibration.py
====================
Post-hoc statistical analysis of validation_results.json.

Produces:
  - Bootstrap 95% CIs for sensitivity, specificity, precision, F1
  - AUROC and AUPRC (with CIs)
  - Platt scaling: maps raw scores -> calibrated probabilities
  - Score separability analysis (Cohen's d)
  - Calibration curve data (reliability diagram)

Usage:
    python score_calibration.py validation_results.json
    python score_calibration.py validation_results.json --n-bootstrap 2000 --output calibration_results.json
"""

import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def _metrics_from_labels(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def _bootstrap_ci(labels, preds, scores, n=1000, alpha=0.05, seed=42):
    rng = random.Random(seed)
    k   = len(labels)
    boot = {"sensitivity": [], "specificity": [], "precision": [], "f1": [], "auroc": []}
    for _ in range(n):
        idx = [rng.randint(0, k - 1) for _ in range(k)]
        bl  = [labels[i] for i in idx]
        bp  = [preds[i]  for i in idx]
        bs  = [scores[i] for i in idx]
        m   = _metrics_from_labels(bl, bp)
        for key in ("sensitivity", "specificity", "precision", "f1"):
            boot[key].append(m[key])
        boot["auroc"].append(_auroc(bl, bs))
    lo, hi = alpha / 2, 1 - alpha / 2
    result = {}
    for metric, vals in boot.items():
        vals.sort()
        result[metric] = {
            "mean":  sum(vals) / len(vals),
            "lower": vals[int(lo * n)],
            "upper": vals[int(hi * n)],
        }
    return result


def _auroc(labels, scores):
    paired = sorted(zip(scores, labels), reverse=True)
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tp = fp = prev_fp = prev_tp = 0
    auc = 0.0
    prev_score = None
    for score, label in paired:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_fp, prev_tp = fp, tp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2
    return auc / (n_pos * n_neg)


def _auprc(labels, scores):
    paired  = sorted(zip(scores, labels), reverse=True)
    n_pos   = sum(labels)
    if n_pos == 0:
        return 0.0
    tp = fp = 0
    prev_prec = 1.0
    prev_rec  = 0.0
    auc = 0.0
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec  = tp / n_pos
        auc += (rec - prev_rec) * (prec + prev_prec) / 2
        prev_prec, prev_rec = prec, rec
    return auc


def _platt_scale(scores, labels, n_iter=1000, lr=0.1):
    A, B = 1.0, 0.0
    for _ in range(n_iter):
        dA = dB = 0.0
        for s, y in zip(scores, labels):
            p   = 1 / (1 + math.exp(-(A * s + B)))
            err = p - y
            dA += err * s
            dB += err
        A -= lr * dA / len(scores)
        B -= lr * dB / len(scores)
    return A, B


def _apply_platt(scores, A, B):
    return [1 / (1 + math.exp(-(A * s + B))) for s in scores]


def _cohens_d(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return 0.0
    n1, n2  = len(pos_scores), len(neg_scores)
    m1 = sum(pos_scores) / n1
    m2 = sum(neg_scores) / n2
    v1 = sum((x - m1) ** 2 for x in pos_scores) / max(n1 - 1, 1)
    v2 = sum((x - m2) ** 2 for x in neg_scores) / max(n2 - 1, 1)
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0


def _calibration_curve(labels, scores, n_bins=5):
    bins = [[] for _ in range(n_bins)]
    for s, l in zip(scores, labels):
        b = min(int(s * n_bins), n_bins - 1)
        bins[b].append(l)
    result = []
    for i, b in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if b:
            result.append({
                "bin":            f"{lo:.1f}-{hi:.1f}",
                "mean_predicted": (lo + hi) / 2,
                "fraction_pos":   sum(b) / len(b),
                "count":          len(b),
            })
    return result


def _print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to validation_results.json")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output", default="calibration_results.json")
    args = parser.parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    neg_cases  = data.get("negative_cases", [])

    pos_scores = [r.get("score") or 0.0 for r in test_cases]
    pos_labels = [1] * len(pos_scores)
    neg_scores = [r.get("score") or 0.0 for r in neg_cases]
    neg_labels = [0] * len(neg_scores)

    all_scores = pos_scores + neg_scores
    all_labels = pos_labels + neg_labels

    THRESHOLD = 0.15
    all_preds  = [1 if s >= THRESHOLD else 0 for s in all_scores]

    _print_section("POINT ESTIMATES")
    point = _metrics_from_labels(all_labels, all_preds)
    auroc = _auroc(all_labels, all_scores)
    auprc = _auprc(all_labels, all_scores)
    for k, v in point.items():
        print(f"  {k:<15} {v:.3f}")
    print(f"  {'AUROC':<15} {auroc:.3f}")
    print(f"  {'AUPRC':<15} {auprc:.3f}")

    _print_section(f"BOOTSTRAP 95% CIs  (n={args.n_bootstrap})")
    print("  Computing...", end="", flush=True)
    cis = _bootstrap_ci(all_labels, all_preds, all_scores, n=args.n_bootstrap)
    print(" done.")
    print(f"\n  {'Metric':<15} {'Mean':>8} {'Lower 95%':>10} {'Upper 95%':>10}")
    print(f"  {'-'*45}")
    for metric, vals in cis.items():
        print(f"  {metric:<15} {vals['mean']:>8.3f} {vals['lower']:>10.3f} {vals['upper']:>10.3f}")

    _print_section("PLATT SCALING (score -> probability)")
    A, B = _platt_scale(all_scores, all_labels)
    print(f"  Sigmoid params: A={A:.4f}, B={B:.4f}")
    print(f"  Interpretation: score 0.15 -> {_apply_platt([0.15], A, B)[0]:.1%} probability")
    print(f"                  score 0.30 -> {_apply_platt([0.30], A, B)[0]:.1%} probability")
    print(f"                  score 0.50 -> {_apply_platt([0.50], A, B)[0]:.1%} probability")

    _print_section("SCORE SEPARABILITY (Cohen's d)")
    tp_scores = [r.get("score") or 0.0 for r in test_cases if r.get("status") == "found"]
    fp_scores = [r.get("score") or 0.0 for r in neg_cases  if r.get("status") == "false_positive"]
    tn_scores = [r.get("score") or 0.0 for r in neg_cases  if "true_negative" in r.get("status", "")]
    d = _cohens_d(tp_scores, tn_scores + fp_scores)
    interp = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    if tp_scores:
        print(f"  TP mean score:  {sum(tp_scores)/len(tp_scores):.3f}  (n={len(tp_scores)})")
    if tn_scores:
        print(f"  TN mean score:  {sum(tn_scores)/len(tn_scores):.3f}  (n={len(tn_scores)})")
    print(f"  Cohen's d:      {d:.3f}  ({interp} effect)")

    _print_section("CALIBRATION CURVE")
    cal_curve = _calibration_curve(all_labels, all_scores)
    print(f"  {'Bin':<10} {'Mean predicted':>15} {'Fraction positive':>18} {'Count':>6}")
    for row in cal_curve:
        print(f"  {row['bin']:<10} {row['mean_predicted']:>15.2f} {row['fraction_pos']:>18.2f} {row['count']:>6}")

    output = {
        "point_estimates": {**point, "auroc": auroc, "auprc": auprc},
        "bootstrap_cis":   cis,
        "platt_scaling":   {"A": A, "B": B},
        "cohens_d":        {
            "d": d, "interpretation": interp,
            "tp_mean": sum(tp_scores)/len(tp_scores) if tp_scores else 0,
            "tn_mean": sum(tn_scores)/len(tn_scores) if tn_scores else 0,
        },
        "calibration_curve": cal_curve,
        "n_bootstrap": args.n_bootstrap,
        "threshold":   THRESHOLD,
    }
    out_path = Path(args.results_file).parent / args.output
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()