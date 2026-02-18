#!/usr/bin/env python3
"""
FIXED Algorithm Validation Script
==================================
Key fixes vs original:
1. Reports metrics on TEST SET only (KNOWN_REPURPOSING_CASES).
   TUNING_SET cases are run separately for diagnostics but excluded
   from the headline numbers.
2. Adds per-category breakdown (mechanism_congruent vs empirical).
3. Adds random-baseline comparison so reviewers can see added value.
4. Score-range check now treats "too low" and "too high" separately
   with per-category context.
5. Saves a richer JSON with per-case details for supplementary material.
"""

import asyncio
import sys
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from validation_dataset import (
    KNOWN_REPURPOSING_CASES,
    TUNING_SET,
    NEGATIVE_CONTROLS,
    get_validation_metrics_target,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _random_baseline_sensitivity(n_positive: int, top_k: int = 100,
                                  n_drugs: int = 500) -> float:
    """Probability that a random ranker places all positives in top_k."""
    if n_positive == 0:
        return 0.0
    # Expected fraction of positives recovered if we randomly pick top_k drugs
    return min(top_k / n_drugs, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

async def _run_case(pipeline, case: dict, top_k: int = 100) -> dict:
    """Run one positive test case and return a result dict."""
    drug_name   = case["drug_name"]
    disease     = case["repurposed_for"]
    exp_min, exp_max = case["expected_score_range"]

    try:
        result = await pipeline.analyze_disease(
            disease_name=disease,
            min_score=0.0,          # fetch everything; we filter later
            max_results=top_k,
        )
    except Exception as exc:
        return {
            "drug":     drug_name,
            "disease":  disease,
            "category": case.get("category", "unknown"),
            "status":   "error",
            "reason":   str(exc),
            "score":    None,
        }

    if not result.get("success"):
        return {
            "drug":     drug_name,
            "disease":  disease,
            "category": case.get("category", "unknown"),
            "status":   "analysis_failed",
            "reason":   result.get("error", "unknown"),
            "score":    None,
        }

    found = next(
        (c for c in result["candidates"]
         if drug_name.lower() in c["drug_name"].lower()),
        None,
    )

    if found is None:
        return {
            "drug":           drug_name,
            "disease":        disease,
            "category":       case.get("category", "unknown"),
            "status":         "false_negative",
            "reason":         f"not in top {top_k}",
            "score":          None,
            "top_3":          [(c["drug_name"], round(c["score"], 3))
                               for c in result["candidates"][:3]],
            "expected_range": (exp_min, exp_max),
        }

    score = found["score"]
    in_range = exp_min <= score <= exp_max
    return {
        "drug":            drug_name,
        "disease":         disease,
        "category":        case.get("category", "unknown"),
        "status":          "found",
        "score":           score,
        "in_range":        in_range,
        "expected_range":  (exp_min, exp_max),
        "confidence":      found.get("confidence"),
        "shared_genes":    found.get("shared_genes", []),
        "shared_pathways": found.get("shared_pathways", []),
        "above_range":     score > exp_max,
        "below_range":     score < exp_min,
    }


async def _run_negative(pipeline, case: dict, top_k: int = 150) -> dict:
    drug_name = case["drug_name"]
    disease   = case["disease"]
    exp_max   = case["expected_score_range"][1]

    try:
        result = await pipeline.analyze_disease(
            disease_name=disease,
            min_score=0.0,
            max_results=top_k,
        )
    except Exception as exc:
        return {
            "drug": drug_name, "disease": disease,
            "status": "error", "reason": str(exc),
        }

    if not result.get("success"):
        return {
            "drug": drug_name, "disease": disease,
            "status": "analysis_failed",
        }

    found = next(
        (c for c in result["candidates"]
         if drug_name.lower() in c["drug_name"].lower()),
        None,
    )

    if found is None:
        return {
            "drug": drug_name, "disease": disease,
            "status": "true_negative_not_found",
            "score": 0.0,
        }

    score = found["score"]
    if score <= exp_max:
        return {
            "drug": drug_name, "disease": disease,
            "status": "true_negative_low_score",
            "score": score, "expected_max": exp_max,
        }
    return {
        "drug": drug_name, "disease": disease,
        "status": "false_positive",
        "score": score, "expected_max": exp_max,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run_validation() -> bool:
    _print_section("DRUG REPURPOSING ALGORITHM — VALIDATION STUDY")
    print()
    print("  TEST SET:   KNOWN_REPURPOSING_CASES  (final metrics reported here)")
    print("  TUNING SET: TUNING_SET               (diagnostics only, excluded from metrics)")
    print()

    try:
        from backend.pipeline.production_pipeline import ProductionPipeline
    except ImportError as exc:
        print(f"❌ Cannot import pipeline: {exc}")
        print("   Run from the project root directory.")
        return False

    pipeline = ProductionPipeline()

    # ── TUNING SET (diagnostics only) ────────────────────────────────────────
    _print_section("DIAGNOSTIC: Tuning Set (excluded from headline metrics)")
    tuning_results = []
    for i, case in enumerate(TUNING_SET, 1):
        print(f"\nTuning {i}/{len(TUNING_SET)}: "
              f"{case['drug_name']} → {case['repurposed_for']}")
        r = await _run_case(pipeline, case)
        tuning_results.append(r)
        _print_case_result(r)

    # ── TEST SET ──────────────────────────────────────────────────────────────
    _print_section("PHASE 1: Test Set — Known Repurposing Cases")
    test_results = []
    for i, case in enumerate(KNOWN_REPURPOSING_CASES, 1):
        print(f"\nTest {i}/{len(KNOWN_REPURPOSING_CASES)}: "
              f"{case['drug_name']} → {case['repurposed_for']}")
        print(f"  Category:       {case.get('category', 'unknown')}")
        print(f"  Expected score: {case['expected_score_range'][0]:.2f} – "
              f"{case['expected_score_range'][1]:.2f}")
        r = await _run_case(pipeline, case)
        test_results.append(r)
        _print_case_result(r)

    # ── NEGATIVE CONTROLS ─────────────────────────────────────────────────────
    _print_section("PHASE 2: Negative Controls")
    neg_results = []
    for i, case in enumerate(NEGATIVE_CONTROLS, 1):
        print(f"\nNegative {i}/{len(NEGATIVE_CONTROLS)}: "
              f"{case['drug_name']} → {case['disease']}")
        print(f"  Expected score: ≤ {case['expected_score_range'][1]:.2f}")
        r = await _run_negative(pipeline, case)
        neg_results.append(r)
        _print_neg_result(r)

    # ── COMPUTE METRICS ───────────────────────────────────────────────────────
    _print_section("VALIDATION RESULTS")

    tp = [r for r in test_results if r["status"] == "found"]
    fn = [r for r in test_results if r["status"] != "found"]
    tn = [r for r in neg_results  if "true_negative" in r["status"]]
    fp = [r for r in neg_results  if r["status"] == "false_positive"]

    n_pos = len(KNOWN_REPURPOSING_CASES)
    n_neg = len(NEGATIVE_CONTROLS)

    sensitivity = len(tp) / n_pos if n_pos else 0
    specificity = len(tn) / n_neg if n_neg else 0
    precision   = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    accuracy    = (len(tp) + len(tn)) / (n_pos + n_neg)

    print(f"\nTrue Positives  (TP): {len(tp)} / {n_pos}")
    print(f"False Negatives (FN): {len(fn)} / {n_pos}")
    print(f"True Negatives  (TN): {len(tn)} / {n_neg}")
    print(f"False Positives (FP): {len(fp)} / {n_neg}")

    print("\nOVERALL PERFORMANCE METRICS:")
    print(f"  Sensitivity (Recall): {sensitivity:.2%}")
    print(f"  Specificity:          {specificity:.2%}")
    print(f"  Precision:            {precision:.2%}")
    print(f"  Accuracy:             {accuracy:.2%}")

    # Category breakdown
    mech   = [r for r in test_results if r.get("category") == "mechanism_congruent"]
    emp    = [r for r in test_results if r.get("category") == "empirical"]
    mech_tp = [r for r in mech if r["status"] == "found"]
    emp_tp  = [r for r in emp  if r["status"] == "found"]

    print("\nCATEGORY BREAKDOWN:")
    print(f"  Mechanism-congruent cases: {len(mech_tp)}/{len(mech)}"
          f"  sensitivity={len(mech_tp)/len(mech):.2%}" if mech else
          "  Mechanism-congruent cases: 0/0")
    print(f"  Empirical cases:           {len(emp_tp)}/{len(emp)}"
          f"  sensitivity={len(emp_tp)/len(emp):.2%}" if emp else
          "  Empirical cases: 0/0")
    print()
    print("  (Note: empirical cases were discovered clinically, not via gene-overlap;")
    print("   lower scores for these cases are expected and scientifically justified)")

    # Score distribution for found cases
    scores = [r["score"] for r in tp]
    if scores:
        print("\nSCORE DISTRIBUTION (detected cases):")
        print(f"  Mean:   {np.mean(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Std:    {np.std(scores):.3f}")
        print(f"  Min:    {np.min(scores):.3f}")
        print(f"  Max:    {np.max(scores):.3f}")
        in_range = sum(1 for r in tp if r.get("in_range", False))
        print(f"  Scores within expected range: {in_range}/{len(tp)}")

    # Random baseline comparison
    rand_sens = _random_baseline_sensitivity(n_pos)
    print(f"\nRANDOM BASELINE (top-100 from 500 drugs): {rand_sens:.2%} sensitivity")
    print(f"ALGORITHM LIFT over baseline:              "
          f"{sensitivity - rand_sens:+.2%}")

    # Pass/fail
    targets = get_validation_metrics_target()
    print("\nTARGET METRICS (publication threshold):")
    sens_ok = sensitivity >= targets["sensitivity"]
    spec_ok = specificity >= targets["specificity"]
    prec_ok = precision   >= targets["precision"]
    print(f"  Sensitivity: >{targets['sensitivity']:.0%}  →  "
          f"{'✅ PASS' if sens_ok else '❌ FAIL'}")
    print(f"  Specificity: >{targets['specificity']:.0%}  →  "
          f"{'✅ PASS' if spec_ok else '❌ FAIL'}")
    print(f"  Precision:   >{targets['precision']:.0%}  →  "
          f"{'✅ PASS' if prec_ok else '❌ FAIL'}")

    passed = sens_ok and spec_ok and prec_ok
    print()
    print("=" * 80)
    if passed:
        print("✅ VALIDATION PASSED — Algorithm meets publication standards")
        print()
        print("Metrics to cite in your paper:")
        print(f"  • Validated on {n_pos} known repurposing successes "
              f"({len([c for c in KNOWN_REPURPOSING_CASES if c['category']=='mechanism_congruent'])} "
              f"mechanism-congruent, "
              f"{len([c for c in KNOWN_REPURPOSING_CASES if c['category']=='empirical'])} empirical)")
        print(f"  • Sensitivity: {sensitivity:.1%}")
        print(f"  • Specificity: {specificity:.1%}")
        print(f"  • Precision:   {precision:.1%}")
        print(f"  • Lift over random baseline: {sensitivity - rand_sens:+.1%}")
    else:
        print("❌ VALIDATION FAILED — Improvements needed")
        print()
        if not sens_ok:
            print(f"  Sensitivity {sensitivity:.1%} < {targets['sensitivity']:.0%}")
            print("   → Check DGIdb enrichment rate in logs")
            print("   → Verify pathway map covers disease's key genes")
            print("   → Lower min_score threshold in validation call")
        if not spec_ok:
            print(f"  Specificity {specificity:.1%} < {targets['specificity']:.0%}")
            print("   → False positives detected — raise scoring thresholds")
        if not prec_ok:
            print(f"  Precision {precision:.1%} < {targets['precision']:.0%}")
            print("   → Too many false positives among predictions")
    print("=" * 80)

    # Save results
    output = {
        "test_cases":      test_results,
        "tuning_cases":    tuning_results,
        "negative_cases":  neg_results,
        "metrics": {
            "sensitivity":  sensitivity,
            "specificity":  specificity,
            "precision":    precision,
            "accuracy":     accuracy,
            "random_baseline_sensitivity": rand_sens,
            "lift_over_baseline": sensitivity - rand_sens,
        },
        "passed": passed,
    }
    out_file = Path(__file__).parent / "validation_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n📁 Detailed results saved to: {out_file}")

    await pipeline.close()
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_case_result(r: dict) -> None:
    status = r["status"]
    if status == "found":
        rng = r.get("expected_range", (0, 1))
        flag = "✅ Within expected range" if r.get("in_range") else (
            f"⚠️  Score too HIGH (expected {rng[0]:.2f}–{rng[1]:.2f})"
            if r.get("above_range") else
            f"⚠️  Score too LOW (expected {rng[0]:.2f}–{rng[1]:.2f})"
        )
        print(f"  ✅ Found! Score: {r['score']:.3f}  confidence={r.get('confidence','?')}")
        print(f"     Shared genes: {r.get('shared_genes', [])}")
        print(f"     Shared pathways: {r.get('shared_pathways', [])}")
        print(f"  {flag}")
    elif status == "false_negative":
        print(f"  ❌ Not found in top candidates")
        print(f"     → FALSE NEGATIVE — algorithm missed known success")
        top3 = r.get("top_3", [])
        if top3:
            print("     Top-3 instead:")
            for name, sc in top3:
                print(f"       {name}: {sc}")
    elif status in ("analysis_failed", "error"):
        print(f"  ❌ Analysis failed: {r.get('reason', 'unknown')}")
        print(f"     Suggestion: Try searching with full disease name")


def _print_neg_result(r: dict) -> None:
    s = r["status"]
    if "true_negative" in s:
        score = r.get("score", 0)
        print(f"  ✅ Correctly low score: {score:.3f}")
    elif s == "false_positive":
        print(f"  ❌ FALSE POSITIVE: score={r.get('score', '?'):.3f} "
              f"(expected ≤ {r.get('expected_max', '?'):.2f})")
    else:
        print(f"  ⚠️  {s}: {r.get('reason', '')}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("⚠️  IMPORTANT: Run this BEFORE publication.")
    print("   Metrics are reported on the held-out TEST SET only.")
    print("   Estimated runtime: 10–20 minutes.")
    print()
    success = asyncio.run(run_validation())
    sys.exit(0 if success else 1)