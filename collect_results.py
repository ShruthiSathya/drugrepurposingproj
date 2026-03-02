"""
collect_results.py — Aggregate all validation outputs into one file
===================================================================
Run this after all tests complete to produce a single results summary
you can share for review.

Usage
-----
    python3 collect_results.py
    python3 collect_results.py --output my_results.json
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


FILES = {
    "validation":   "validation_results.json",
    "calibration":  "calibration_results.json",
    "statistical":  "statistical_results.json",
    "fn_analysis":  "fn_analysis.json",
    "repodb":       "repodb_paper.json",           # falls back to repodb_filtered.json or repodb_quick.json
    "prospective":  "pbc_predictions.json",
}

REPODB_FALLBACKS = ["repodb_paper.json", "repodb_full.json", "repodb_filtered.json", "repodb_quick.json",
                    "repodb_benchmark_results.json"]


def load(path: str):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None


def safe(val, decimals=4):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return val


def collect(output_path: str = "all_results.json"):
    out = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_found": [],
        "files_missing": [],
    }

    # ── 1. Curated validation ────────────────────────────────────────────────
    v = load(FILES["validation"])
    if v:
        out["files_found"].append(FILES["validation"])
        m = v.get("metrics", {})
        h = v.get("header", {})
        out["curated_validation"] = {
            "dataset_version":  h.get("dataset_version"),
            "n_cases":          h.get("n_test_cases"),
            "n_positive":       h.get("n_positive_cases"),
            "n_negative":       h.get("n_negative_cases"),
            "pass_criterion":   h.get("pass_criterion"),
            "elapsed_seconds":  h.get("elapsed_seconds"),
            "metrics": {
                "tp":           m.get("tp"),
                "fn":           m.get("fn"),
                "tn":           m.get("tn"),
                "fp":           m.get("fp"),
                "sensitivity":  safe(m.get("sensitivity")),
                "specificity":  safe(m.get("specificity")),
                "precision":    safe(m.get("precision")),
                "f1":           safe(m.get("f1")),
                "rank_only_passes":     m.get("rank_only_passes"),
                "score_only_passes":    m.get("score_only_passes"),
                "both_criteria_passes": m.get("both_criteria_passes"),
            },
            "stratified_metrics": v.get("stratified_metrics", {}),
            "baseline_comparison": v.get("baseline_comparison", {}),
            "baseline_f1_delta":   v.get("baseline_f1_delta", {}),
            "sensitivity_analysis": v.get("sensitivity_analysis", {}),
            "data_integrity":      v.get("data_integrity", {}),
            "pipeline_fingerprint": v.get("pipeline_fingerprint", {}),
            # Individual case results
            "positive_results": [
                {
                    "drug":             r.get("drug"),
                    "disease":          r.get("disease"),
                    "disease_area":     r.get("disease_area"),
                    "pass":             r.get("pass"),
                    "raw_score":        safe(r.get("raw_score")),
                    "calibrated_score": safe(r.get("calibrated_score")),
                    "rank":             r.get("rank"),
                    "rank_pass":        r.get("rank_pass"),
                    "score_pass":       r.get("score_pass"),
                    "status":           r.get("status"),
                    "mechanism":        r.get("mechanism"),
                    "notes":            r.get("notes"),
                    "score_components": r.get("score_components", {}),
                }
                for r in (v.get("positive_results") or v.get("test_cases") or [])
            ],
            "negative_results": [
                {
                    "drug":             r.get("drug"),
                    "disease":          r.get("disease"),
                    "disease_area":     r.get("disease_area"),
                    "pass":             r.get("pass"),
                    "raw_score":        safe(r.get("raw_score")),
                    "status":           r.get("status"),
                }
                for r in (v.get("negative_results") or v.get("negative_cases") or [])
            ],
        }
    else:
        out["files_missing"].append(FILES["validation"])
        out["curated_validation"] = None

    # ── 2. Calibration ───────────────────────────────────────────────────────
    c = load(FILES["calibration"])
    if c:
        out["files_found"].append(FILES["calibration"])
        pp = c.get("platt_parameters", {})
        ct = c.get("classification_threshold", {})
        ei = c.get("ece_interpretation", {})
        out["calibration"] = {
            "platt_A":              safe(pp.get("A")),
            "platt_B":              safe(pp.get("B")),
            "platt_source":         pp.get("source"),
            "ece":                  safe(c.get("ece")),
            "ece_status":           ei.get("status"),
            "ece_paper_note":       ei.get("paper_note"),
            "calibrated_scores_usable": ei.get("calibrated_scores_usable"),
            "raw_threshold_at_50pct":   safe(ct.get("raw_equivalent_50")),
            "practical_raw_threshold":  ct.get("practical_raw_threshold"),
            "effect_sizes":         c.get("effect_sizes", {}),
            "score_distributions":  c.get("score_distributions", {}),
            "paper_reporting_note": c.get("paper_reporting_note", {}).get(
                "recommended_methods_text"
            ),
        }
    else:
        out["files_missing"].append(FILES["calibration"])
        out["calibration"] = None

    # ── 3. Statistical tests ─────────────────────────────────────────────────
    s = load(FILES["statistical"])
    if s:
        out["files_found"].append(FILES["statistical"])
        out["statistical_tests"] = {
            "bootstrap_ci":  s.get("bootstrap_ci", {}),
            "mcnemar_test":  s.get("mcnemar_test", {}),
            "delong_test":   s.get("delong_test", {}),
            "paper_summary": s.get("paper_summary", {}),
            "metadata":      s.get("metadata", {}),
        }
    else:
        out["files_missing"].append(FILES["statistical"])
        out["statistical_tests"] = None

    # ── 4. False negative analysis ───────────────────────────────────────────
    fn = load(FILES["fn_analysis"])
    if fn:
        out["files_found"].append(FILES["fn_analysis"])
        out["fn_analysis"] = {
            "summary":              fn.get("summary", {}),
            "false_negative_cases": fn.get("false_negative_cases", []),
        }
    else:
        out["files_missing"].append(FILES["fn_analysis"])
        out["fn_analysis"] = None

    # ── 5. RepoDB benchmark ──────────────────────────────────────────────────
    repodb_file = None
    repodb_data = None
    for fname in REPODB_FALLBACKS:
        repodb_data = load(fname)
        if repodb_data:
            repodb_file = fname
            break

    if repodb_data:
        out["files_found"].append(repodb_file)
        sm = repodb_data.get("summary", {})
        per_disease = repodb_data.get("per_disease_results", [])
        top_n = sm.get("top_n", 50)

        # Sort per-disease by AUC-ROC for easy review
        valid_pd = [d for d in per_disease if d.get("auc_roc") is not None]
        valid_pd.sort(key=lambda x: x["auc_roc"])

        out["repodb_benchmark"] = {
            "source_file":          repodb_file,
            "n_diseases_requested": sm.get("n_diseases_requested"),
            "n_diseases_resolved":  sm.get("n_diseases_resolved"),
            "n_diseases_skipped":   sm.get("n_diseases_skipped"),
            "resolve_rate":         safe(sm.get("resolve_rate")),
            "disease_filter":       sm.get("disease_filter"),
            "n_drugs":              sm.get("n_drugs"),
            "top_n":                top_n,
            "global_auc_roc":       safe(sm.get("global_auc_roc")),
            "global_auc_pr":        safe(sm.get("global_auc_pr")),
            f"hit_at_{top_n}":      safe(sm.get(f"hit_at_{top_n}")),
            "mrr":                  safe(sm.get("mrr")),
            "skipped_diseases":     repodb_data.get("skipped_diseases", []),
            # Bottom 5 and top 5 by AUC-ROC for quick review
            "lowest_auc_diseases":  valid_pd[:5],
            "highest_auc_diseases": valid_pd[-5:][::-1],
            "null_auc_diseases": [
                d for d in per_disease if d.get("auc_roc") is None
            ],
        }
    else:
        out["files_missing"].extend(REPODB_FALLBACKS[:1])
        out["repodb_benchmark"] = None

    # ── 6. Prospective predictions ───────────────────────────────────────────
    pp = load(FILES["prospective"])
    if pp:
        out["files_found"].append(FILES["prospective"])
        out["prospective_predictions"] = {
            "metadata":          pp.get("metadata", {}),
            "paper_table":       pp.get("paper_table", []),
            "paper_methods_blurb": pp.get("paper_methods_blurb", ""),
            # Top predictions by score
            "top_predictions": sorted(
                pp.get("all_predictions", []),
                key=lambda x: x.get("composite_score", 0),
                reverse=True
            )[:10],
        }
    else:
        out["files_missing"].append(FILES["prospective"])
        out["prospective_predictions"] = None

    # ── Summary banner ───────────────────────────────────────────────────────
    cv = out.get("curated_validation") or {}
    m  = cv.get("metrics") or {}
    rb = out.get("repodb_benchmark") or {}
    st = out.get("statistical_tests") or {}
    ci = st.get("bootstrap_ci") or {}

    out["summary_for_paper"] = {
        "curated_validation": {
            "sensitivity": safe(m.get("sensitivity")),
            "specificity": safe(m.get("specificity")),
            "precision":   safe(m.get("precision")),
            "f1":          safe(m.get("f1")),
            "tp": m.get("tp"), "fn": m.get("fn"),
            "tn": m.get("tn"), "fp": m.get("fp"),
        },
        "bootstrap_95ci": {
            metric: {
                "point": safe(ci.get(metric, {}).get("point_estimate")),
                "lower": safe(ci.get(metric, {}).get("ci_lower")),
                "upper": safe(ci.get(metric, {}).get("ci_upper")),
            }
            for metric in ["sensitivity", "specificity", "precision", "f1"]
            if metric in ci
        },
        "repodb": {
            "auc_roc":              safe(rb.get("global_auc_roc")),
            "auc_pr":               safe(rb.get("global_auc_pr")),
            f"hit_at_{rb.get('top_n', 50)}": safe(rb.get(f"hit_at_{rb.get('top_n', 50)}")),
            "mrr":                  safe(rb.get("mrr")),
            "n_diseases":           rb.get("n_diseases_resolved"),
        },
        "calibration": {
            "platt_A": safe((out.get("calibration") or {}).get("platt_A")),
            "platt_B": safe((out.get("calibration") or {}).get("platt_B")),
            "ece":     safe((out.get("calibration") or {}).get("ece")),
        },
        "baseline_f1_deltas": cv.get("baseline_f1_delta", {}),
        "sensitivity_analysis_stable": (
            cv.get("sensitivity_analysis", {}).get("stable")
        ),
    }

    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Files included:  {out['files_found']}")
    if out["files_missing"]:
        print(f"  Files missing:   {out['files_missing']}")

    # Print paper-ready summary
    s4p = out["summary_for_paper"]
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)

    cv_m = s4p.get("curated_validation", {})
    if any(cv_m.values()):
        print(f"\nCurated validation (n=55):")
        print(f"  Sensitivity : {cv_m.get('sensitivity')}  (TP={cv_m.get('tp')}, FN={cv_m.get('fn')})")
        print(f"  Specificity : {cv_m.get('specificity')}  (TN={cv_m.get('tn')}, FP={cv_m.get('fp')})")
        print(f"  Precision   : {cv_m.get('precision')}")
        print(f"  F1          : {cv_m.get('f1')}")

    bci = s4p.get("bootstrap_95ci", {})
    if bci:
        print(f"\nBootstrap 95% CI:")
        for metric, vals in bci.items():
            print(f"  {metric:12s}: {vals.get('point')}  [{vals.get('lower')} – {vals.get('upper')}]")

    rb2 = s4p.get("repodb", {})
    if rb2.get("auc_roc"):
        top_n_key = [k for k in rb2 if k.startswith("hit_at_")]
        hit_str = f"  {top_n_key[0]}: {rb2.get(top_n_key[0])}" if top_n_key else ""
        print(f"\nRepoDB benchmark (n={rb2.get('n_diseases')} diseases):")
        print(f"  AUC-ROC : {rb2.get('auc_roc')}")
        print(f"  AUC-PR  : {rb2.get('auc_pr')}")
        if hit_str:
            print(hit_str)
        print(f"  MRR     : {rb2.get('mrr')}")

    bl = s4p.get("baseline_f1_deltas", {})
    if bl:
        print(f"\nBaseline F1 deltas:")
        for name, d in bl.items():
            print(f"  vs {name:15s}: main={d.get('main_f1')}  "
                  f"baseline={d.get('baseline_f1')}  Δ={d.get('f1_delta'):+.4f}")

    cal = s4p.get("calibration", {})
    if cal.get("ece") is not None:
        print(f"\nCalibration:")
        print(f"  Platt A={cal.get('platt_A')}  B={cal.get('platt_B')}  ECE={cal.get('ece')}")

    print("=" * 60)
    print(f"\nShare '{output_path}' for full review.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all validation outputs into one file"
    )
    parser.add_argument(
        "--output", type=str, default="all_results.json",
        help="Output filename (default: all_results.json)"
    )
    args = parser.parse_args()
    collect(args.output)


if __name__ == "__main__":
    main()