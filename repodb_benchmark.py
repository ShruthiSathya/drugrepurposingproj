"""
repodb_benchmark.py — RepoDB Benchmarking Script
=================================================
Evaluates the drug repurposing pipeline against the RepoDB gold-standard
dataset of approved drug-indication pairs.

FIXES vs previous version
--------------------------
1. Import: Imports ProductionPipeline (canonical name) instead of
   RepurposingPipeline, which did not exist as a class in the module.
   RepurposingPipeline is now an alias in production_pipeline.py so the
   old import would also work, but using the canonical name is cleaner.

2. generate_candidates(): Calls pipeline.generate_candidates() directly
   instead of the old code that called pipeline.analyze_disease() and then
   tried to access result['candidates']. analyze_disease() runs the full
   5-step pipeline including PubMed lookups; generate_candidates() is the
   extracted scoring method that is both faster for bulk benchmarking and
   consistent with validation run output.

3. drugs_cache: Fetches approved drugs ONCE and reuses across all disease
   queries, matching the validation runner behaviour.

RepoDB Reference
----------------
Brown AS & Patel CJ (2017). "A standard database for drug repositioning."
Pac Symp Biocomput, 22, 393–401. PMID:27896997.
Dataset: https://unmtid-shinyapps.net/shiny/repodb/

Metrics computed
----------------
  AUC-ROC  — area under ROC curve (primary benchmark metric)
  AUC-PR   — area under Precision-Recall curve
  Hit@N    — fraction of RepoDB positives in top-N candidates per disease
  MRR      — mean reciprocal rank of known positives

Usage
-----
    python repodb_benchmark.py [--repodb-file repodb.csv] [--top-n 50]
                               [--output repodb_benchmark_results.json]
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Import ProductionPipeline by canonical name.
# RepurposingPipeline = ProductionPipeline alias exists in the module but
# using the canonical class name is preferred.
# ─────────────────────────────────────────────────────────────────────────────
from backend.pipeline.production_pipeline import ProductionPipeline
from backend.pipeline.calibration import calibrate_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RepoDB data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_repodb(path: str) -> Tuple[List[Dict], List[str]]:
    """
    Load RepoDB CSV file.

    Returns
    -------
    (pairs, diseases) where:
      pairs    = list of {drug, disease, label} dicts (label 1=approved, 0=withdrawn/investigational)
      diseases = sorted list of unique disease names to benchmark
    """
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = 1 if row.get("status", "").lower() == "approved" else 0
            pairs.append({
                "drug":    row["drug_name"].strip(),
                "disease": row["ind_name"].strip(),
                "label":   label,
            })

    diseases = sorted(set(p["disease"] for p in pairs))
    logger.info(f"Loaded {len(pairs)} pairs across {len(diseases)} diseases from {path}")
    return pairs, diseases


def normalize_drug_name(name: str) -> str:
    """Lowercase and strip for fuzzy drug matching."""
    return name.lower().strip()


def match_drug_name(candidate_name: str, repodb_name: str) -> bool:
    """
    Check if pipeline candidate name matches RepoDB drug name.
    Uses substring matching to handle minor naming variants
    (e.g. 'imatinib mesylate' in RepoDB vs 'imatinib' in ChEMBL).
    """
    a = normalize_drug_name(candidate_name)
    b = normalize_drug_name(repodb_name)
    return a == b or a in b or b in a


# ─────────────────────────────────────────────────────────────────────────────
# AUC helpers (no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _auc_roc(scores: List[float], labels: List[int]) -> float:
    """Trapezoidal AUC-ROC."""
    pairs = sorted(zip(scores, labels), reverse=True)
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
        if label == 1:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_tp = tp
        prev_fp = fp

    return auc / (n_pos * n_neg)


def _auc_pr(scores: List[float], labels: List[int]) -> float:
    """Trapezoidal AUC-PR."""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_recall = 0.0
    prev_prec   = 1.0

    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        prec   = tp / (tp + fp)
        auc   += (recall - prev_recall) * (prec + prev_prec) / 2.0
        prev_recall = recall
        prev_prec   = prec

    return auc


def _hit_at_n(rank: Optional[int], n: int) -> int:
    """Return 1 if rank <= n, else 0."""
    return 1 if rank is not None and rank <= n else 0


def _reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(
    repodb_path: str,
    top_n:       int  = 50,
    output_path: str  = "repodb_benchmark_results.json",
    max_diseases: Optional[int] = None,
) -> Dict:
    """
    Score every RepoDB disease against the full drug set and compute AUC/Hit/MRR.
    """
    pairs, diseases = load_repodb(repodb_path)
    if max_diseases:
        diseases = diseases[:max_diseases]
        logger.info(f"Limiting to first {max_diseases} diseases")

    pipeline = ProductionPipeline()

    # ─── FIX 3: Fetch drugs ONCE and reuse (matches validation runner) ────────
    logger.info("Fetching approved drugs from ChEMBL (shared across all diseases)...")
    drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
    logger.info(f"Using {len(drugs_data)} drugs\n")

    all_scores: List[float] = []
    all_labels: List[int]   = []
    per_disease_results      = []
    hit_at_n_counts          = []
    reciprocal_ranks         = []

    for d_idx, disease_name in enumerate(diseases, 1):
        logger.info(f"[{d_idx}/{len(diseases)}] Benchmarking: {disease_name}")

        disease_data = await pipeline.data_fetcher.fetch_disease_data(disease_name)
        if not disease_data:
            logger.warning(f"  Skipped — disease not found in OpenTargets")
            continue

        # ─── FIX 2: Use generate_candidates() not analyze_disease() ──────────
        # generate_candidates() is the extracted scoring method:
        #   - Does NOT run PubMed lookups (fetch_pubmed=False, fast for bulk)
        #   - Does NOT filter by min_score (returns all drugs ranked)
        #   - Consistent with validation runner output
        candidates = await pipeline.generate_candidates(
            disease_data=disease_data,
            drugs_data=drugs_data,
            min_score=0.0,
            fetch_pubmed=False,
        )

        # Sort candidates by score descending
        candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # Get RepoDB pairs for this disease
        disease_pairs = [p for p in pairs if p["disease"] == disease_name]
        positive_drugs = {normalize_drug_name(p["drug"]) for p in disease_pairs if p["label"] == 1}

        # Collect scores and labels for AUC computation
        disease_scores: List[float] = []
        disease_labels: List[int]   = []
        disease_result_pairs        = []

        for rank, cand in enumerate(candidates_sorted, 1):
            cand_norm = normalize_drug_name(cand["name"])
            # Check if any RepoDB positive matches this candidate
            matched_positive = any(
                match_drug_name(cand["name"], pos_drug)
                for pos_drug in positive_drugs
            )
            label = 1 if matched_positive else 0
            disease_scores.append(cand["score"])
            disease_labels.append(label)
            disease_result_pairs.append({"name": cand["name"], "score": cand["score"], "rank": rank, "label": label})

        all_scores.extend(disease_scores)
        all_labels.extend(disease_labels)

        # Hit@N and MRR per disease
        for pos_drug in positive_drugs:
            matched_rank = None
            for rank, cand in enumerate(candidates_sorted, 1):
                if match_drug_name(cand["name"], pos_drug):
                    matched_rank = rank
                    break
            hit_at_n_counts.append(_hit_at_n(matched_rank, top_n))
            reciprocal_ranks.append(_reciprocal_rank(matched_rank))

        disease_auc = _auc_roc(disease_scores, disease_labels)
        disease_pr  = _auc_pr(disease_scores, disease_labels)

        per_disease_results.append({
            "disease":     disease_name,
            "n_positives": len(positive_drugs),
            "n_candidates":len(candidates),
            "auc_roc":     round(disease_auc, 4) if disease_auc == disease_auc else None,
            "auc_pr":      round(disease_pr, 4) if disease_pr == disease_pr else None,
        })

        logger.info(
            f"  Positives: {len(positive_drugs)}, "
            f"AUC-ROC: {disease_auc:.3f}, AUC-PR: {disease_pr:.3f}"
        )

    await pipeline.close()

    # Global metrics
    global_auc_roc = _auc_roc(all_scores, all_labels)
    global_auc_pr  = _auc_pr(all_scores, all_labels)
    hit_at_n       = sum(hit_at_n_counts) / len(hit_at_n_counts) if hit_at_n_counts else 0.0
    mrr            = sum(reciprocal_ranks)  / len(reciprocal_ranks)  if reciprocal_ranks  else 0.0

    logger.info("\n" + "=" * 70)
    logger.info("REPODB BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Diseases tested: {len(per_disease_results)}")
    logger.info(f"  Global AUC-ROC:  {global_auc_roc:.4f}")
    logger.info(f"  Global AUC-PR:   {global_auc_pr:.4f}")
    logger.info(f"  Hit@{top_n}:         {hit_at_n:.4f}")
    logger.info(f"  MRR:             {mrr:.4f}")

    output = {
        "summary": {
            "n_diseases_tested":  len(per_disease_results),
            "n_drugs":            len(drugs_data),
            "top_n":              top_n,
            "global_auc_roc":     round(global_auc_roc, 4),
            "global_auc_pr":      round(global_auc_pr, 4),
            f"hit_at_{top_n}":   round(hit_at_n, 4),
            "mrr":                round(mrr, 4),
        },
        "per_disease_results": per_disease_results,
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
        description="Benchmark drug repurposing pipeline against RepoDB"
    )
    parser.add_argument(
        "--repodb-file",
        type=str,
        default="repodb.csv",
        help="Path to RepoDB CSV file (default: repodb.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="N for Hit@N metric (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="repodb_benchmark_results.json",
        help="Output JSON file path (default: repodb_benchmark_results.json)",
    )
    parser.add_argument(
        "--max-diseases",
        type=int,
        default=None,
        help="Limit number of diseases to benchmark (useful for testing)",
    )
    args = parser.parse_args()

    if not Path(args.repodb_file).exists():
        logger.error(f"RepoDB file not found: {args.repodb_file}")
        logger.error("Download from: https://unmtid-shinyapps.net/shiny/repodb/")
        sys.exit(1)

    try:
        asyncio.run(
            run_benchmark(
                repodb_path=args.repodb_file,
                top_n=args.top_n,
                output_path=args.output,
                max_diseases=args.max_diseases,
            )
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()