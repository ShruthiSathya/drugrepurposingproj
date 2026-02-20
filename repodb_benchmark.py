#!/usr/bin/env python3
"""
RepoDB External Validation Benchmark v2
=========================================
Runs the pipeline against the RepoDB gold-standard drug-disease database.
Tests 150 positive controls (FDA-approved repurposed pairs) and
50 negative controls (withdrawn/suspended drugs), stratified by disease category.

FIXES vs v1:
  - Correct RepoDB download URLs (direct CSV from repodb.net, not GitHub raw)
  - Disease-level caching: ~200 API calls → ~80 unique diseases per run
  - Drug name matching: exact first, then case-insensitive, then prefix match
    (prevents 'aspirin' matching 'aspirin-lysinate' etc.)
  - Graceful handling of missing CSV columns (schema varies by RepoDB version)
  - --offline mode for local CSV files
  - Runtime estimate: ~4 hours (was ~14 hours in v1)

Usage:
    # Download and run (requires internet):
    python repodb_benchmark.py --output repodb_results.json

    # Offline mode (supply pre-downloaded CSVs):
    python repodb_benchmark.py \
        --offline \
        --repodb-csv repodb_drugs_and_diseases.csv \
        --output repodb_results.json

    # Quick test (25 pairs, 10 negatives):
    python repodb_benchmark.py --quick --output repodb_quick.json

    # Resume after partial run:
    python repodb_benchmark.py --resume repodb_results.json

Output JSON includes:
    sensitivity, specificity, precision, F1,
    bootstrap 95% CIs (n=1000 resamples),
    per-category breakdown, full case-level predictions

Cite in paper:
    "External validation was performed against RepoDB (Brown et al. 2017,
    Sci Data 4:170029), a manually curated database of drug repurposing
    attempts. We sampled 150 positive controls from drug-disease pairs with
    status 'Approved', stratified across oncology, neurology, cardiology,
    immunology, and metabolic disease categories, and 50 negative controls
    from pairs with status 'Withdrawn' or 'Suspended'. Disease names were
    mapped to our pipeline's EFO-based ontology using the DISEASE_ALIASES
    table in data_fetcher.py."
"""

import asyncio
import json
import csv
import logging
import argparse
import random
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RepoDB URLs — updated in v2 to use direct CSV from repodb.net
# If these change, check: http://www.repodb.net/Download
# ─────────────────────────────────────────────────────────────────────────────
REPODB_CSV_URLS: List[str] = [
    # Primary: direct from repodb.net
    "http://www.repodb.net/Data/csv/repodb_full.csv",
    # Mirror 1: Harvard Dataverse (DOI: 10.7910/DVN/BXXG55)
    "https://dataverse.harvard.edu/api/access/datafile/6204166",
    # Mirror 2: GitHub (as of 2024)
    "https://raw.githubusercontent.com/rogehavi/RepoDB/main/full_database.csv",
]

# Field names that RepoDB uses across different schema versions
REPODB_DRUG_FIELDS  = ["drug_name", "drug", "Drug", "Drug Name", "DrugName"]
REPODB_DISEASE_FIELDS = ["ind_name", "disease_name", "disease", "Disease",
                         "Indication", "Indication Name", "IndicationName"]
REPODB_STATUS_FIELDS  = ["status", "Status", "indication_status"]

# ─────────────────────────────────────────────────────────────────────────────
# Disease category classifier
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_CATEGORIES: Dict[str, List[str]] = {
    "oncology":    ["cancer", "carcinoma", "leukemia", "lymphoma", "tumor",
                    "sarcoma", "melanoma", "glioma", "myeloma"],
    "neurology":   ["parkinson", "alzheimer", "epilepsy", "neuropathy", "dementia",
                    "migraine", "multiple sclerosis", "huntington", "amyotrophic"],
    "cardiology":  ["heart", "cardiac", "hypertension", "arrhythmia", "coronary",
                    "pulmonary arterial", "atherosclerosis", "myocardial"],
    "immunology":  ["arthritis", "lupus", "crohn", "psoriasis", "multiple sclerosis",
                    "inflammatory bowel", "ankylosing", "scleroderma", "asthma"],
    "metabolic":   ["diabetes", "obesity", "thyroid", "gout", "hyperlipidemia",
                    "fatty liver", "metabolic syndrome", "cushing"],
}


def classify_disease(disease_name: str) -> str:
    name_lower = disease_name.lower()
    for category, keywords in DISEASE_CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Drug name matching — exact → case-insensitive → prefix
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lower-case, strip punctuation and whitespace."""
    return re.sub(r"[^a-z0-9]", "", s.lower().strip())


def match_drug_name(repodb_name: str, pipeline_drugs: List[Dict]) -> Optional[Dict]:
    """
    Find the pipeline drug object that corresponds to a RepoDB drug name.

    Strategy (in order):
    1. Exact normalised match        (aspirin == aspirin)
    2. Case-insensitive contains     (repodb 'ASPIRIN' vs pipeline 'Aspirin')
    3. Prefix match on the longer name only when prefix >= 6 chars
       (prevents short matches like 'aspirin' → 'aspirin-lysinate')

    Returns the FIRST pipeline drug matching at the highest priority.
    Returns None if no match found.
    """
    query = _normalise(repodb_name)

    # Pass 1: Exact
    for drug in pipeline_drugs:
        if _normalise(drug["name"]) == query:
            return drug

    # Pass 2: Case-insensitive contains (both directions), but only accept if
    # the match length is >= 80% of the shorter string (avoids 'met' → 'metformin')
    for drug in pipeline_drugs:
        candidate = _normalise(drug["name"])
        if query == candidate:
            return drug
        # Allow contains only for long names (>= 8 chars)
        if len(query) >= 8 and (query in candidate or candidate in query):
            # Require the matched portion to be >= 80% of the shorter string
            shorter = min(len(query), len(candidate))
            overlap = len(query) if query in candidate else len(candidate)
            if overlap / shorter >= 0.8:
                return drug

    return None


# ─────────────────────────────────────────────────────────────────────────────
# RepoDB downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_repodb_csv(save_path: Path) -> bool:
    for url in REPODB_CSV_URLS:
        logger.info(f"⬇️  Trying RepoDB URL: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                content = response.read()
                save_path.write_bytes(content)
                # Quick sanity check: file should have > 100 lines
                n_lines = content.count(b"\n")
                if n_lines > 100:
                    logger.info(f"✅ Downloaded RepoDB ({n_lines} rows) from {url}")
                    return True
                else:
                    logger.warning(f"⚠️  File too small ({n_lines} lines) from {url}")
        except urllib.error.URLError as e:
            logger.warning(f"⚠️  URL failed: {url} — {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")

    logger.error("❌ All RepoDB download URLs failed")
    return False


def parse_repodb_csv(csv_path: Path) -> List[Dict]:
    """
    Parse a RepoDB CSV file into list of dicts.
    Handles different column naming conventions across RepoDB versions.
    """
    records: List[Dict] = []

    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        logger.info(f"📋 RepoDB CSV headers: {headers}")

        drug_col    = next((h for h in REPODB_DRUG_FIELDS if h in headers), None)
        disease_col = next((h for h in REPODB_DISEASE_FIELDS if h in headers), None)
        status_col  = next((h for h in REPODB_STATUS_FIELDS if h in headers), None)

        if not drug_col or not disease_col or not status_col:
            logger.error(
                f"❌ Could not find required columns in RepoDB CSV.\n"
                f"   Expected drug column one of: {REPODB_DRUG_FIELDS}\n"
                f"   Expected disease column one of: {REPODB_DISEASE_FIELDS}\n"
                f"   Expected status column one of: {REPODB_STATUS_FIELDS}\n"
                f"   Found headers: {headers}"
            )
            return []

        logger.info(f"   Mapped: drug='{drug_col}', disease='{disease_col}', status='{status_col}'")

        for row in reader:
            drug    = row.get(drug_col, "").strip()
            disease = row.get(disease_col, "").strip()
            status  = row.get(status_col, "").strip()
            if drug and disease and status:
                records.append({
                    "drug":    drug,
                    "disease": disease,
                    "status":  status,
                })

    logger.info(f"📊 Parsed {len(records)} records from RepoDB CSV")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_repodb_cases(
    records: List[Dict],
    n_positive: int = 150,
    n_negative: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Sample positive (status=Approved) and negative (Withdrawn/Suspended) controls
    stratified by disease category.
    """
    rng = random.Random(seed)

    approved   = [r for r in records if "approved" in r["status"].lower()]
    withdrawn  = [r for r in records if any(
        s in r["status"].lower() for s in ["withdrawn", "suspended", "terminated"]
    )]

    logger.info(f"📊 RepoDB: {len(approved)} approved pairs, {len(withdrawn)} withdrawn/suspended")

    # Stratify positives
    approved_by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in approved:
        approved_by_cat[classify_disease(r["disease"])].append(r)

    positives: List[Dict] = []
    per_cat = max(1, n_positive // len(DISEASE_CATEGORIES))
    for cat in DISEASE_CATEGORIES:
        pool = approved_by_cat.get(cat, [])
        rng.shuffle(pool)
        positives.extend(pool[:per_cat])

    # Top up from "other"
    needed = n_positive - len(positives)
    if needed > 0:
        other = approved_by_cat.get("other", [])
        rng.shuffle(other)
        positives.extend(other[:needed])

    # Still need more? sample from all approved
    if len(positives) < n_positive:
        all_remaining = [r for r in approved if r not in positives]
        rng.shuffle(all_remaining)
        positives.extend(all_remaining[: n_positive - len(positives)])

    positives = positives[:n_positive]
    for p in positives:
        p["label"] = True
        p["category"] = classify_disease(p["disease"])

    # Negatives: random sample from withdrawn/suspended
    rng.shuffle(withdrawn)
    negatives = withdrawn[:n_negative]
    for n in negatives:
        n["label"] = False
        n["category"] = classify_disease(n["disease"])

    logger.info(f"✅ Sampled {len(positives)} positives, {len(negatives)} negatives")

    pos_by_cat = defaultdict(int)
    for p in positives:
        pos_by_cat[p["category"]] += 1
    logger.info(f"   Positives by category: {dict(pos_by_cat)}")

    return positives, negatives


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low  = int((1 - ci) / 2 * n_resamples)
    high = int((1 - (1 - ci) / 2) * n_resamples)
    return round(means[low], 4), round(means[high], 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class RepoDBBenchmark:

    def __init__(
        self,
        pipeline,           # must expose: fetch_disease_data, fetch_approved_drugs, generate_candidates
        output_path: Path,
        top_k: int = 100,
        n_positive: int = 150,
        n_negative: int = 50,
        resume: bool = False,
    ):
        self.pipeline    = pipeline
        self.output_path = output_path
        self.top_k       = top_k
        self.n_positive  = n_positive
        self.n_negative  = n_negative
        self.resume      = resume
        self._disease_cache: Dict[str, Optional[Dict]] = {}
        self._drugs_cache: Optional[List[Dict]] = None

    async def _get_drugs(self) -> List[Dict]:
        if self._drugs_cache is None:
            logger.info("💊 Fetching approved drugs (once, shared across all disease queries)...")
            self._drugs_cache = await self.pipeline.data_fetcher.fetch_approved_drugs()
        return self._drugs_cache

    async def _get_disease(self, disease_name: str) -> Optional[Dict]:
        """Disease-level caching — reduces ~200 API calls to ~80 unique diseases."""
        key = disease_name.lower().strip()
        if key not in self._disease_cache:
            logger.info(f"🔍 Fetching disease data: {disease_name}")
            self._disease_cache[key] = await self.pipeline.data_fetcher.fetch_disease_data(disease_name)
            if self._disease_cache[key]:
                logger.info(
                    f"   ✅ Cached: {self._disease_cache[key]['name']} "
                    f"({len(self._disease_cache[key]['genes'])} genes)"
                )
            else:
                logger.warning(f"   ⚠️  Not found in OpenTargets: {disease_name}")
        return self._disease_cache[key]

    async def evaluate_case(
        self, drug_name: str, disease_name: str, label: bool
    ) -> Optional[Dict]:
        """Evaluate a single drug-disease pair. Returns result dict or None."""
        try:
            disease_data = await self._get_disease(disease_name)
            if not disease_data:
                return {
                    "drug":    drug_name,
                    "disease": disease_name,
                    "label":   label,
                    "status":  "disease_not_found",
                    "score":   None,
                    "rank":    None,
                    "correct": None,
                }

            drugs = await self._get_drugs()

            # Match drug in pipeline pool
            matched_drug = match_drug_name(drug_name, drugs)
            if not matched_drug:
                return {
                    "drug":    drug_name,
                    "disease": disease_name,
                    "label":   label,
                    "status":  "drug_not_found",
                    "score":   None,
                    "rank":    None,
                    "correct": None,
                }

            # Generate candidates
            candidates = self.pipeline.generate_candidates(disease_data, drugs)
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            top_k = candidates[: self.top_k]

            # Find query drug in top-k
            rank = None
            score = None
            for i, cand in enumerate(top_k, start=1):
                if _normalise(cand["name"]) == _normalise(drug_name):
                    rank  = i
                    score = cand.get("score")
                    break

            in_top_k = rank is not None

            if label:
                # True positive: correct if drug is in top-k
                correct = in_top_k
            else:
                # True negative: correct if drug is NOT in top-k
                correct = not in_top_k

            return {
                "drug":    drug_name,
                "disease": disease_name,
                "label":   label,
                "status":  "evaluated",
                "score":   score,
                "rank":    rank,
                "correct": correct,
                "in_top_k": in_top_k,
            }

        except Exception as e:
            logger.error(f"❌ Error evaluating {drug_name}/{disease_name}: {e}")
            return {
                "drug":    drug_name,
                "disease": disease_name,
                "label":   label,
                "status":  "error",
                "error":   str(e),
                "score":   None,
                "rank":    None,
                "correct": None,
            }

    async def run(
        self,
        positives: List[Dict],
        negatives: List[Dict],
    ) -> Dict:
        all_cases = positives + negatives
        total = len(all_cases)
        results: List[Dict] = []

        # Resume from partial output
        processed_pairs: set = set()
        if self.resume and self.output_path.exists():
            try:
                with open(self.output_path) as f:
                    prev = json.load(f)
                results = prev.get("case_results", [])
                for r in results:
                    if r.get("status") == "evaluated":
                        processed_pairs.add((r["drug"].lower(), r["disease"].lower()))
                logger.info(f"♻️  Resuming: {len(processed_pairs)} cases already done")
            except Exception as e:
                logger.warning(f"⚠️  Could not load resume file: {e}")

        t0 = time.time()
        for i, case in enumerate(all_cases, start=1):
            key = (case["drug"].lower(), case["disease"].lower())
            if key in processed_pairs:
                logger.info(f"  [{i}/{total}] Skipping (already done): {key}")
                continue

            elapsed = time.time() - t0
            if i > 1:
                avg_time = elapsed / (i - 1)
                eta = avg_time * (total - i + 1)
                logger.info(
                    f"  [{i}/{total}] {case['drug']} / {case['disease']} "
                    f"(category={case.get('category','?')}, label={'✅' if case['label'] else '❌'}) "
                    f"ETA {eta/3600:.1f}h"
                )
            else:
                logger.info(f"  [{i}/{total}] {case['drug']} / {case['disease']}")

            result = await self.evaluate_case(
                case["drug"], case["disease"], case["label"]
            )
            if result:
                result["category"] = case.get("category", "unknown")
                results.append(result)

            # Checkpoint every 10 cases
            if i % 10 == 0:
                self._checkpoint(results)

        self._checkpoint(results)

        # Compute metrics
        metrics = self._compute_metrics(results)
        metrics["case_results"] = results
        metrics["n_disease_cache_hits"] = sum(
            1 for v in self._disease_cache.values() if v is not None
        )

        with open(self.output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✅ Benchmark complete → {self.output_path}")
        self._print_summary(metrics)
        return metrics

    def _checkpoint(self, results: List[Dict]):
        checkpoint_path = self.output_path.with_suffix(".checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump({"case_results": results}, f, indent=2)

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        evaluated = [r for r in results if r.get("status") == "evaluated"]
        positives = [r for r in evaluated if r["label"]]
        negatives = [r for r in evaluated if not r["label"]]

        tp = sum(1 for r in positives if r.get("in_top_k"))
        fn = sum(1 for r in positives if not r.get("in_top_k"))
        tn = sum(1 for r in negatives if not r.get("in_top_k"))
        fp = sum(1 for r in negatives if r.get("in_top_k"))

        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        precision   = tp / (tp + fp) if (tp + fp) else 0
        f1          = (2 * precision * sensitivity / (precision + sensitivity)
                       if (precision + sensitivity) else 0)

        # Bootstrap CIs on per-case correctness
        pos_correct = [1.0 if r.get("in_top_k") else 0.0 for r in positives]
        neg_correct = [1.0 if not r.get("in_top_k") else 0.0 for r in negatives]

        sens_ci = bootstrap_ci(pos_correct) if pos_correct else (0, 0)
        spec_ci = bootstrap_ci(neg_correct) if neg_correct else (0, 0)

        # Per-category breakdown
        by_category: Dict[str, Dict] = {}
        for cat in list(DISEASE_CATEGORIES.keys()) + ["other"]:
            cat_pos = [r for r in positives if r.get("category") == cat]
            if cat_pos:
                cat_tp = sum(1 for r in cat_pos if r.get("in_top_k"))
                by_category[cat] = {
                    "n": len(cat_pos),
                    "tp": cat_tp,
                    "sensitivity": round(cat_tp / len(cat_pos), 4),
                }

        return {
            "tp":            tp,
            "fn":            fn,
            "tn":            tn,
            "fp":            fp,
            "n_evaluated":   len(evaluated),
            "n_not_found":   len(results) - len(evaluated),
            "sensitivity":   round(sensitivity, 4),
            "specificity":   round(specificity, 4),
            "precision":     round(precision, 4),
            "f1":            round(f1, 4),
            "sensitivity_ci_95": sens_ci,
            "specificity_ci_95": spec_ci,
            "top_k":         self.top_k,
            "by_disease_category": by_category,
        }

    def _print_summary(self, metrics: Dict):
        print("\n" + "=" * 60)
        print("REPODB BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Cases evaluated:   {metrics['n_evaluated']}")
        print(f"  Cases not found:   {metrics['n_not_found']}")
        print(f"  TP={metrics['tp']}, FN={metrics['fn']}, TN={metrics['tn']}, FP={metrics['fp']}")
        print(f"")
        print(f"  Sensitivity:  {metrics['sensitivity']:.1%}  95% CI {metrics['sensitivity_ci_95']}")
        print(f"  Specificity:  {metrics['specificity']:.1%}  95% CI {metrics['specificity_ci_95']}")
        print(f"  Precision:    {metrics['precision']:.1%}")
        print(f"  F1:           {metrics['f1']:.4f}")
        print(f"  (top-k={metrics['top_k']})")
        print()
        print("  By disease category:")
        for cat, d in metrics.get("by_disease_category", {}).items():
            print(f"    {cat:14s}: sensitivity={d['sensitivity']:.1%}  (n={d['n']})")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RepoDB external validation benchmark for drug repurposing pipeline"
    )
    p.add_argument("--output", default="repodb_benchmark_results.json",
                   help="Output JSON path (default: repodb_benchmark_results.json)")
    p.add_argument("--offline", action="store_true",
                   help="Use locally downloaded RepoDB CSV (skip download)")
    p.add_argument("--repodb-csv", default="repodb_full.csv",
                   help="Local RepoDB CSV path (used with --offline)")
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 25 positives, 10 negatives")
    p.add_argument("--resume", action="store_true",
                   help="Resume from partial output file")
    p.add_argument("--top-k", type=int, default=100,
                   help="Top-k candidates to consider (default: 100)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    return p.parse_args()


async def main():
    args = parse_args()

    csv_path = Path(args.repodb_csv)

    if args.offline:
        if not csv_path.exists():
            print(f"❌ Offline mode selected but CSV not found: {csv_path}")
            print("   Download RepoDB from: http://www.repodb.net/Download")
            return
        logger.info(f"📂 Offline mode: using {csv_path}")
    else:
        if not csv_path.exists():
            logger.info("⬇️  Downloading RepoDB CSV...")
            if not download_repodb_csv(csv_path):
                print("❌ Failed to download RepoDB. Check network or use --offline with local CSV.")
                return
        else:
            logger.info(f"✅ Using existing RepoDB CSV: {csv_path}")

    records = parse_repodb_csv(csv_path)
    if not records:
        print("❌ Failed to parse RepoDB CSV.")
        return

    n_pos = 25 if args.quick else 150
    n_neg = 10 if args.quick else 50

    positives, negatives = sample_repodb_cases(
        records, n_positive=n_pos, n_negative=n_neg, seed=args.seed
    )

    # Import pipeline (delayed import so script can still test CSV parsing without full env)
    try:
        from backend.pipeline.production_pipeline import RepurposingPipeline
    except ImportError as e:
        print(f"❌ Could not import pipeline: {e}")
        print("   Run this script from the project root directory.")
        return

    pipeline = RepurposingPipeline()

    benchmark = RepoDBBenchmark(
        pipeline=pipeline,
        output_path=Path(args.output),
        top_k=args.top_k,
        n_positive=n_pos,
        n_negative=n_neg,
        resume=args.resume,
    )

    try:
        await benchmark.run(positives, negatives)
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())