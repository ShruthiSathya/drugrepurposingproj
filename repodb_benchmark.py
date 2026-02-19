"""
RepoDB Standardized Benchmark Integration
==========================================
Replaces hand-curated validation set with the RepoDB gold standard:
  Brown & Patel (2017) Sci Data 4:170029. PMID: 28291243
  URL: https://repodb.net

RepoDB contains 4,271 drug-disease pairs with FDA approval status.
We use the "approved" subset as positive controls and a stratified
random sample of "withdrawn/suspended" pairs as negative controls.

This directly addresses Reviewer Comment:
  "The benchmark is too small (34 cases) and may be biased toward
   cases the algorithm was tuned on."

Usage:
  python repodb_benchmark.py --download   # fetches RepoDB CSV
  python repodb_benchmark.py --run        # runs full evaluation
  python repodb_benchmark.py --sample 100 # runs stratified 100-pair sample
"""

import asyncio
import csv
import json
import logging
import random
import ssl
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RepoDB download URL (direct CSV, no auth required)
# ─────────────────────────────────────────────────────────────────────────────
REPODB_APPROVED_URL = (
    "https://raw.githubusercontent.com/rsinghlab/repodb/"
    "master/data/approved.csv"
)
REPODB_WITHDRAWN_URL = (
    "https://raw.githubusercontent.com/rsinghlab/repodb/"
    "master/data/withdrawn.csv"
)


def _make_ssl_context() -> ssl.SSLContext:
    """Create SSL context, using certifi if available, otherwise system certs."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass
    try:
        ctx = ssl.create_default_context()
        return ctx
    except Exception:
        return False  # aiohttp accepts False to disable verification


class RepoDBBenchmark:
    """
    Downloads RepoDB and runs standardized evaluation.

    Positive controls:  FDA-approved drug-disease pairs (RepoDB 'approved')
    Negative controls:  withdrawn/suspended pairs (mechanistically wrong fits)

    Stratification ensures disease-category balance across:
      oncology, neurology, cardiology, immunology, metabolic, other
    """

    CACHE_DIR = Path("/tmp/repodb_cache")

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.approved:  List[Dict] = []
        self.withdrawn: List[Dict] = []

    # ── Download ──────────────────────────────────────────────────────────────

    async def download(self) -> None:
        """Download RepoDB CSV files if not already cached."""
        ssl_ctx = _make_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as session:
            for url, name in [
                (REPODB_APPROVED_URL,  "approved.csv"),
                (REPODB_WITHDRAWN_URL, "withdrawn.csv"),
            ]:
                dest = self.CACHE_DIR / name
                if dest.exists():
                    logger.info(f"✅ {name} already cached")
                    continue
                logger.info(f"📥 Downloading RepoDB {name}...")
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
                dest.write_text(text)
                logger.info(f"✅ Saved {name} ({len(text)} bytes)")

    def load(self) -> None:
        """Load downloaded RepoDB CSVs into memory."""
        for attr, name in [("approved", "approved.csv"), ("withdrawn", "withdrawn.csv")]:
            path = self.CACHE_DIR / name
            if not path.exists():
                raise FileNotFoundError(f"RepoDB file not found: {path}. Run download() first.")
            rows = []
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            setattr(self, attr, rows)
            logger.info(f"✅ Loaded {len(rows)} {attr} pairs from RepoDB")

    # ── Disease name normalisation ────────────────────────────────────────────

    def _normalize_disease(self, raw_name: str) -> str:
        """
        Map RepoDB disease names to OpenTargets-compatible names.
        RepoDB uses MeSH terms; OpenTargets uses EFO names.
        This mapping covers the most common divergences.
        """
        mapping = {
            "Carcinoma, Non-Small-Cell Lung": "non-small cell lung carcinoma",
            "Carcinoma, Hepatocellular":      "hepatocellular carcinoma",
            "Colorectal Neoplasms":           "colorectal cancer",
            "Breast Neoplasms":               "breast carcinoma",
            "Leukemia, Myeloid, Acute":       "acute myeloid leukemia",
            "Alzheimer Disease":              "Alzheimer disease",
            "Parkinson Disease":              "Parkinson disease",
            "Diabetes Mellitus, Type 2":      "type 2 diabetes mellitus",
            "Arthritis, Rheumatoid":          "rheumatoid arthritis",
            "Hypertension":                   "essential hypertension",
            "Heart Failure":                  "heart failure",
            "Multiple Sclerosis":             "multiple sclerosis",
            "Asthma":                         "asthma",
            "Schizophrenia":                  "schizophrenia",
            "Depressive Disorder, Major":     "major depressive disorder",
            "HIV Infections":                 "human immunodeficiency virus infectious disease",
            "Lupus Erythematosus, Systemic":  "systemic lupus erythematosus",
            "Crohn Disease":                  "Crohn's disease",
            "Psoriasis":                      "psoriasis",
            "Myeloma, Multiple":              "multiple myeloma",
            "Lymphoma, Non-Hodgkin":          "non-Hodgkin lymphoma",
            "Melanoma":                       "melanoma",
            "Glioblastoma":                   "glioblastoma",
            "Ovarian Neoplasms":              "ovarian carcinoma",
            "Prostatic Neoplasms":            "prostate carcinoma",
            "Epilepsy":                       "epilepsy",
            "Migraine Disorders":             "migraine",
            "Fibromyalgia":                   "fibromyalgia",
            "Osteoporosis":                   "osteoporosis",
            "Gout":                           "gout",
            "Atrial Fibrillation":            "atrial fibrillation",
            "Obesity":                        "obesity",
            "Pulmonary Arterial Hypertension":"pulmonary arterial hypertension",
        }
        return mapping.get(raw_name, raw_name.lower())

    # ── Stratified sampling ───────────────────────────────────────────────────

    DISEASE_CATEGORIES = {
        "oncology":    ["carcinoma", "cancer", "leukemia", "lymphoma", "melanoma",
                        "myeloma", "neoplasm", "glioblastoma", "tumor"],
        "neurology":   ["alzheimer", "parkinson", "epilepsy", "migraine", "sclerosis",
                        "schizophrenia", "depression", "depressive", "dementia"],
        "cardiology":  ["hypertension", "heart", "atrial", "coronary", "cardiac",
                        "arterial", "vascular", "arrhythmia"],
        "immunology":  ["arthritis", "lupus", "crohn", "psoriasis", "asthma",
                        "inflammatory", "autoimmune"],
        "metabolic":   ["diabetes", "obesity", "gout", "osteoporosis", "thyroid"],
    }

    def _categorize(self, disease_name: str) -> str:
        name_lower = disease_name.lower()
        for cat, keywords in self.DISEASE_CATEGORIES.items():
            if any(kw in name_lower for kw in keywords):
                return cat
        return "other"

    def get_stratified_sample(
        self,
        n_positive: int = 150,
        n_negative: int = 50,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        rng = random.Random(seed)

        def _sample_stratified(pairs: List[Dict], n: int) -> List[Dict]:
            buckets: Dict[str, List[Dict]] = {}
            for pair in pairs:
                cat = self._categorize(pair.get("ind_name", ""))
                buckets.setdefault(cat, []).append(pair)
            total = len(pairs)
            allocated: Dict[str, int] = {}
            remaining = n
            cats = list(buckets.keys())
            for cat in cats[:-1]:
                quota = max(1, round(n * len(buckets[cat]) / total))
                allocated[cat] = min(quota, len(buckets[cat]))
                remaining -= allocated[cat]
            allocated[cats[-1]] = min(remaining, len(buckets[cats[-1]]))
            sample = []
            for cat, quota in allocated.items():
                sample.extend(rng.sample(buckets[cat], quota))
            return sample

        positives = _sample_stratified(self.approved,  n_positive)
        negatives = _sample_stratified(self.withdrawn,  n_negative)

        pos_cases = [
            {
                "drug_name":            row["drug_name"],
                "repurposed_for":       self._normalize_disease(row["ind_name"]),
                "expected_score_range": (0.15, 0.85),
                "category":             self._categorize(row["ind_name"]),
                "source":               "RepoDB",
                "repodb_status":        "approved",
            }
            for row in positives
        ]

        neg_cases = [
            {
                "drug_name":             row["drug_name"],
                "disease":               self._normalize_disease(row["ind_name"]),
                "expected_score_range":  (0.0, 0.30),
                "reason":                f"RepoDB withdrawn/suspended: {row.get('detail', '')}",
                "source":                "RepoDB",
                "repodb_status":         "withdrawn",
            }
            for row in negatives
        ]

        logger.info(f"📊 Stratified sample: {len(pos_cases)} positive, {len(neg_cases)} negative")
        return pos_cases, neg_cases

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(
        self,
        pipeline,
        n_positive: int = 150,
        n_negative: int = 50,
        top_k: int = 100,
        seed: int = 42,
    ) -> Dict:
        pos_cases, neg_cases = self.get_stratified_sample(n_positive, n_negative, seed)

        results = {
            "positive": [],
            "negative": [],
            "config": {
                "n_positive": n_positive,
                "n_negative": n_negative,
                "top_k": top_k,
                "seed": seed,
                "source": "RepoDB",
            }
        }

        logger.info(f"🔬 Evaluating {len(pos_cases)} positive RepoDB pairs...")
        for i, case in enumerate(pos_cases, 1):
            logger.info(f"  [{i}/{len(pos_cases)}] {case['drug_name']} → {case['repurposed_for']}")
            try:
                result = await pipeline.analyze_disease(
                    case["repurposed_for"], min_score=0.0, max_results=top_k
                )
                if not result.get("success"):
                    results["positive"].append({**case, "status": "disease_not_found"})
                    continue
                found = next(
                    (c for c in result["candidates"]
                     if case["drug_name"].lower() in c["drug_name"].lower()),
                    None,
                )
                results["positive"].append({
                    **case,
                    "status": "tp" if found else "fn",
                    "score":  found["score"] if found else None,
                })
            except Exception as e:
                results["positive"].append({**case, "status": "error", "error": str(e)})

        logger.info(f"🔬 Evaluating {len(neg_cases)} negative RepoDB pairs...")
        for i, case in enumerate(neg_cases, 1):
            logger.info(f"  [{i}/{len(neg_cases)}] {case['drug_name']} ✗ {case['disease']}")
            try:
                result = await pipeline.analyze_disease(
                    case["disease"], min_score=0.0, max_results=top_k
                )
                if not result.get("success"):
                    results["negative"].append({**case, "status": "disease_not_found"})
                    continue
                found = next(
                    (c for c in result["candidates"]
                     if case["drug_name"].lower() in c["drug_name"].lower()),
                    None,
                )
                results["negative"].append({
                    **case,
                    "status": "fp" if found else "tn",
                    "score":  found["score"] if found else None,
                })
            except Exception as e:
                results["negative"].append({**case, "status": "error", "error": str(e)})

        tp = sum(1 for r in results["positive"] if r["status"] == "tp")
        fn = sum(1 for r in results["positive"] if r["status"] == "fn")
        tn = sum(1 for r in results["negative"] if r["status"] == "tn")
        fp = sum(1 for r in results["negative"] if r["status"] == "fp")

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

        results["metrics"] = {
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sens,
            "specificity": spec,
            "precision":   prec,
            "f1":          f1,
            "n_disease_not_found": sum(
                1 for r in results["positive"] + results["negative"]
                if r["status"] == "disease_not_found"
            ),
        }

        cat_results: Dict[str, Dict] = {}
        for r in results["positive"]:
            cat = r.get("category", "other")
            cat_results.setdefault(cat, {"tp": 0, "fn": 0})
            if r["status"] == "tp":
                cat_results[cat]["tp"] += 1
            elif r["status"] == "fn":
                cat_results[cat]["fn"] += 1
        results["per_category"] = {
            cat: {
                **counts,
                "sensitivity": counts["tp"] / (counts["tp"] + counts["fn"])
                if (counts["tp"] + counts["fn"]) > 0 else 0
            }
            for cat, counts in cat_results.items()
        }

        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="RepoDB benchmark for drug repurposing")
    parser.add_argument("--download", action="store_true",
                        help="Download RepoDB CSV files")
    parser.add_argument("--run",      action="store_true",
                        help="Run full evaluation")
    parser.add_argument("--sample",   type=int, default=150,
                        help="Number of positive cases to sample (default: 150)")
    parser.add_argument("--negatives",type=int, default=50,
                        help="Number of negative cases to sample (default: 50)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    bench = RepoDBBenchmark()

    if args.download:
        await bench.download()
        print("✅ RepoDB downloaded")

    if args.run:
        bench.load()

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from backend.pipeline.production_pipeline import ProductionPipeline
        pipeline = ProductionPipeline()
        await pipeline.data_fetcher.fetch_approved_drugs(limit=3000)

        results = await bench.evaluate(
            pipeline,
            n_positive=args.sample,
            n_negative=args.negatives,
            seed=args.seed,
        )

        out_path = Path("repodb_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        m = results["metrics"]
        print(f"\n{'='*60}")
        print(f"RepoDB Benchmark Results  (n={args.sample}+{args.negatives})")
        print(f"{'='*60}")
        print(f"  Sensitivity: {m['sensitivity']:.1%}")
        print(f"  Specificity: {m['specificity']:.1%}")
        print(f"  Precision:   {m['precision']:.1%}")
        print(f"  F1:          {m['f1']:.1%}")
        print(f"  TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}")
        print(f"  Disease-not-found: {m['n_disease_not_found']}")
        print(f"\nPer-category sensitivity:")
        for cat, v in results["per_category"].items():
            print(f"  {cat}: {v['tp']}/{v['tp']+v['fn']} = {v['sensitivity']:.1%}")
        print(f"\nResults saved → {out_path}")

        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())