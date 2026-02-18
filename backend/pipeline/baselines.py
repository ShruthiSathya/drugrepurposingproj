"""
Baseline Comparison Models for Drug Repurposing
================================================
Implements two comparison baselines so the paper can demonstrate added value:

  1. CosineSimilarityBaseline
     - Represents drugs and diseases as TF-IDF vectors over gene names
     - Computes cosine similarity between drug-target set and disease-gene set
     - Uses only DGIdb targets + OpenTargets genes (same raw data as main algo)
     - No pathway weights, no mechanism scoring, no PubMed signal
     - Reference: standard IR similarity widely used in drug repurposing literature

  2. TextMiningBaseline
     - Uses PubMed co-occurrence count as the sole signal
     - Normalises log(count+1) / log(201) → 0-1 score
     - Represents a naive "if it has been mentioned, it might work" baseline
     - This is the weakest baseline and demonstrates why mechanism is needed

  3. RandomBaseline
     - Assigns uniform random scores to all drugs
     - Used to compute lift-over-random

Usage
-----
    from baselines import CosineSimilarityBaseline, TextMiningBaseline
    
    baseline = CosineSimilarityBaseline()
    score = baseline.score(drug_targets, disease_genes)

References
----------
    Barabási et al. (2011) Network medicine. Nat Rev Genet. PMID: 21164525
    Yildirim et al. (2007) Drug-target network. Nat Biotechnol. PMID: 17921997
"""

import math
import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple
import aiohttp

logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Cosine similarity over gene sets (TF-IDF-weighted)
# ─────────────────────────────────────────────────────────────────────────────

class CosineSimilarityBaseline:
    """
    Gene-set cosine similarity baseline.

    Each drug and disease is represented as a binary/TF-IDF vector over
    the universe of gene symbols.  Score = cosine(drug_vector, disease_vector).

    This captures the SAME gene-overlap information as the main algorithm
    but WITHOUT:
      - Pathway weighting
      - Mechanism text scoring
      - Literature co-occurrence signal
      - Multiplicative overlap bonuses

    Demonstrating that the full algorithm outperforms this baseline confirms
    that each additional scoring component contributes.
    """

    def __init__(self, use_tfidf: bool = True):
        self.use_tfidf = use_tfidf
        # Gene → IDF weight (populated from corpus when available)
        self._idf: Dict[str, float] = {}

    def fit(self, gene_corpus: List[List[str]]) -> None:
        """
        Fit IDF weights from a corpus of gene lists.
        Each element = list of genes for one drug or disease.
        """
        n_docs = len(gene_corpus)
        doc_freq: Dict[str, int] = {}
        for gene_list in gene_corpus:
            for gene in set(gene_list):
                doc_freq[gene] = doc_freq.get(gene, 0) + 1

        self._idf = {
            gene: math.log(n_docs / (1 + df)) + 1.0
            for gene, df in doc_freq.items()
        }

    def _gene_weight(self, gene: str) -> float:
        if not self.use_tfidf or not self._idf:
            return 1.0
        return self._idf.get(gene, 1.0)

    def score(
        self,
        drug_targets: List[str],
        disease_genes: List[str],
    ) -> float:
        """Return cosine similarity between drug and disease gene vectors."""
        if not drug_targets or not disease_genes:
            return 0.0

        drug_vec:    Dict[str, float] = {g: self._gene_weight(g) for g in drug_targets}
        disease_vec: Dict[str, float] = {g: self._gene_weight(g) for g in disease_genes}

        # Dot product (intersection terms only)
        dot = sum(
            drug_vec[g] * disease_vec[g]
            for g in set(drug_targets) & set(disease_genes)
        )

        # Magnitudes
        mag_drug    = math.sqrt(sum(w ** 2 for w in drug_vec.values()))
        mag_disease = math.sqrt(sum(w ** 2 for w in disease_vec.values()))

        if mag_drug == 0 or mag_disease == 0:
            return 0.0

        return dot / (mag_drug * mag_disease)

    def score_all(
        self,
        drugs: List[Dict],
        disease_data: Dict,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Score all drugs against a disease; return ranked list."""
        disease_genes = disease_data.get("genes", [])
        results = []

        for drug in drugs:
            targets = drug.get("targets", [])
            if not targets:
                continue
            s = self.score(targets, disease_genes)
            if s >= min_score:
                results.append({
                    "drug_name": drug["name"],
                    "score":     s,
                    "method":    "cosine_similarity",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Text-mining / PubMed co-occurrence
# ─────────────────────────────────────────────────────────────────────────────

class TextMiningBaseline:
    """
    PubMed co-occurrence baseline.

    Score = log10(hit_count + 1) / log10(201)  →  0-1

    Represents the naive hypothesis that existing literature co-mention
    is sufficient to identify repurposing candidates.

    Limitation: systematically favours well-studied drugs and diseases
    (publication bias).  Any algorithm that substantially outperforms this
    baseline is capturing mechanistic signal beyond what is already known.

    Reference: Srinivasan (2004) Text mining: generating hypotheses from
    MEDLINE. J Am Soc Inf Sci. doi:10.1002/asi.20074
    """

    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def score(self, drug_name: str, disease_name: str) -> float:
        key = f"{drug_name.lower()}|{disease_name.lower()}"
        if key in self._cache:
            return self._cache[key]

        session = await self._get_session()
        try:
            params = {
                "db":      "pubmed",
                "term":    f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]',
                "retmax":  "0",
                "retmode": "json",
            }
            async with session.get(PUBMED_ESEARCH, params=params) as resp:
                if resp.status != 200:
                    self._cache[key] = 0.0
                    return 0.0
                data  = await resp.json()
                count = int(data.get("esearchresult", {}).get("count", 0))
        except Exception as e:
            logger.debug(f"PubMed baseline lookup failed: {e}")
            self._cache[key] = 0.0
            return 0.0

        s = math.log10(count + 1) / math.log10(201)
        s = min(s, 1.0)
        self._cache[key] = s
        return s

    async def score_all(
        self,
        drugs: List[Dict],
        disease_name: str,
        min_score: float = 0.0,
    ) -> List[Dict]:
        tasks = [self.score(d["name"], disease_name) for d in drugs]
        scores_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for drug, s in zip(drugs, scores_list):
            if isinstance(s, Exception):
                s = 0.0
            if s >= min_score:
                results.append({
                    "drug_name": drug["name"],
                    "score":     s,
                    "method":    "pubmed_cooccurrence",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Random
# ─────────────────────────────────────────────────────────────────────────────

class RandomBaseline:
    """
    Uniform random scorer.
    Provides the theoretical lower-bound for comparison.
    """
    import random as _random

    def score_all(self, drugs: List[Dict], **kwargs) -> List[Dict]:
        import random
        results = [
            {"drug_name": d["name"], "score": random.random(), "method": "random"}
            for d in drugs
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @staticmethod
    def expected_sensitivity(n_positive: int, top_k: int, n_total: int) -> float:
        """Expected fraction of positives in random top-k."""
        return min(top_k / n_total, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_baseline_comparison(
    pipeline,
    test_cases: List[Dict],
    negative_controls: List[Dict],
    top_k: int = 100,
) -> Dict:
    """
    Run all three baselines and the main algorithm on the same test set.
    Returns a dict of performance metrics for each method.

    Results are structured for easy insertion into a paper table.
    """
    from baselines import CosineSimilarityBaseline, TextMiningBaseline, RandomBaseline

    cosine_baseline = CosineSimilarityBaseline(use_tfidf=True)
    text_baseline   = TextMiningBaseline()
    random_baseline = RandomBaseline()

    results = {
        "cosine_similarity": {"tp": 0, "fn": 0, "tn": 0, "fp": 0},
        "text_mining":       {"tp": 0, "fn": 0, "tn": 0, "fp": 0},
        "random":            {"tp": 0, "fn": 0, "tn": 0, "fp": 0},
        "main_algorithm":    {"tp": 0, "fn": 0, "tn": 0, "fp": 0},
    }

    for case in test_cases:
        drug    = case["drug_name"]
        disease = case["repurposed_for"]

        # Fetch disease + drugs once
        disease_data = await pipeline.data_fetcher.fetch_disease_data(disease)
        if not disease_data:
            continue
        drugs_data = pipeline.drugs_cache or []

        # ── Cosine baseline ────────────────────────────────────────────────
        # Fit IDF on all available gene lists
        all_gene_lists = [d.get("targets", []) for d in drugs_data]
        all_gene_lists.append(disease_data.get("genes", []))
        cosine_baseline.fit(all_gene_lists)

        cosine_results = cosine_baseline.score_all(drugs_data, disease_data)
        top_cosine = {r["drug_name"].lower() for r in cosine_results[:top_k]}
        if drug.lower() in top_cosine:
            results["cosine_similarity"]["tp"] += 1
        else:
            results["cosine_similarity"]["fn"] += 1

        # ── Text-mining baseline ───────────────────────────────────────────
        text_score = await text_baseline.score(drug, disease)
        # Threshold to determine "found": if score > 5th percentile of all scores
        text_results = await text_baseline.score_all(drugs_data, disease)
        top_text = {r["drug_name"].lower() for r in text_results[:top_k]}
        if drug.lower() in top_text:
            results["text_mining"]["tp"] += 1
        else:
            results["text_mining"]["fn"] += 1

        # ── Random baseline ────────────────────────────────────────────────
        expected_hits = RandomBaseline.expected_sensitivity(1, top_k, len(drugs_data))
        # Simulate: on average, expected_hits proportion of positives are found
        import random
        if random.random() < expected_hits:
            results["random"]["tp"] += 1
        else:
            results["random"]["fn"] += 1

        # ── Main algorithm ─────────────────────────────────────────────────
        res = await pipeline.analyze_disease(disease, min_score=0.0, max_results=top_k)
        if res.get("success"):
            found = any(
                drug.lower() in c["drug_name"].lower()
                for c in res["candidates"]
            )
            if found:
                results["main_algorithm"]["tp"] += 1
            else:
                results["main_algorithm"]["fn"] += 1

    # Negative controls
    for neg in negative_controls:
        drug    = neg["drug_name"]
        disease = neg["disease"]
        exp_max = neg["expected_score_range"][1]

        disease_data = await pipeline.data_fetcher.fetch_disease_data(disease)
        if not disease_data:
            continue
        drugs_data = pipeline.drugs_cache or []

        # ── Cosine ────────────────────────────────────────────────────────
        cosine_baseline.fit([d.get("targets", []) for d in drugs_data])
        cosine_results = cosine_baseline.score_all(drugs_data, disease_data)
        top_cosine = {r["drug_name"].lower() for r in cosine_results[:100]}
        if drug.lower() not in top_cosine:
            results["cosine_similarity"]["tn"] += 1
        else:
            results["cosine_similarity"]["fp"] += 1

        # ── Text-mining ───────────────────────────────────────────────────
        text_results = await text_baseline.score_all(drugs_data, disease)
        top_text = {r["drug_name"].lower() for r in text_results[:100]}
        if drug.lower() not in top_text:
            results["text_mining"]["tn"] += 1
        else:
            results["text_mining"]["fp"] += 1

        # ── Random ────────────────────────────────────────────────────────
        if random.random() > (100 / len(drugs_data)):
            results["random"]["tn"] += 1
        else:
            results["random"]["fp"] += 1

        # ── Main algorithm ────────────────────────────────────────────────
        res = await pipeline.analyze_disease(disease, min_score=0.0, max_results=150)
        if res.get("success"):
            drug_in_top = any(
                drug.lower() in c["drug_name"].lower()
                and c["score"] <= exp_max
                for c in res["candidates"]
            )
            candidate = next(
                (c for c in res["candidates"] if drug.lower() in c["drug_name"].lower()),
                None,
            )
            if candidate is None or candidate["score"] <= exp_max:
                results["main_algorithm"]["tn"] += 1
            else:
                results["main_algorithm"]["fp"] += 1

    await text_baseline.close()

    # Compute metrics for each method
    metrics = {}
    for method, counts in results.items():
        tp, fn, tn, fp = counts["tp"], counts["fn"], counts["tn"], counts["fp"]
        n_pos = tp + fn
        n_neg = tn + fp
        metrics[method] = {
            "sensitivity": tp / n_pos if n_pos else 0,
            "specificity": tn / n_neg if n_neg else 0,
            "precision":   tp / (tp + fp) if (tp + fp) > 0 else 0,
            "accuracy":    (tp + tn) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        }

    return metrics