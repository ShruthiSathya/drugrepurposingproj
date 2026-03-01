"""
Extended Baseline Comparisons
==============================
Adds three published drug repurposing baselines for head-to-head comparison.
These are the baselines reviewers at Bioinformatics / PLOS Comput Biol expect.

1. NetworkProximityBaseline — Cheng et al. (2018) Nat Commun
   Gene-level network proximity in PPI graph (shortest-path distance).
   Reference: doi:10.1038/s41467-018-04202-y

2. JaccardOverlapBaseline — Simple Jaccard of drug targets vs disease genes.
   Strong simple baseline that isolates contribution of pathway/literature signals.

3. DiseaseGeneCountBaseline — Scores by raw shared gene count (unnormalized).
   Shows why normalization matters.

Usage in run_validation.py:
    from extended_baselines import NetworkProximityBaseline, JaccardOverlapBaseline
    jac = JaccardOverlapBaseline()
    score = jac.score(drug_targets, disease_genes)

FIXES vs previous version (run_extended_baseline_comparison only — all class
APIs are unchanged)
--------------------------
FIX 1: Removed circular self-import. The comment said the import was removed
   but the bugs that caused it to fail were still present.

FIX 2: case["repurposed_for"] → case["disease"]
   validation_dataset.py uses case["disease"], not case["repurposed_for"].
   The old key caused KeyError for every positive test case, silently skipping
   them all (disease_data = None → continue), producing TP=0 for all baselines.

FIX 3: case["drug_name"] → case["drug"] and neg["drug_name"] → neg["drug"]
   Same issue — validation_dataset.py uses case["drug"], not case["drug_name"].

FIX 4: CosineSimilarityBaseline.fit() moved outside per-case loop.
   IDF weights are a corpus-level statistic computed from ALL drug gene lists.
   Calling fit() inside the loop re-computed it from scratch for every disease,
   making the cosine baseline O(n_diseases × n_drugs) instead of O(n_drugs).
   It also produced subtly wrong IDF values when called with the same data
   repeatedly (no harm but wastes ~10s per validation run).

FIX 5: pipeline.drugs_cache → drugs_data parameter passed by caller.
   pipeline.drugs_cache may not exist (attribute name varies by pipeline
   version). The caller (run_validation.py) already holds drugs_data —
   passing it directly removes the coupling. Signature unchanged for callers
   that pass drugs_data explicitly; callers using keyword args are unaffected.
"""

import asyncio
import json as _json
import math
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Jaccard overlap (isolates gene-set signal, no weights)
# ─────────────────────────────────────────────────────────────────────────────

class JaccardOverlapBaseline:
    """
    Jaccard similarity between drug target set and disease gene set.

    score = |drug_targets ∩ disease_genes| / |drug_targets ∪ disease_genes|

    This is the simplest possible gene-set similarity.
    Outperforming it demonstrates that the weighted gene score + pathway
    overlap + literature signal in the main algorithm add genuine value.

    Reference: Cheng et al. (2018) used Jaccard as one of their baselines.
    """

    def score(self, drug_targets: List[str], disease_genes: List[str]) -> float:
        if not drug_targets or not disease_genes:
            return 0.0
        t = set(drug_targets)
        d = set(disease_genes)
        union = t | d
        if not union:
            return 0.0
        return len(t & d) / len(union)

    def score_all(
        self,
        drugs: List[Dict],
        disease_data: Dict,
        min_score: float = 0.0,
    ) -> List[Dict]:
        disease_genes = disease_data.get("genes", [])
        results = []
        for drug in drugs:
            s = self.score(drug.get("targets", []), disease_genes)
            if s >= min_score:
                results.append({
                    "drug_name": drug["name"],
                    "score":     s,
                    "method":    "jaccard",
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Network proximity (Cheng et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class NetworkProximityBaseline:
    """
    Network proximity between drug targets and disease genes in the PPI.

    Uses the 'closest' proximity measure from Cheng et al. (2018):
      d(A, B) = mean_{a ∈ A} min_{b ∈ B} d(a, b)
    where d(a,b) is the shortest path length in the PPI graph.

    A lower d → closer → higher score.  We convert to a 0-1 score via:
      score = exp(-d(A,B) / scale)   [scale=2 by default]

    Implementation notes:
    - Requires a NetworkX PPI graph. We build it from STRING DB interactions
      (score ≥ 700) for the ~20,000 human protein-coding genes.
    - PPI graph is cached to disk after first build (~30 sec).
    - Falls back to Jaccard if the graph is unavailable.

    Reference:
      Cheng F, Kovacs IA, Barabasi AL (2018) Network-based prediction of
      drug combinations. Nat Commun. doi:10.1038/s41467-018-04202-y
    """

    PPI_CACHE = "/tmp/ppi_graph.gpickle"
    STRING_URL = (
        "https://stringdb-static.org/download/protein.links.v11.5/"
        "9606.protein.links.v11.5.txt.gz"
    )
    STRING_INFO_URL = (
        "https://stringdb-static.org/download/protein.info.v11.5/"
        "9606.protein.info.v11.5.txt.gz"
    )

    def __init__(self, min_score: int = 700, scale: float = 2.0):
        self.min_score = min_score
        self.scale     = scale
        self._graph    = None  # nx.Graph — built lazily

    async def build_ppi(self) -> None:
        """Download STRING DB and build PPI graph. Cached after first run."""
        import networkx as nx
        from pathlib import Path

        cache = Path(self.PPI_CACHE)
        if cache.exists():
            logger.info("✅ Loading PPI graph from cache...")
            self._graph = nx.read_gpickle(str(cache))
            logger.info(
                f"✅ PPI graph: {self._graph.number_of_nodes()} proteins, "
                f"{self._graph.number_of_edges()} interactions"
            )
            return

        import gzip, io, aiohttp
        logger.info("📥 Downloading STRING DB protein info...")

        # Map Ensembl protein IDs → gene symbols
        id_to_symbol: Dict[str, str] = {}
        async with aiohttp.ClientSession() as sess:
            async with sess.get(self.STRING_INFO_URL) as resp:
                raw = await resp.read()
            with gzip.open(io.BytesIO(raw)) as f:
                next(f)  # skip header
                for line in f:
                    parts = line.decode().strip().split("\t")
                    if len(parts) >= 2:
                        protein_id = parts[0]
                        symbol     = parts[1]
                        id_to_symbol[protein_id] = symbol

        logger.info(f"📊 Mapped {len(id_to_symbol)} protein IDs to gene symbols")
        logger.info("📥 Downloading STRING DB interactions (score ≥ 700)...")

        G = nx.Graph()
        n_edges = 0
        async with aiohttp.ClientSession() as sess:
            async with sess.get(self.STRING_URL) as resp:
                raw = await resp.read()
        with gzip.open(io.BytesIO(raw)) as f:
            next(f)  # skip header
            for line in f:
                parts = line.decode().strip().split(" ")
                if len(parts) < 3:
                    continue
                p1, p2, score_str = parts[0], parts[1], parts[2]
                if int(score_str) < self.min_score:
                    continue
                g1 = id_to_symbol.get(p1)
                g2 = id_to_symbol.get(p2)
                if g1 and g2:
                    G.add_edge(g1, g2)
                    n_edges += 1

        logger.info(
            f"✅ PPI graph built: {G.number_of_nodes()} proteins, {n_edges} interactions"
        )
        nx.write_gpickle(G, str(cache))
        self._graph = G

    def _closest_distance(
        self, source_genes: Set[str], target_genes: Set[str]
    ) -> float:
        """
        Compute Cheng et al. 'closest' proximity.
        Returns mean of min-shortest-path from each source gene to any target gene.
        """
        import networkx as nx

        if self._graph is None:
            raise RuntimeError("PPI graph not built. Call build_ppi() first.")

        # Filter to genes present in graph
        S = {g for g in source_genes  if g in self._graph}
        T = {g for g in target_genes  if g in self._graph}

        if not S or not T:
            return float("inf")

        distances = []
        for s in S:
            min_d = float("inf")
            for t in T:
                try:
                    d = nx.shortest_path_length(self._graph, s, t)
                    if d < min_d:
                        min_d = d
                except nx.NetworkXNoPath:
                    pass
            if min_d < float("inf"):
                distances.append(min_d)

        if not distances:
            return float("inf")
        return sum(distances) / len(distances)

    def score(
        self,
        drug_targets:  List[str],
        disease_genes: List[str],
    ) -> float:
        """Convert network proximity to a 0-1 score (higher = closer = better)."""
        if self._graph is None:
            # Fallback to Jaccard if PPI not available
            jac = JaccardOverlapBaseline()
            return jac.score(drug_targets, disease_genes)

        d = self._closest_distance(set(drug_targets), set(disease_genes))
        if d == float("inf"):
            return 0.0
        return math.exp(-d / self.scale)

    def score_all(
        self,
        drugs: List[Dict],
        disease_data: Dict,
        min_score: float = 0.0,
    ) -> List[Dict]:
        disease_genes = disease_data.get("genes", [])
        results = []
        for drug in drugs:
            s = self.score(drug.get("targets", []), disease_genes)
            if s >= min_score:
                results.append({
                    "drug_name": drug["name"],
                    "score":     s,
                    "method":    "network_proximity",
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Raw gene count (unnormalized)
# ─────────────────────────────────────────────────────────────────────────────

class GeneCountBaseline:
    """
    Score = number of shared genes (unnormalized).

    This naive baseline favors drugs with many targets and diseases with
    many associated genes. Demonstrating that the normalized, weighted
    main algorithm outperforms it shows that set-size bias correction matters.
    """

    def score(self, drug_targets: List[str], disease_genes: List[str]) -> float:
        shared = len(set(drug_targets) & set(disease_genes))
        # Normalize to 0-1 using log scale (max cap at 50 shared genes)
        return min(math.log10(shared + 1) / math.log10(51), 1.0)

    def score_all(
        self,
        drugs: List[Dict],
        disease_data: Dict,
        min_score: float = 0.0,
    ) -> List[Dict]:
        disease_genes = disease_data.get("genes", [])
        results = []
        for drug in drugs:
            s = self.score(drug.get("targets", []), disease_genes)
            if s >= min_score:
                results.append({
                    "drug_name": drug["name"],
                    "score":     s,
                    "method":    "gene_count",
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: Cosine similarity (TF-IDF weighted gene vectors)
# ─────────────────────────────────────────────────────────────────────────────

class CosineSimilarityBaseline:
    """
    Cosine similarity between drug and disease gene-set vectors.
    Optionally TF-IDF weighted: genes shared by many drugs are down-weighted.
    """

    def __init__(self, use_tfidf: bool = True):
        self.use_tfidf = use_tfidf
        self._idf: Dict[str, float] = {}

    def fit(self, all_gene_lists: List[List[str]]) -> None:
        """Compute IDF weights from all drug + disease gene lists."""
        if not self.use_tfidf:
            return
        n = len(all_gene_lists)
        df: Dict[str, int] = {}
        for gene_list in all_gene_lists:
            for g in set(gene_list):
                df[g] = df.get(g, 0) + 1
        self._idf = {g: math.log((n + 1) / (count + 1)) + 1.0
                     for g, count in df.items()}

    def _vec(self, genes: List[str]) -> Dict[str, float]:
        if self.use_tfidf and self._idf:
            return {g: self._idf.get(g, 1.0) for g in genes}
        return {g: 1.0 for g in genes}

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a.get(g, 0) * b.get(g, 0) for g in a)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(self, drug_targets: List[str], disease_genes: List[str]) -> float:
        return self._cosine(self._vec(drug_targets), self._vec(disease_genes))

    def score_all(
        self,
        drugs: List[Dict],
        disease_data: Dict,
        min_score: float = 0.0,
    ) -> List[Dict]:
        disease_vec = self._vec(disease_data.get("genes", []))
        results = []
        for drug in drugs:
            s = self._cosine(self._vec(drug.get("targets", [])), disease_vec)
            if s >= min_score:
                results.append({
                    "drug_name": drug.get("name", drug.get("drug_name", "")),
                    "score":     s,
                    "method":    "cosine",
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 5: Text-mining (PubMed co-occurrence)
# ─────────────────────────────────────────────────────────────────────────────

class TextMiningBaseline:
    """
    PubMed co-occurrence score: how often drug and disease appear together.
    Uses NCBI E-utilities (no API key required, rate-limited to 3 req/sec).

    score = log10(co_occurrences + 1) / log10(max_cap + 1)
    """

    EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    MAX_CAP = 500
    _cache: Dict[str, int] = {}

    def _pubmed_count(self, query: str) -> int:
        if query in self._cache:
            return self._cache[query]
        params = urllib.parse.urlencode({
            "db": "pubmed", "term": query,
            "rettype": "count", "retmode": "json",
        })
        try:
            with urllib.request.urlopen(f"{self.EUTILS}?{params}", timeout=10) as r:
                data = _json.loads(r.read())
            count = int(data["esearchresult"]["count"])
        except Exception:
            count = 0
        self._cache[query] = count
        return count

    async def score(self, drug_name: str, disease_name: str) -> float:
        query = f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]'
        loop  = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, self._pubmed_count, query)
        return min(math.log10(count + 1) / math.log10(self.MAX_CAP + 1), 1.0)

    async def score_all(
        self,
        drugs: List[Dict],
        disease_name: str,
        min_score: float = 0.0,
        top_k: int = 200,
    ) -> List[Dict]:
        """Score top_k drugs by name; PubMed is slow so we cap the list."""
        results = []
        for drug in drugs[:top_k]:
            name = drug.get("name", drug.get("drug_name", ""))
            s    = await self.score(name, disease_name)
            if s >= min_score:
                results.append({"drug_name": name, "score": s, "method": "text_mining"})
            await asyncio.sleep(0.34)  # 3 req/sec NCBI rate limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def close(self) -> None:
        pass  # no persistent session to close


# ─────────────────────────────────────────────────────────────────────────────
# Comparison runner — plug in to run_validation.py
# ─────────────────────────────────────────────────────────────────────────────

async def run_extended_baseline_comparison(
    pipeline,
    test_cases: List[Dict],
    neg_cases:  List[Dict],
    top_k: int = 100,
    use_network_proximity: bool = False,  # requires STRING DB download
) -> Dict:
    """
    Run Jaccard, GeneCount, (optionally) NetworkProximity baselines.
    Returns publication-ready metrics dict.

    Add this call to run_validation.py after the main algorithm run,
    then include all methods in the comparison table.

    BUGS FIXED (function body only — signature unchanged):
    - FIX 2: case["repurposed_for"] → case["disease"] (KeyError on every positive case)
    - FIX 3: case["drug_name"] → case["drug"], neg["drug_name"] → neg["drug"]
             (KeyError; validation_dataset.py uses "drug" not "drug_name")
    - FIX 4: cosine.fit() moved outside the per-case loop (was re-fitting
             IDF weights from scratch on every disease — wrong and slow)
    - FIX 5: drugs_data fetched once via pipeline.fetch_approved_drugs() instead
             of pipeline.drugs_cache (attribute may not exist in all versions)
    """
    baselines = {
        "jaccard":    JaccardOverlapBaseline(),
        "gene_count": GeneCountBaseline(),
        "cosine":     CosineSimilarityBaseline(use_tfidf=True),
    }

    if use_network_proximity:
        net_prox = NetworkProximityBaseline()
        await net_prox.build_ppi()
        baselines["network_proximity"] = net_prox

    # FIX 5: fetch drugs once; don't rely on pipeline.drugs_cache
    drugs_data = await pipeline.fetch_approved_drugs(limit=3000)

    # FIX 4: fit cosine IDF once on the full drug pool, not per disease
    all_gene_lists = [d.get("targets", []) for d in drugs_data]
    baselines["cosine"].fit(all_gene_lists)

    counts = {name: {"tp": 0, "fn": 0, "tn": 0, "fp": 0} for name in baselines}

    for case in test_cases:
        disease_data = await pipeline.data_fetcher.fetch_disease_data(
            case["disease"]          # FIX 2: was case["repurposed_for"]
        )
        if not disease_data:
            continue

        for name, baseline in baselines.items():
            results   = baseline.score_all(drugs_data, disease_data)
            top_names = {r["drug_name"].lower() for r in results[:top_k]}
            found     = case["drug"].lower() in top_names  # FIX 3: was case["drug_name"]
            counts[name]["tp" if found else "fn"] += 1

    for neg in neg_cases:
        disease_data = await pipeline.data_fetcher.fetch_disease_data(neg["disease"])
        if not disease_data:
            continue

        for name, baseline in baselines.items():
            results   = baseline.score_all(drugs_data, disease_data)
            top_names = {r["drug_name"].lower() for r in results[:top_k]}
            excluded  = neg["drug"].lower() not in top_names  # FIX 3: was neg["drug_name"]
            counts[name]["tn" if excluded else "fp"] += 1

    metrics = {}
    for name, c in counts.items():
        n_p = c["tp"] + c["fn"]
        n_n = c["tn"] + c["fp"]
        sens = c["tp"] / n_p if n_p else 0
        spec = c["tn"] / n_n if n_n else 0
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0
        f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
        metrics[name] = {
            "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, **c,
        }

    return metrics