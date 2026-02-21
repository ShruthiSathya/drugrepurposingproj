"""
ppi_network.py — Protein-Protein Interaction Network Proximity Scoring
=======================================================================
Fetches PPI data from the STRING database and computes network proximity
between drug targets and disease genes.

WHY THIS MATTERS
----------------
Simple gene overlap scoring misses indirect associations: if drug target A
doesn't directly overlap with disease gene B, but A interacts with B's
binding partner, that's a biologically meaningful connection that pure
overlap scoring ignores entirely.

Network proximity (Cheng et al. 2018) addresses this by measuring the
shortest path distance between drug targets and disease genes in the
full PPI network. A drug whose targets are 1-2 hops away from disease
genes is a strong repurposing candidate even with zero direct gene overlap.

Method: "Closest" network proximity (d_c)
    d_c = mean over all disease genes g of: min over all drug targets t of: d(t, g)
    where d(t, g) is the shortest path length in the STRING PPI graph.

    Normalized proximity score: 1 / (1 + d_c)
    - Score = 1.0: drug targets ARE disease genes (perfect overlap)
    - Score = 0.5: average shortest path = 1 hop (direct interactors)
    - Score = 0.33: average shortest path = 2 hops
    - Score = 0.0: no path exists

STRING API
----------
Uses STRING v12 public API (no authentication required).
Endpoint: https://string-db.org/api/json/network
Filters to interactions with combined_score >= 400 (medium confidence).

Reference
---------
Cheng F, Kovács IA, Barabási AL (2018). Network-based prediction of drug
combinations. Nature Communications 9:3410.
doi:10.1038/s41467-018-05681-7

Usage
-----
    from backend.pipeline.ppi_network import PPINetworkScorer

    scorer = PPINetworkScorer()
    await scorer.initialize()  # fetches STRING data once

    score, evidence = await scorer.compute_proximity(
        drug_targets=['PDE5A', 'NOS3'],
        disease_genes=['BMPR2', 'EDNRA', 'PDE5A'],
    )
    # score: float in [0, 1]
    # evidence: dict with path details
"""

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import certifi
import ssl

logger = logging.getLogger(__name__)

# STRING API settings
STRING_API_BASE    = "https://string-db.org/api/json"
STRING_SPECIES     = 9606          # Human (Homo sapiens)
STRING_MIN_SCORE   = 400           # Medium confidence (0–1000)
STRING_LIMIT       = 500           # Max interactors per protein

# Cache location
CACHE_DIR          = Path("/tmp/drug_repurposing_cache")
PPI_CACHE_FILE     = CACHE_DIR / "string_ppi_cache.json"
PPI_CACHE_TTL_DAYS = 7            # Refresh weekly


class PPINetworkScorer:
    """
    Computes network proximity scores between drug targets and disease genes
    using the STRING protein-protein interaction database.

    The scorer fetches PPI data lazily (per protein set) and caches results
    locally to avoid redundant API calls across validation runs.

    Attributes
    ----------
    graph : dict
        Adjacency list: {protein: {neighbor: score, ...}, ...}
        Built from STRING interactions with combined_score >= STRING_MIN_SCORE.
    _cache : dict
        Disk-backed cache of STRING API responses keyed by protein name.
    """

    def __init__(self, min_score: int = STRING_MIN_SCORE):
        self.min_score = min_score
        self.graph: Dict[str, Dict[str, float]] = {}
        self._cache: Dict[str, List] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssl_context = self._create_ssl_context()
        self._load_cache()

    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            ctx = ssl.create_default_context(cafile=certifi.where())
            return ctx
        except Exception:
            return ssl.create_default_context()

    def _load_cache(self):
        """Load PPI cache from disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if PPI_CACHE_FILE.exists():
            try:
                data = json.loads(PPI_CACHE_FILE.read_text())
                # Check TTL
                cached_at = data.get("_cached_at", 0)
                age_days  = (time.time() - cached_at) / 86400
                if age_days < PPI_CACHE_TTL_DAYS:
                    self._cache = data.get("interactions", {})
                    logger.info(f"✅ Loaded STRING PPI cache ({len(self._cache)} proteins, "
                                f"{age_days:.1f} days old)")
                else:
                    logger.info(f"PPI cache expired ({age_days:.1f} days old) — will refresh")
                    self._cache = {}
            except Exception as e:
                logger.warning(f"Could not load PPI cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Persist PPI cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = {"_cached_at": time.time(), "interactions": self._cache}
            PPI_CACHE_FILE.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Could not save PPI cache: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _fetch_string_interactions(self, proteins: List[str]) -> List[Dict]:
        """
        Fetch STRING interactions for a list of proteins.
        Returns list of interaction dicts with keys:
            preferredName_A, preferredName_B, score (combined_score / 1000)
        """
        # Filter to uncached proteins
        uncached = [p for p in proteins if p not in self._cache]
        if not uncached:
            # Assemble from cache
            interactions = []
            for p in proteins:
                interactions.extend(self._cache.get(p, []))
            return interactions

        session = await self._get_session()

        # STRING accepts up to 500 proteins per request
        batch_size = 100
        all_interactions = []

        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            try:
                url    = f"{STRING_API_BASE}/network"
                params = {
                    "identifiers":    "%0d".join(batch),
                    "species":        STRING_SPECIES,
                    "required_score": self.min_score,
                    "limit":          STRING_LIMIT,
                    "caller_identity": "drug_repurposing_research",
                }
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"STRING API returned {resp.status} for batch {i}")
                        continue
                    data = await resp.json(content_type=None)
                    if not isinstance(data, list):
                        logger.warning(f"Unexpected STRING response type: {type(data)}")
                        continue

                    batch_interactions = [
                        {
                            "protein_a": item.get("preferredName_A", ""),
                            "protein_b": item.get("preferredName_B", ""),
                            "score":     item.get("score", 0),  # already 0-1
                        }
                        for item in data
                        if item.get("score", 0) >= self.min_score / 1000.0
                    ]
                    all_interactions.extend(batch_interactions)

                    # Cache per protein
                    for protein in batch:
                        self._cache[protein] = [
                            ix for ix in batch_interactions
                            if ix["protein_a"] == protein or ix["protein_b"] == protein
                        ]

                    logger.debug(f"STRING: fetched {len(batch_interactions)} "
                                 f"interactions for {len(batch)} proteins")
                    await asyncio.sleep(0.2)  # be polite to STRING API

            except Exception as e:
                logger.warning(f"STRING fetch failed for batch {i}: {e}")
                # Cache empty to avoid retrying failed proteins this session
                for protein in batch:
                    if protein not in self._cache:
                        self._cache[protein] = []

        # Also include cached interactions for proteins that were already cached
        for p in proteins:
            if p not in uncached:
                all_interactions.extend(self._cache.get(p, []))

        self._save_cache()
        return all_interactions

    def _build_subgraph(self, interactions: List[Dict]):
        """Build adjacency list from interaction list."""
        for ix in interactions:
            a = ix["protein_a"]
            b = ix["protein_b"]
            s = float(ix["score"])
            if a and b and a != b:
                if a not in self.graph:
                    self.graph[a] = {}
                if b not in self.graph:
                    self.graph[b] = {}
                # Keep highest score if multiple interactions
                self.graph[a][b] = max(self.graph[a].get(b, 0), s)
                self.graph[b][a] = max(self.graph[b].get(a, 0), s)

    def _bfs_shortest_paths(
        self,
        sources: Set[str],
        targets: Set[str],
        max_depth: int = 4,
    ) -> Dict[str, int]:
        """
        BFS from all source nodes simultaneously.
        Returns {target: shortest_path_length} for reachable targets.
        Stops at max_depth to keep runtime bounded.
        """
        distances: Dict[str, int] = {}
        visited   = set(sources)
        queue     = [(s, 0) for s in sources if s in self.graph]

        while queue:
            next_queue = []
            for node, depth in queue:
                if depth >= max_depth:
                    continue
                for neighbor in self.graph.get(node, {}):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_depth = depth + 1
                        if neighbor in targets:
                            distances[neighbor] = new_depth
                        next_queue.append((neighbor, new_depth))
            queue = next_queue

        return distances

    async def compute_proximity(
        self,
        drug_targets:   List[str],
        disease_genes:  List[str],
        max_depth:      int = 4,
    ) -> Tuple[float, Dict]:
        """
        Compute network proximity score between drug targets and disease genes.

        Parameters
        ----------
        drug_targets : list of str
            HGNC gene symbols of drug's protein targets.
        disease_genes : list of str
            HGNC gene symbols associated with the disease.
        max_depth : int
            Maximum BFS depth (default 4). Paths longer than this are
            treated as unreachable (score contribution = 0).

        Returns
        -------
        (score, evidence) : tuple
            score : float in [0, 1]
                1 / (1 + mean_shortest_path)
                Higher = closer in PPI network.
            evidence : dict
                direct_hits: genes in both sets
                nearest_neighbors: drug targets that are 1-hop from disease genes
                mean_shortest_path: float
                n_reachable: int (disease genes reachable from drug targets)
                n_disease_genes: int
                paths_found: dict {disease_gene: shortest_path_length}

        Paper reporting:
            "Network proximity was computed as d_c = mean shortest path
             between drug targets and disease genes in the STRING PPI network
             (v12, combined_score ≥ 400). Proximity score = 1/(1+d_c)."
        """
        evidence = {
            "direct_hits":        [],
            "nearest_neighbors":  [],
            "mean_shortest_path": None,
            "n_reachable":        0,
            "n_disease_genes":    len(disease_genes),
            "paths_found":        {},
            "ppi_score":          0.0,
            "note":               "",
        }

        if not drug_targets or not disease_genes:
            evidence["note"] = "No drug targets or disease genes provided"
            return 0.0, evidence

        # Direct hits (gene overlap) — counted with path length 0
        direct_hits = set(drug_targets) & set(disease_genes)
        evidence["direct_hits"] = list(direct_hits)

        if not self.graph:
            # Graph not yet built — fetch from STRING
            all_proteins = list(set(drug_targets) | set(disease_genes))
            interactions = await self._fetch_string_interactions(all_proteins)
            self._build_subgraph(interactions)

            if not self.graph:
                # STRING unavailable — fall back to direct overlap only
                if direct_hits:
                    score = len(direct_hits) / len(disease_genes)
                    evidence["note"] = "STRING unavailable — using direct overlap only"
                    evidence["ppi_score"] = round(score, 4)
                    return round(score, 4), evidence
                evidence["note"] = "STRING unavailable and no direct overlap"
                return 0.0, evidence

        # BFS from drug targets to disease genes
        target_set  = set(drug_targets)
        disease_set = set(disease_genes)

        path_lengths: Dict[str, int] = {}

        # Direct hits have path length 0
        for gene in direct_hits:
            path_lengths[gene] = 0

        # BFS for remaining disease genes
        remaining = disease_set - direct_hits
        if remaining:
            bfs_results = self._bfs_shortest_paths(
                sources=target_set,
                targets=remaining,
                max_depth=max_depth,
            )
            path_lengths.update(bfs_results)

        # Nearest neighbors (path length = 1)
        nearest = [g for g, d in path_lengths.items() if d == 1]
        evidence["nearest_neighbors"] = nearest
        evidence["paths_found"]       = path_lengths
        evidence["n_reachable"]       = len(path_lengths)

        if not path_lengths:
            evidence["note"] = "No paths found within max_depth"
            return 0.0, evidence

        # Mean shortest path over ALL disease genes (unreachable = max_depth + 1)
        total_distance = 0.0
        for gene in disease_genes:
            if gene in path_lengths:
                total_distance += path_lengths[gene]
            else:
                total_distance += (max_depth + 1)  # penalty for unreachable

        mean_path = total_distance / len(disease_genes)
        evidence["mean_shortest_path"] = round(mean_path, 3)

        # Proximity score: 1 / (1 + mean_path)
        score = 1.0 / (1.0 + mean_path)
        score = round(min(score, 1.0), 4)
        evidence["ppi_score"] = score

        if direct_hits:
            evidence["note"] = (
                f"{len(direct_hits)} direct target-gene overlaps + "
                f"{len(nearest)} 1-hop neighbors"
            )
        else:
            evidence["note"] = (
                f"No direct overlap; {len(nearest)} 1-hop neighbors found"
            )

        logger.debug(
            f"PPI proximity: mean_path={mean_path:.2f} → score={score:.3f} "
            f"(direct={len(direct_hits)}, 1-hop={len(nearest)}, "
            f"reachable={len(path_lengths)}/{len(disease_genes)})"
        )

        return score, evidence

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for use inside generate_candidates()
# ─────────────────────────────────────────────────────────────────────────────

async def batch_ppi_scores(
    drugs_data:    List[Dict],
    disease_genes: List[str],
    scorer:        PPINetworkScorer,
) -> Dict[str, Tuple[float, Dict]]:
    """
    Compute PPI proximity scores for all drugs in parallel.

    Parameters
    ----------
    drugs_data : list of dict
        Each dict must have 'name' and 'targets' keys.
    disease_genes : list of str
        Disease-associated gene symbols.
    scorer : PPINetworkScorer
        Initialized scorer instance.

    Returns
    -------
    dict : {drug_name: (ppi_score, evidence)}
    """
    # Pre-fetch all unique proteins in one STRING batch call
    all_proteins = set(disease_genes)
    for drug in drugs_data:
        all_proteins.update(drug.get("targets", []))

    interactions = await scorer._fetch_string_interactions(list(all_proteins))
    scorer._build_subgraph(interactions)
    logger.info(f"STRING PPI graph: {len(scorer.graph)} proteins, "
                f"{sum(len(v) for v in scorer.graph.values())//2} edges")

    # Score each drug (graph already built — no more API calls)
    results = {}
    for drug in drugs_data:
        targets = drug.get("targets", [])
        if targets:
            score, evidence = await scorer.compute_proximity(
                drug_targets=targets,
                disease_genes=disease_genes,
            )
        else:
            score   = 0.0
            evidence = {"ppi_score": 0.0, "note": "No drug targets"}
        results[drug["name"]] = (score, evidence)

    return results