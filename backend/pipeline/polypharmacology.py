"""
polypharmacology.py — Multi-Target Drug Combination Scoring
============================================================
Analyzes top drug candidates for polypharmacological potential — the ability
of a drug (or drug combination) to hit multiple disease-relevant targets
simultaneously. This is critical for cancers like PDAC where single-target
approaches have universally failed.

WHY THIS MATTERS FOR CANCER
----------------------------
Pancreatic cancer (PDAC) has failed >50 Phase 2/3 trials largely because:
  1. KRAS G12D/G12V mutations bypass single-target inhibition via feedback loops
  2. Dense stroma creates drug delivery barriers requiring combination approaches
  3. Multiple resistance mechanisms emerge simultaneously

Polypharmacology addresses this by:
  - Identifying drugs that hit 2+ PDAC-relevant targets (e.g., EGFR + SRC)
  - Finding drug pairs with synergistic target coverage (e.g., MEK + PI3K)
  - Penalizing drugs with highly overlapping target profiles (redundant combinations)

Scoring components
------------------
  1. Multi-target score       : does a single drug hit multiple disease targets?
  2. Pathway coverage score   : do targets span multiple cancer hallmark pathways?
  3. Combination synergy score: for drug pairs, do they cover complementary targets?
  4. Resistance bypass score  : do targets include known resistance mechanisms?

Usage
-----
    from polypharmacology import PolypharmacologyScorer

    scorer = PolypharmacologyScorer(disease_genes=disease_data["genes"])
    scored = await scorer.score_candidates(top_50_candidates)
    pairs  = await scorer.find_synergistic_pairs(top_20_candidates)

Integration
-----------
In production pipeline after scoring:
    from polypharmacology import PolypharmacologyScorer
    poly_scorer = PolypharmacologyScorer(disease_genes=disease_data["genes"])
    candidates = await poly_scorer.score_candidates(candidates)
    top_pairs  = await poly_scorer.find_synergistic_pairs(candidates[:20])
"""

import asyncio
import aiohttp
import json
import logging
import ssl
import certifi
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR       = Path("/tmp/drug_repurposing_cache")
POLY_CACHE_FILE = CACHE_DIR / "polypharmacology_cache.json"

# Cancer hallmark pathways and their key genes
# Based on Hanahan & Weinberg (2011) Hallmarks of Cancer
CANCER_HALLMARK_PATHWAYS: Dict[str, List[str]] = {
    "proliferation": [
        "KRAS", "BRAF", "RAF1", "MEK1", "MEK2", "MAP2K1", "MAP2K2",
        "ERK1", "ERK2", "MAPK1", "MAPK3", "EGFR", "ERBB2", "MYC",
        "CCND1", "CDK4", "CDK6", "RB1",
    ],
    "apoptosis_evasion": [
        "BCL2", "BCL2L1", "BCL2L2", "MCL1", "BAX", "BAK1",
        "TP53", "MDM2", "MDM4", "PUMA", "BIM",
    ],
    "invasion_metastasis": [
        "MMP2", "MMP9", "SNAI1", "SNAI2", "VIM", "CDH1",
        "TWIST1", "ZEB1", "ZEB2", "ITGB1", "FAK", "PTK2",
    ],
    "angiogenesis": [
        "VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "PDGFRA",
        "PDGFRB", "FGF2", "FGFR1",
    ],
    "immune_evasion": [
        "CD274", "PDCD1LG2", "CTLA4", "PDCD1", "IDO1", "FOXP3",
        "CD8A", "TIGIT", "LAG3",
    ],
    "dna_repair": [
        "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2",
        "PARP1", "RAD51", "PALB2",
    ],
    "cell_cycle": [
        "TP53", "RB1", "CDKN2A", "CDKN1A", "CDK4", "CDK6",
        "CDK2", "CCNE1", "CCNB1",
    ],
    "pi3k_akt_mtor": [
        "PIK3CA", "PIK3CB", "PTEN", "AKT1", "AKT2", "MTOR",
        "TSC1", "TSC2", "S6K1", "RPS6KB1",
    ],
    "wnt_signaling": [
        "CTNNB1", "APC", "AXIN1", "GSK3B", "DVL1", "LRP5", "LRP6",
    ],
    "tgf_smad": [
        "TGFB1", "TGFB2", "SMAD2", "SMAD3", "SMAD4", "ACVR1B",
    ],
    "stroma_ecm": [
        "ACTA2", "COL1A1", "FN1", "HAS2", "FAP", "PDPN",
        "POSTN", "TNC",
    ],
}

# Known resistance bypass genes for PDAC specifically
PDAC_RESISTANCE_GENES = {
    "KRAS", "KRAS2", "KRASG12D", "KRASG12V",  # KRAS variants
    "YAP1", "WWTR1",     # Hippo bypass
    "MCL1",              # Apoptosis bypass
    "ABCB1", "ABCG2",    # Drug efflux
    "FGF1", "FGF2",      # RTK bypass signals
    "IL6", "STAT3",      # JAK-STAT activation
    "NF2", "LATS1",      # Hippo pathway
}


class PolypharmacologyScorer:
    """
    Scores drug candidates for polypharmacological potential.

    Parameters
    ----------
    disease_genes : list of str
        Gene symbols associated with the disease (from disease_data["genes"]).
    gene_scores : dict, optional
        Mapping of gene → association score from OpenTargets/EFO expansion.
    resistance_genes : set, optional
        Genes linked to drug resistance in this cancer. Defaults to PDAC set.
    session : aiohttp.ClientSession, optional
        Reuse existing session.
    """

    def __init__(
        self,
        disease_genes: List[str],
        gene_scores: Optional[Dict[str, float]] = None,
        resistance_genes: Optional[Set[str]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.disease_genes    = set(disease_genes)
        self.gene_scores      = gene_scores or {}
        self.resistance_genes = resistance_genes or PDAC_RESISTANCE_GENES
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssl_context     = self._create_ssl_context()
        self._disk_cache: Dict = self._load_disk_cache()

        # Build gene → pathway index for fast lookup
        self._gene_pathway_map: Dict[str, List[str]] = {}
        for pathway, genes in CANCER_HALLMARK_PATHWAYS.items():
            for gene in genes:
                if gene not in self._gene_pathway_map:
                    self._gene_pathway_map[gene] = []
                self._gene_pathway_map[gene].append(pathway)

    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _load_disk_cache(self) -> Dict:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if POLY_CACHE_FILE.exists():
            try:
                with open(POLY_CACHE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_disk_cache(self) -> None:
        try:
            with open(POLY_CACHE_FILE, "w") as f:
                json.dump(self._disk_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Polypharmacology cache save failed: {e}")

    # ── Scoring components ────────────────────────────────────────────────────

    def _score_multi_target(self, target_genes: List[str]) -> float:
        """
        Score based on number of disease-relevant targets hit.
        Returns 0.0 – 1.0.
        """
        if not target_genes:
            return 0.0
        disease_relevant = [g for g in target_genes if g in self.disease_genes]
        n_relevant = len(disease_relevant)

        # Scoring: 1 target = baseline, 2 = good, 3+ = excellent
        if n_relevant == 0:
            return 0.0
        elif n_relevant == 1:
            return 0.3
        elif n_relevant == 2:
            return 0.6
        elif n_relevant == 3:
            return 0.8
        else:
            return min(0.3 + (n_relevant * 0.15), 1.0)

    def _score_pathway_coverage(self, target_genes: List[str]) -> Tuple[float, List[str]]:
        """
        Score based on number of distinct cancer hallmark pathways covered.
        Returns (score, list_of_covered_pathways).
        """
        covered_pathways: Set[str] = set()
        for gene in target_genes:
            for pathway in self._gene_pathway_map.get(gene, []):
                covered_pathways.add(pathway)

        n_pathways = len(covered_pathways)
        if n_pathways == 0:
            score = 0.0
        elif n_pathways == 1:
            score = 0.2
        elif n_pathways == 2:
            score = 0.5
        elif n_pathways == 3:
            score = 0.75
        else:
            score = min(0.5 + (n_pathways * 0.1), 1.0)

        return score, list(covered_pathways)

    def _score_resistance_bypass(self, target_genes: List[str]) -> float:
        """
        Score based on whether targets include known resistance mechanism genes.
        """
        resistance_targets = [
            g for g in target_genes if g in self.resistance_genes
        ]
        if not resistance_targets:
            return 0.0
        return min(len(resistance_targets) * 0.25, 0.75)

    def _calculate_polypharmacology_score(self, candidate: Dict) -> Dict:
        """
        Calculate full polypharmacology score for a single candidate.
        Returns updated candidate dict with polypharmacology fields.
        """
        targets = candidate.get("target_genes", [])
        if isinstance(targets, str):
            targets = [targets]
        if not targets and candidate.get("primary_target"):
            targets = [candidate["primary_target"]]

        # Calculate component scores
        multi_target_score  = self._score_multi_target(targets)
        pathway_score, pathways = self._score_pathway_coverage(targets)
        resistance_score    = self._score_resistance_bypass(targets)

        # Weighted composite polypharmacology score
        poly_score = (
            multi_target_score  * 0.45 +
            pathway_score       * 0.35 +
            resistance_score    * 0.20
        )

        return {
            **candidate,
            "polypharmacology_score": round(poly_score, 4),
            "poly_components": {
                "multi_target":      round(multi_target_score, 4),
                "pathway_coverage":  round(pathway_score, 4),
                "resistance_bypass": round(resistance_score, 4),
                "pathways_covered":  pathways,
                "n_disease_targets": len([g for g in targets if g in self.disease_genes]),
            },
        }

    # ── Drug pair synergy ─────────────────────────────────────────────────────

    def _score_pair_synergy(
        self, drug_a: Dict, drug_b: Dict
    ) -> Tuple[float, Dict]:
        """
        Score synergy potential between two drugs based on complementary target coverage.

        Logic:
          - High synergy: drugs cover non-overlapping pathways (complementary)
          - Medium synergy: drugs share some targets but have unique ones
          - Low synergy: drugs are nearly redundant (overlapping targets/pathways)
        """
        targets_a = set(drug_a.get("target_genes", []))
        targets_b = set(drug_b.get("target_genes", []))

        if not targets_a or not targets_b:
            return 0.0, {"reason": "missing_target_data"}

        # Target overlap
        overlap = targets_a & targets_b
        union   = targets_a | targets_b
        overlap_ratio = len(overlap) / max(len(union), 1)

        # Pathway coverage of combination
        combined_targets = list(union)
        _, combined_pathways  = self._score_pathway_coverage(combined_targets)
        _, pathways_a = self._score_pathway_coverage(list(targets_a))
        _, pathways_b = self._score_pathway_coverage(list(targets_b))
        unique_pathways = set(combined_pathways) - (set(pathways_a) & set(pathways_b))

        # Resistance bypass
        combined_resistance = self._score_resistance_bypass(combined_targets)

        # Synergy is high when:
        #   1. Low target overlap (non-redundant)
        #   2. Combined pathway coverage > individual
        #   3. Together they bypass resistance
        target_complementarity = 1.0 - overlap_ratio
        pathway_additive_value = len(unique_pathways) * 0.15

        synergy_score = (
            target_complementarity  * 0.5 +
            min(pathway_additive_value, 0.4) * 0.35 +
            combined_resistance     * 0.15
        )
        synergy_score = min(synergy_score, 1.0)

        details = {
            "target_overlap":         sorted(overlap),
            "overlap_ratio":          round(overlap_ratio, 3),
            "unique_pathways_gained": sorted(unique_pathways),
            "combined_pathway_count": len(combined_pathways),
            "resistance_bypass":      round(combined_resistance, 4),
            "synergy_rationale": (
                "Highly complementary non-overlapping mechanisms" if synergy_score > 0.7
                else "Moderate complementarity with some overlap" if synergy_score > 0.4
                else "Largely redundant targets"
            ),
        }

        return round(synergy_score, 4), details

    # ── Main public methods ───────────────────────────────────────────────────

    async def score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Score candidates for polypharmacological potential.

        Adds 'polypharmacology_score' and 'poly_components' to each candidate.
        """
        logger.info(f"💊 Polypharmacology scoring: {len(candidates)} candidates")

        scored = [self._calculate_polypharmacology_score(c) for c in candidates]

        # Sort by composite + poly score
        scored.sort(
            key=lambda c: (
                c.get("composite_score", 0) +
                c.get("polypharmacology_score", 0) * 0.3  # 30% weight in final rank
            ),
            reverse=True,
        )

        n_multi   = sum(1 for c in scored if c["poly_components"]["n_disease_targets"] >= 2)
        n_pathway = sum(1 for c in scored if len(c["poly_components"]["pathways_covered"]) >= 2)
        logger.info(
            f"   ✅ Poly scoring: {n_multi} multi-target drugs, "
            f"{n_pathway} multi-pathway drugs"
        )

        self._save_disk_cache()
        return scored

    async def find_synergistic_pairs(
        self,
        candidates: List[Dict],
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Find most synergistic drug pairs from top candidates.

        Parameters
        ----------
        candidates : list of dict
            Top drug candidates (recommend top 20).
        top_n : int
            Number of top pairs to return.

        Returns
        -------
        list of dict
            Top synergistic pairs with scores and rationale.
        """
        logger.info(
            f"🔗 Finding synergistic pairs from {len(candidates)} candidates"
        )

        pairs = []
        for drug_a, drug_b in itertools.combinations(candidates, 2):
            synergy_score, details = self._score_pair_synergy(drug_a, drug_b)
            pairs.append({
                "drug_a":          drug_a.get("drug_name", "Unknown"),
                "drug_b":          drug_b.get("drug_name", "Unknown"),
                "drug_a_id":       drug_a.get("chembl_id", ""),
                "drug_b_id":       drug_b.get("chembl_id", ""),
                "synergy_score":   synergy_score,
                "individual_scores": {
                    "drug_a": drug_a.get("composite_score", 0),
                    "drug_b": drug_b.get("composite_score", 0),
                },
                "combination_details": details,
            })

        # Sort by synergy score
        pairs.sort(key=lambda p: p["synergy_score"], reverse=True)
        top_pairs = pairs[:top_n]

        logger.info(
            f"   ✅ Top pair: {top_pairs[0]['drug_a']} + {top_pairs[0]['drug_b']} "
            f"(synergy: {top_pairs[0]['synergy_score']:.3f})"
            if top_pairs else "   No pairs found"
        )

        return top_pairs

    async def close(self) -> None:
        self._save_disk_cache()
        if self._session and not self._session.closed:
            await self._session.close()