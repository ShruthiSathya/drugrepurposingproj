"""
tissue_expression.py — Tissue-Specific Target Expression Filter
================================================================
Scores drug targets by their expression levels in the relevant cancer tissue
using the Human Protein Atlas (HPA) and GTEx APIs. Targets highly expressed
in the relevant tissue but low in critical tissues get boosted; targets
absent in the disease tissue get penalized.

WHY THIS MATTERS
----------------
A drug targeting KRAS is relevant to pancreatic cancer only if the target
(or its downstream effectors) is actually expressed in pancreatic tissue.
A drug that hits a target expressed only in neurons would be:
  1. Ineffective in pancreatic cancer
  2. Potentially neurotoxic

Expression-based scoring:
  - Highly expressed in tumor tissue → score boost (up to +0.3)
  - Not expressed in tumor tissue → score penalty (-0.2)
  - Highly expressed in critical organs (heart, liver, kidney, brain) → safety flag

APIs used
---------
  Human Protein Atlas (HPA)
    https://www.proteinatlas.org/api/search_download.php?...
    JSON per gene: https://www.proteinatlas.org/{gene}-{ensembl_id}/tissue.json
  
  GTEx v8 API (alternative/supplementary)
    https://gtexportal.org/api/v2/expression/geneExpression

Usage
-----
    from tissue_expression import TissueExpressionFilter

    tef = TissueExpressionFilter(cancer_type="pancreatic")
    scored_candidates = await tef.score_candidates(candidates)
    # Each candidate gets tissue_expression_score and tissue_flags

Integration with production pipeline
-------------------------------------
After scoring in ProductionPipeline.analyze_disease(), insert:
    from tissue_expression import TissueExpressionFilter
    tef = TissueExpressionFilter(cancer_type=disease_name)
    candidates = await tef.score_candidates(candidates)
"""

import asyncio
import aiohttp
import json
import logging
import ssl
import certifi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HPA_API_BASE   = "https://www.proteinatlas.org"
GTEX_API_BASE  = "https://gtexportal.org/api/v2"
CACHE_DIR      = Path("/tmp/drug_repurposing_cache")
HPA_CACHE_FILE = CACHE_DIR / "hpa_expression_cache.json"

# ── Cancer tissue → HPA tissue names mapping ────────────────────────────────
# Maps cancer type strings to HPA tissue ontology labels.
# Allows fuzzy matching from disease names like "pancreatic ductal adenocarcinoma"
CANCER_TISSUE_MAP: Dict[str, List[str]] = {
    "pancreatic": ["pancreas", "pancreatic cancer"],
    "glioblastoma": ["cerebral cortex", "hippocampus", "glioblastoma"],
    "glioma": ["cerebral cortex", "hippocampus", "glioma"],
    "lung": ["lung", "lung cancer"],
    "breast": ["breast", "breast cancer"],
    "colorectal": ["colon", "rectum", "colorectal cancer"],
    "ovarian": ["ovary", "ovarian cancer"],
    "prostate": ["prostate", "prostate cancer"],
    "liver": ["liver", "liver cancer", "hepatocellular carcinoma"],
    "melanoma": ["skin", "melanoma"],
    "leukemia": ["bone marrow", "blood"],
    "lymphoma": ["lymph node", "spleen"],
}

# Critical safety tissues — high expression here → adverse event risk
SAFETY_TISSUES = {
    "heart muscle": "cardiac toxicity",
    "liver": "hepatotoxicity",
    "kidney": "nephrotoxicity",
    "cerebral cortex": "CNS toxicity",
    "bone marrow": "myelosuppression",
    "testis": "reproductive toxicity",
}

# HPA expression level → numeric score
HPA_LEVEL_SCORES = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.2,
    "not detected": 0.0,
}


class TissueExpressionFilter:
    """
    Scores drug candidates by tissue-specific target expression.

    Parameters
    ----------
    cancer_type : str
        Cancer type string (e.g., "pancreatic", "glioblastoma").
        Used to look up relevant HPA tissue names.
    session : aiohttp.ClientSession, optional
        Reuse existing session for efficiency.
    """

    def __init__(
        self,
        cancer_type: str,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.cancer_type       = cancer_type.lower()
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssl_context      = self._create_ssl_context()
        self._disk_cache: Dict = self._load_disk_cache()
        self._target_tissues   = self._resolve_target_tissues()

    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _load_disk_cache(self) -> Dict:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if HPA_CACHE_FILE.exists():
            try:
                with open(HPA_CACHE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_disk_cache(self) -> None:
        try:
            with open(HPA_CACHE_FILE, "w") as f:
                json.dump(self._disk_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"HPA cache save failed: {e}")

    def _resolve_target_tissues(self) -> List[str]:
        """Resolve cancer type to HPA tissue label list."""
        for key, tissues in CANCER_TISSUE_MAP.items():
            if key in self.cancer_type:
                logger.info(f"Tissue filter: '{self.cancer_type}' → {tissues}")
                return tissues
        # Fallback: use cancer_type directly as tissue name
        logger.warning(
            f"No tissue map for '{self.cancer_type}' — using raw name as tissue label"
        )
        return [self.cancer_type]

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._external_session and not self._external_session.closed:
            return self._external_session
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    # ── HPA API ──────────────────────────────────────────────────────────────

    async def _fetch_hpa_expression(self, gene_symbol: str) -> Dict[str, float]:
        """
        Fetch tissue expression data from Human Protein Atlas for a gene.

        Returns dict mapping tissue name → expression score (0.0–1.0).
        """
        cache_key = f"hpa_{gene_symbol}"
        if cache_key in self._disk_cache:
            return self._disk_cache[cache_key]

        session = await self._get_session()
        expression: Dict[str, float] = {}

        # HPA JSON search endpoint
        url = f"{HPA_API_BASE}/search/{gene_symbol}.json"
        params = {
            "format": "json",
            "columns": "g,t,rnatsm",  # gene, tissue, RNA tissue specificity
        }

        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    expression = self._parse_hpa_response(data, gene_symbol)
        except Exception as e:
            logger.debug(f"HPA fetch failed for {gene_symbol}: {e}")
            # Try alternative HPA API format
            expression = await self._fetch_hpa_rna_tissue(gene_symbol)

        self._disk_cache[cache_key] = expression
        return expression

    async def _fetch_hpa_rna_tissue(self, gene_symbol: str) -> Dict[str, float]:
        """
        Alternative HPA RNA tissue consensus dataset fetch.
        Uses the proteinatlas.org search API with RNA tissue specificity data.
        """
        session = await self._get_session()
        expression: Dict[str, float] = {}

        url = f"{HPA_API_BASE}/search_download.php"
        params = {
            "search": gene_symbol,
            "format": "json",
            "columns": "g,t,rnatsm",
            "compress": "no",
        }

        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    # Parse tsv-like response
                    lines = text.strip().split("\n")
                    if len(lines) > 1:
                        # header: Gene name, Tissue, RNA consensus tissue gene data
                        for line in lines[1:]:
                            parts = line.split("\t")
                            if len(parts) >= 3 and parts[0].upper() == gene_symbol.upper():
                                tissue = parts[1].lower()
                                try:
                                    # nTPM value
                                    ntpm = float(parts[2])
                                    # Normalize: >100 nTPM = high, 10-100 = medium, 1-10 = low
                                    score = min(ntpm / 100.0, 1.0)
                                    expression[tissue] = score
                                except ValueError:
                                    pass
        except Exception as e:
            logger.debug(f"HPA RNA tissue fetch failed for {gene_symbol}: {e}")

        return expression

    def _parse_hpa_response(self, data: object, gene_symbol: str) -> Dict[str, float]:
        """Parse HPA JSON response into tissue → score dict."""
        expression: Dict[str, float] = {}
        if not isinstance(data, list):
            return expression
        for entry in data:
            if not isinstance(entry, dict):
                continue
            # Check gene name matches
            gene_name = entry.get("Gene", "") or entry.get("gene", "")
            if gene_name.upper() != gene_symbol.upper():
                continue
            tissue = (entry.get("Tissue", "") or entry.get("tissue", "")).lower()
            level  = (entry.get("Level", "") or entry.get("level", "")).lower()
            if tissue:
                expression[tissue] = HPA_LEVEL_SCORES.get(level, 0.0)
        return expression

    # ── Scoring logic ─────────────────────────────────────────────────────────

    def _calculate_tissue_score(
        self, expression: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Calculate tissue expression score for a candidate target.

        Returns
        -------
        (score_delta, safety_flags)
            score_delta : float
                Positive = expressed in target tissue (good)
                Negative = not expressed in target tissue (bad)
            safety_flags : list of str
                Tissue safety concerns based on off-target expression.
        """
        if not expression:
            # No data — neutral, no penalty/bonus
            return 0.0, []

        # Check expression in disease-relevant tissues
        target_expression = 0.0
        for tissue in self._target_tissues:
            tissue_lower = tissue.lower()
            for exp_tissue, exp_score in expression.items():
                if tissue_lower in exp_tissue or exp_tissue in tissue_lower:
                    target_expression = max(target_expression, exp_score)

        # Convert to score delta: -0.2 to +0.3
        if target_expression >= 0.7:
            score_delta = 0.3   # high expression in target tissue
        elif target_expression >= 0.3:
            score_delta = 0.15  # moderate expression
        elif target_expression >= 0.05:
            score_delta = 0.0   # low expression, neutral
        else:
            score_delta = -0.2  # not expressed in target tissue

        # Check safety tissues
        safety_flags: List[str] = []
        for safety_tissue, concern in SAFETY_TISSUES.items():
            for exp_tissue, exp_score in expression.items():
                if safety_tissue in exp_tissue and exp_score >= 0.7:
                    safety_flags.append(f"{concern} risk ({safety_tissue}: high)")

        return score_delta, safety_flags

    # ── Main public method ────────────────────────────────────────────────────

    async def score_candidates(
        self, candidates: List[Dict]
    ) -> List[Dict]:
        """
        Score a list of drug candidates by tissue expression of their targets.

        Parameters
        ----------
        candidates : list of dict
            List of drug candidates from the pipeline.
            Each dict should have a 'target_genes' or 'primary_target' field.

        Returns
        -------
        list of dict
            Same candidates with added 'tissue_expression_score' and
            'tissue_expression_flags' fields.
        """
        # Collect all unique target genes
        all_targets: List[str] = []
        for c in candidates:
            targets = c.get("target_genes", [])
            if isinstance(targets, str):
                targets = [targets]
            if not targets and c.get("primary_target"):
                targets = [c["primary_target"]]
            all_targets.extend(targets)

        unique_targets = list(set(all_targets))
        logger.info(
            f"🔬 Tissue expression scoring: {len(unique_targets)} unique targets "
            f"for {self.cancer_type}"
        )

        # Fetch expression data for all targets concurrently
        # Batch to avoid overwhelming HPA API
        batch_size = 10
        expression_map: Dict[str, Dict[str, float]] = {}
        for i in range(0, len(unique_targets), batch_size):
            batch = unique_targets[i:i + batch_size]
            tasks = [self._fetch_hpa_expression(gene) for gene in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for gene, result in zip(batch, results):
                if isinstance(result, Exception):
                    expression_map[gene] = {}
                else:
                    expression_map[gene] = result
            if i + batch_size < len(unique_targets):
                await asyncio.sleep(0.5)  # Rate limiting

        # Score each candidate
        scored = []
        for candidate in candidates:
            targets = candidate.get("target_genes", [])
            if isinstance(targets, str):
                targets = [targets]
            if not targets and candidate.get("primary_target"):
                targets = [candidate["primary_target"]]

            if not targets:
                scored.append({
                    **candidate,
                    "tissue_expression_score": 0.0,
                    "tissue_expression_flags": [],
                    "tissue_expression_data": {},
                })
                continue

            # Aggregate across all targets for this drug
            total_delta = 0.0
            all_flags: List[str] = []
            target_details: Dict[str, Dict] = {}

            for gene in targets:
                expr = expression_map.get(gene, {})
                delta, flags = self._calculate_tissue_score(expr)
                total_delta += delta
                all_flags.extend(flags)
                target_details[gene] = {
                    "score_delta": delta,
                    "flags": flags,
                    "tissues_detected": [
                        t for t, s in expr.items() if s > 0
                    ][:5],
                }

            # Average delta across targets, clamp to [-0.3, 0.3]
            avg_delta = max(-0.3, min(0.3, total_delta / max(len(targets), 1)))
            unique_flags = list(dict.fromkeys(all_flags))  # dedup preserving order

            scored.append({
                **candidate,
                "tissue_expression_score": round(avg_delta, 4),
                "tissue_expression_flags": unique_flags,
                "tissue_expression_data": target_details,
            })

        # Sort by existing score + tissue delta
        scored.sort(
            key=lambda c: (
                c.get("composite_score", 0) + c.get("tissue_expression_score", 0)
            ),
            reverse=True,
        )

        n_boosted  = sum(1 for c in scored if c["tissue_expression_score"] > 0.1)
        n_penalized = sum(1 for c in scored if c["tissue_expression_score"] < -0.1)
        n_flagged  = sum(1 for c in scored if c["tissue_expression_flags"])
        logger.info(
            f"   ✅ Tissue scoring complete: {n_boosted} boosted, "
            f"{n_penalized} penalized, {n_flagged} safety flagged"
        )

        self._save_disk_cache()
        return scored

    async def close(self) -> None:
        self._save_disk_cache()
        if self._session and not self._session.closed:
            await self._session.close()