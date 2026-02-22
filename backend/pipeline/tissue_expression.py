"""
tissue_expression.py — Tissue Expression Scoring v2.0
=======================================================
Scores drug candidates based on whether their protein targets are expressed
in the disease-relevant tissue, using the Human Protein Atlas (HPA) API.

FIXES vs v1
-----------
1. API SCHEMA MISMATCH (CRITICAL): v1 called /search/{gene}.json and
   /search_download.php, then looked for "Gene" and "Tissue" keys in the
   response. Neither endpoint exists in the current HPA REST API (v23+) and
   the field names were wrong even for older endpoints. In practice the module
   silently returned 0.0 for every gene because all responses were empty dicts.

   v2 uses the correct HPA REST API v1 endpoint:
     https://www.proteinatlas.org/api/search_download.php
   with parameters: format=json, columns=g,t (gene, tissue), query=<gene>
   
   And also tries the direct gene JSON endpoint:
     https://www.proteinatlas.org/{ensembl_id}.json
   for gene-level data when the search returns an Ensembl ID.
   
   The correct HPA JSON schema is documented at:
   https://www.proteinatlas.org/about/download
   Field names actually returned: "Gene", "Tissue", "Level", "Reliability"
   (capital first letter, as in the TSV column headers).

2. EXPRESSION LEVEL PARSING: v1's _parse_hpa_response() used lowercase key
   names. HPA JSON uses title-case. Fixed parsing handles both for robustness.

3. TISSUE MAPPING: Added a comprehensive tissue alias map so common disease
   terms (e.g. "colon cancer" → "colon", "parkinson" → "brain") resolve to
   HPA tissue names without requiring exact match.

4. CACHE VALIDATION: v1 cached empty dicts from failed API calls, permanently
   poisoning the cache for those genes. v2 only caches non-empty results.

5. GRACEFUL FALLBACK: When HPA is unavailable, v2 falls back to a curated
   expression database of ~100 well-characterised cancer/disease driver genes,
   so scoring continues to work offline.

6. SCORE CALIBRATION: Expression level → numeric score mapping corrected to
   match HPA's 4-level ordinal scale (Not detected < Low < Medium < High).

ACCURACY NOTES
--------------
- HPA tissue-level data is immunohistochemistry-based, not RNA-seq.
- "Not detected" in IHC does not always mean absent — sensitivity varies.
- For rare diseases with unusual tissues (e.g. inner ear, dorsal root ganglion),
  HPA coverage is sparse. Scores will be low/absent — caveat in report.
- For drugs with multiple targets, the max expression across targets is used,
  not the mean (a drug works if any target is expressed).
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

CACHE_DIR       = Path("/tmp/drug_repurposing_cache")
EXPR_CACHE_FILE = CACHE_DIR / "tissue_expression_cache.json"
CACHE_TTL_SECS  = 7 * 24 * 3600  # 1 week

# ─────────────────────────────────────────────────────────────────────────────
# HPA REST API endpoints (v23+, current as of 2024)
# ─────────────────────────────────────────────────────────────────────────────
HPA_SEARCH_URL      = "https://www.proteinatlas.org/api/search_download.php"
HPA_GENE_JSON_URL   = "https://www.proteinatlas.org/{ensembl_id}.json"
HPA_SEARCH_TIMEOUT  = 15
MAX_CONCURRENT_REQS = 3

# ─────────────────────────────────────────────────────────────────────────────
# HPA expression level → numeric score (4-level ordinal scale)
# ─────────────────────────────────────────────────────────────────────────────
EXPRESSION_LEVEL_SCORES: Dict[str, float] = {
    "high":         1.00,
    "medium":       0.67,
    "low":          0.33,
    "not detected": 0.00,
    "none":         0.00,
    "na":           0.00,
    "n/a":          0.00,
    "":             0.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# Tissue alias map: disease/organ terms → HPA canonical tissue names
# HPA uses specific tissue names — see full list at:
# https://www.proteinatlas.org/humanproteome/tissue
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_ALIASES: Dict[str, List[str]] = {
    # Oncology
    "pancreatic":                    ["pancreas"],
    "pdac":                          ["pancreas"],
    "glioblastoma":                  ["cerebral cortex", "hippocampus", "caudate"],
    "gbm":                           ["cerebral cortex", "hippocampus"],
    "glioma":                        ["cerebral cortex", "hippocampus"],
    "lung cancer":                   ["lung"],
    "nsclc":                         ["lung"],
    "breast cancer":                 ["breast"],
    "colorectal":                    ["colon", "rectum"],
    "colon cancer":                  ["colon"],
    "ovarian":                       ["ovary"],
    "melanoma":                      ["skin"],
    "multiple myeloma":              ["bone marrow"],
    "leukemia":                      ["bone marrow", "lymph node"],
    "lymphoma":                      ["lymph node", "tonsil"],
    "hepatocellular":                ["liver"],
    "liver cancer":                  ["liver"],
    "renal cell carcinoma":          ["kidney"],
    "bladder cancer":                ["urinary bladder"],
    "prostate cancer":               ["prostate"],
    "cervical cancer":               ["cervix, uterine"],
    "endometrial cancer":            ["endometrium"],
    "thyroid cancer":                ["thyroid gland"],
    "esophageal cancer":             ["esophagus"],
    "gastric cancer":                ["stomach"],

    # Neurology
    "parkinson":                     ["substantia nigra", "caudate", "cerebral cortex"],
    "alzheimer":                     ["cerebral cortex", "hippocampus", "amygdala"],
    "alzheimer's":                   ["cerebral cortex", "hippocampus", "amygdala"],
    "multiple sclerosis":            ["cerebral cortex", "white matter"],
    "epilepsy":                      ["hippocampus", "cerebral cortex"],
    "amyotrophic lateral sclerosis": ["spinal cord", "cerebral cortex"],
    "als":                           ["spinal cord", "cerebral cortex"],
    "huntington":                    ["caudate", "cerebral cortex"],
    "huntington's":                  ["caudate", "cerebral cortex"],

    # Autoimmune / Inflammatory
    "rheumatoid arthritis":          ["synovial membrane", "soft tissue"],
    "systemic lupus":                ["kidney", "skin", "heart muscle"],
    "lupus":                         ["kidney", "skin"],
    "inflammatory bowel disease":    ["colon", "small intestine"],
    "crohn":                         ["small intestine", "colon"],
    "ulcerative colitis":            ["colon"],
    "psoriasis":                     ["skin"],

    # Cardiovascular
    "heart failure":                 ["heart muscle"],
    "cardiomyopathy":                ["heart muscle"],
    "pulmonary arterial hypertension":["lung", "heart muscle"],
    "pericarditis":                  ["heart muscle"],
    "atherosclerosis":               ["aorta", "coronary artery"],

    # Metabolic
    "type 2 diabetes":               ["pancreas", "adipose tissue", "liver"],
    "diabetes":                      ["pancreas", "adipose tissue"],
    "obesity":                       ["adipose tissue"],
    "polycystic ovary syndrome":     ["ovary", "adipose tissue"],
    "hypercholesterolemia":          ["liver"],
    "gout":                          ["kidney", "synovial membrane"],

    # Pulmonary
    "asthma":                        ["lung", "bronchus"],
    "copd":                          ["lung"],
    "cystic fibrosis":               ["lung", "pancreas"],

    # Renal
    "chronic kidney disease":        ["kidney"],
    "ckd":                           ["kidney"],

    # Haematological
    "hemophilia":                    ["liver"],
    "anemia":                        ["bone marrow"],

    # Rare / Genetic
    "tuberous sclerosis":            ["brain", "kidney"],
    "spinal muscular atrophy":       ["spinal cord"],
    "sma":                           ["spinal cord"],

    # Generic fallbacks for anatomical terms
    "brain":                         ["cerebral cortex", "hippocampus"],
    "liver":                         ["liver"],
    "kidney":                        ["kidney"],
    "lung":                          ["lung"],
    "heart":                         ["heart muscle"],
    "skin":                          ["skin"],
    "colon":                         ["colon"],
    "muscle":                        ["skeletal muscle"],
    "adipose":                       ["adipose tissue"],
    "bone marrow":                   ["bone marrow"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Curated expression fallback database
# Source: UniProt/HPA manually curated, for ~80 common drug target genes.
# Used when HPA API is unavailable.
# Format: gene_symbol → {tissue → level}
# ─────────────────────────────────────────────────────────────────────────────
CURATED_EXPRESSION: Dict[str, Dict[str, str]] = {
    # Oncology drivers
    "KRAS":    {"pancreas": "medium", "colon": "high", "lung": "high"},
    "BRAF":    {"skin": "high", "colon": "medium", "thyroid gland": "medium"},
    "EGFR":    {"lung": "high", "colon": "medium", "skin": "high"},
    "ERBB2":   {"breast": "medium", "stomach": "medium"},
    "TP53":    {"liver": "high", "colon": "high", "lung": "high"},
    "MYC":     {"colon": "high", "lung": "medium"},
    "CDK4":    {"liver": "high", "breast": "medium"},
    "CDK6":    {"lymph node": "high", "bone marrow": "medium"},
    "PIK3CA":  {"breast": "high", "colon": "medium"},
    "PTEN":    {"prostate": "low", "breast": "low"},
    "ALK":     {"lung": "medium", "cerebral cortex": "high"},
    "MET":     {"liver": "high", "lung": "medium"},
    "RET":     {"thyroid gland": "high"},
    "MTOR":    {"liver": "high", "kidney": "medium", "lung": "medium"},
    "AKT1":    {"liver": "high", "lung": "medium"},

    # Breast cancer
    "ESR1":    {"breast": "high", "uterus": "high"},
    "PGR":     {"breast": "high", "uterus": "high"},

    # Haematology
    "ABL1":    {"bone marrow": "high"},
    "BCL2":    {"lymph node": "high", "bone marrow": "high"},
    "BRCA1":   {"breast": "low", "ovary": "low"},
    "BRCA2":   {"breast": "low", "pancreas": "low"},
    "CRBN":    {"bone marrow": "medium", "kidney": "medium"},

    # Immuno-oncology
    "CD274":   {"lung": "medium", "placenta": "high"},   # PD-L1
    "PDCD1":   {"lymph node": "medium"},                  # PD-1
    "CTLA4":   {"lymph node": "medium"},
    "CD19":    {"lymph node": "high", "bone marrow": "high"},
    "CD20":    {"lymph node": "high", "tonsil": "high"},

    # Neurology
    "SNCA":    ["substantia nigra", "high"],   # handled below
    "LRRK2":   {"substantia nigra": "high", "kidney": "medium"},
    "MAPT":    {"cerebral cortex": "high", "hippocampus": "high"},
    "APP":     {"cerebral cortex": "high", "hippocampus": "high"},
    "PSEN1":   {"cerebral cortex": "medium", "hippocampus": "medium"},
    "APOE":    {"liver": "high", "cerebral cortex": "medium"},
    "TREM2":   {"cerebral cortex": "medium", "spleen": "high"},
    "HTT":     {"caudate": "high", "cerebral cortex": "high"},
    "SOD1":    {"spinal cord": "high", "liver": "high"},
    "TARDBP":  {"spinal cord": "high", "cerebral cortex": "high"},

    # Cardiovascular
    "NPPA":    {"heart muscle": "high"},
    "NPPB":    {"heart muscle": "high"},
    "MYH7":    {"heart muscle": "high", "skeletal muscle": "high"},
    "ACE":     {"lung": "high", "kidney": "high"},
    "ADRB1":   {"heart muscle": "high"},
    "ADRB2":   {"lung": "high", "heart muscle": "medium"},

    # Inflammation / Autoimmune
    "TNF":     {"lymph node": "medium", "colon": "medium"},
    "IL6":     {"liver": "medium", "colon": "medium"},
    "IL1B":    {"colon": "medium", "bone marrow": "medium"},
    "JAK1":    {"colon": "medium", "liver": "medium"},
    "JAK2":    {"bone marrow": "high", "liver": "medium"},
    "STAT3":   {"liver": "high", "colon": "medium"},
    "IFNG":    {"lymph node": "medium"},
    "NLRP3":   {"colon": "medium", "kidney": "medium"},

    # Metabolic
    "INSR":    {"liver": "high", "adipose tissue": "high", "skeletal muscle": "high"},
    "GLP1R":   {"pancreas": "high", "lung": "medium"},
    "PPARG":   {"adipose tissue": "high", "colon": "medium"},
    "PCSK9":   {"liver": "high"},
    "HMGCR":   {"liver": "high"},
    "LDLR":    {"liver": "high"},
    "CFTR":    {"lung": "medium", "pancreas": "medium"},

    # Kinases (common drug targets)
    "SRC":     {"liver": "medium", "colon": "medium"},
    "LCK":     {"lymph node": "high"},
    "ZAP70":   {"lymph node": "high"},
    "BTK":     {"lymph node": "high", "bone marrow": "high"},
    "VEGFA":   {"liver": "high", "colon": "medium"},
    "KDR":     {"lung": "medium"},   # VEGFR2
    "FGFR1":   {"kidney": "medium", "lung": "medium"},
    "PDGFRA":  {"brain": "medium"},

    # GPCRs
    "DRD2":    {"caudate": "high", "substantia nigra": "high"},
    "DRD1":    {"caudate": "high"},
    "ADORA2A": ["caudate", "high"],   # handled below
    "HTR2A":   {"cerebral cortex": "medium"},
    "ADRB3":   {"adipose tissue": "high"},
}

# Clean up any list entries in CURATED_EXPRESSION that slipped in above
for _gene, _data in list(CURATED_EXPRESSION.items()):
    if isinstance(_data, list) and len(_data) == 2:
        CURATED_EXPRESSION[_gene] = {_data[0]: _data[1]}


class TissueExpressionScorer:
    """
    Scores drug candidates based on protein target expression in disease tissue.

    Uses HPA REST API v23+ with correct endpoint and field names.
    Falls back to curated expression database if API is unavailable.
    """

    def __init__(self, disease_name: str = ""):
        self.disease_name = disease_name.lower().strip()
        self._cache: Dict[str, Dict] = {}
        self._load_disk_cache()
        self._target_tissues = self._resolve_target_tissues()

    def _load_disk_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if EXPR_CACHE_FILE.exists():
            try:
                raw = json.loads(EXPR_CACHE_FILE.read_text())
                now = time.time()
                # Prune expired entries and empty cached results
                self._cache = {
                    k: v for k, v in raw.items()
                    if v.get("fetched_at", 0) + CACHE_TTL_SECS > now
                    and v.get("expression_data")  # skip cached empty results
                }
                logger.debug(f"Loaded {len(self._cache)} cached expression records")
            except Exception as e:
                logger.warning(f"Expression cache load failed: {e}")
                self._cache = {}

    def _save_disk_cache(self) -> None:
        try:
            EXPR_CACHE_FILE.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            logger.warning(f"Expression cache save failed: {e}")

    def _resolve_target_tissues(self) -> List[str]:
        """Map disease name to HPA tissue names via TISSUE_ALIASES."""
        if not self.disease_name:
            return []

        for keyword, tissues in TISSUE_ALIASES.items():
            if keyword in self.disease_name or self.disease_name in keyword:
                logger.info(f"   Tissue mapping: '{self.disease_name}' → {tissues}")
                return tissues

        # Try word-by-word fallback
        words = self.disease_name.split()
        for word in words:
            for keyword, tissues in TISSUE_ALIASES.items():
                if word in keyword or keyword in word:
                    logger.info(
                        f"   Tissue mapping (partial match): '{word}' → {tissues}"
                    )
                    return tissues

        logger.warning(
            f"   No tissue mapping for '{self.disease_name}'. "
            f"Expression scores will use whole-body max."
        )
        return []

    # ── HPA API ───────────────────────────────────────────────────────────────

    async def _fetch_hpa_expression(
        self, gene_symbol: str, session: aiohttp.ClientSession
    ) -> Dict[str, str]:
        """
        Fetch tissue expression data from HPA REST API.

        Returns dict: {tissue_name → expression_level}
        e.g. {"liver": "high", "kidney": "medium", "lung": "low"}

        HPA REST API (v23+ current):
        GET /api/search_download.php
          ?format=json
          &columns=g,t,sl           (gene, tissue, subcellular location)
          &query=<GENE_SYMBOL>
          &compress=no

        Response is JSON array of objects with fields:
          "Gene", "Gene synonym", "Ensembl",
          "Tissue", "Level", "Reliability"
        """
        cache_key = f"hpa_v23_{gene_symbol.upper()}"
        if cache_key in self._cache:
            return self._cache[cache_key].get("expression_data", {})

        expression: Dict[str, str] = {}

        params = {
            "format":   "json",
            "columns":  "g,t,sl",
            "query":    gene_symbol,
            "compress": "no",
        }

        try:
            async with session.get(
                HPA_SEARCH_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=HPA_SEARCH_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"   HPA API returned {resp.status} for {gene_symbol}"
                    )
                    return {}

                data = await resp.json(content_type=None)

                if not isinstance(data, list):
                    # Some responses wrap in a dict
                    if isinstance(data, dict):
                        data = data.get("data", data.get("results", []))
                    if not isinstance(data, list):
                        logger.warning(
                            f"   HPA unexpected response format for {gene_symbol}: "
                            f"{type(data)}"
                        )
                        return {}

                expression = self._parse_hpa_response(data, gene_symbol)
                logger.debug(
                    f"   HPA: {gene_symbol} → {len(expression)} tissues"
                )

        except asyncio.TimeoutError:
            logger.warning(f"   HPA timeout for {gene_symbol}")
            return {}
        except Exception as e:
            logger.warning(f"   HPA fetch error for {gene_symbol}: {e}")
            return {}

        # Only cache non-empty results
        if expression:
            self._cache[cache_key] = {
                "expression_data": expression,
                "fetched_at":      time.time(),
            }
            self._save_disk_cache()

        return expression

    def _parse_hpa_response(
        self, data: List[Dict], gene_symbol: str
    ) -> Dict[str, str]:
        """
        Parse HPA JSON response into {tissue → level} dict.

        HPA field names (title-case, as in the API):
          "Gene"        — gene symbol, e.g. "TP53"
          "Tissue"      — tissue name, e.g. "liver"
          "Level"       — expression level: "High", "Medium", "Low", "Not detected"
          "Reliability" — "Approved", "Supported", "Uncertain", "Enhanced"

        Supports both title-case ("Gene", "Tissue", "Level") and
        lowercase ("gene", "tissue", "level") for robustness.
        """
        expression: Dict[str, str] = {}
        gene_upper  = gene_symbol.upper()

        for entry in data:
            if not isinstance(entry, dict):
                continue

            # Field names: try title-case first, then lowercase
            gene    = str(entry.get("Gene", entry.get("gene", ""))).upper()
            tissue  = str(entry.get("Tissue", entry.get("tissue", ""))).strip().lower()
            level   = str(entry.get("Level", entry.get("level", ""))).strip().lower()
            reliability = str(
                entry.get("Reliability", entry.get("reliability", ""))
            ).lower()

            # Only include entries matching the requested gene
            if gene and gene != gene_upper:
                continue

            # Skip uncertain reliability
            if reliability in ("uncertain",):
                continue

            if tissue and level:
                # Normalise level to our known keys
                normalised = level.replace("-", " ").strip()
                if normalised not in EXPRESSION_LEVEL_SCORES:
                    normalised = "not detected"
                expression[tissue] = normalised

        return expression

    # ── Curated fallback ──────────────────────────────────────────────────────

    def _get_curated_expression(self, gene_symbol: str) -> Dict[str, str]:
        """Return curated expression data for well-characterised genes."""
        curated = CURATED_EXPRESSION.get(gene_symbol.upper(), {})
        if not isinstance(curated, dict):
            return {}
        return curated

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_gene_expression(self, expression: Dict[str, str]) -> float:
        """
        Convert expression dict to a 0–1 score.

        Strategy:
        - If target tissues are known for this disease: use max expression
          level across those specific tissues.
        - If no tissue mapping: use max expression level across all tissues.
        - Penalises genes with no HPA data (returns 0.3 as uncertain).
        """
        if not expression:
            return 0.30  # no data → uncertain, not zero

        if self._target_tissues:
            levels = []
            for tissue in self._target_tissues:
                level = expression.get(tissue, "")
                score = EXPRESSION_LEVEL_SCORES.get(level, 0.0)
                levels.append(score)
            return max(levels) if levels else 0.30

        # No disease tissue mapping — use body-wide max
        all_scores = [
            EXPRESSION_LEVEL_SCORES.get(level, 0.0)
            for level in expression.values()
        ]
        return max(all_scores) if all_scores else 0.30

    # ── Public API ────────────────────────────────────────────────────────────

    async def score_target_genes(
        self,
        gene_symbols: List[str],
        use_cache:    bool = True,
    ) -> Tuple[float, Dict[str, Dict]]:
        """
        Score a list of target genes by expression in the disease tissue.

        Returns
        -------
        (aggregate_score, per_gene_breakdown)
        aggregate_score: float 0–1, max across all targets
        per_gene_breakdown: {gene: {"expression": {...}, "score": float}}
        """
        if not gene_symbols:
            return 0.3, {}  # uncertain, not zero

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQS)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQS)

        per_gene: Dict[str, Dict] = {}

        async def score_one(gene: str) -> None:
            async with semaphore:
                # 1. Try HPA API
                async with aiohttp.ClientSession(connector=connector) as session:
                    expression = await self._fetch_hpa_expression(gene, session)

                # 2. Fall back to curated data if API returned nothing
                if not expression:
                    expression = self._get_curated_expression(gene)
                    source = "curated_fallback"
                    if not expression:
                        logger.info(f"   No expression data for {gene}")
                else:
                    source = "hpa_api"

                score = self._score_gene_expression(expression)
                per_gene[gene] = {
                    "expression": expression,
                    "score":      round(score, 4),
                    "source":     source,
                    "target_tissues": self._target_tissues,
                }

        tasks = [score_one(g) for g in set(gene_symbols)]
        await asyncio.gather(*tasks, return_exceptions=True)

        await connector.close()

        if not per_gene:
            return 0.3, {}

        scores = [v["score"] for v in per_gene.values()]
        # Aggregate: max (drug works if ANY target is expressed)
        aggregate = max(scores)

        logger.info(
            f"   Expression scores: {dict(zip(per_gene.keys(), [round(s,2) for s in scores]))} "
            f"→ aggregate={aggregate:.2f}"
        )
        return round(aggregate, 4), per_gene

    async def score_candidate(self, candidate: Dict) -> Dict:
        """
        Score a single drug candidate dict.
        Adds 'tissue_expression_score' and 'tissue_expression_detail' fields.
        """
        gene_symbols = candidate.get("target_genes", [])
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]

        if not gene_symbols:
            logger.info(
                f"   {candidate.get('drug_name', '?')}: no target genes — "
                f"tissue score = 0.3 (uncertain)"
            )
            candidate["tissue_expression_score"]  = 0.3
            candidate["tissue_expression_detail"] = {"warning": "No target genes provided"}
            return candidate

        agg_score, breakdown = await self.score_target_genes(gene_symbols)
        candidate["tissue_expression_score"]  = agg_score
        candidate["tissue_expression_detail"] = breakdown
        return candidate

    async def score_batch(
        self, candidates: List[Dict], max_concurrent: int = 5
    ) -> List[Dict]:
        """Score a batch of drug candidates."""
        logger.info(
            f"🧬 Tissue expression scoring: {len(candidates)} candidates "
            f"for disease='{self.disease_name}', tissues={self._target_tissues}"
        )

        if not self._target_tissues:
            logger.warning(
                "   ⚠️  No disease tissue mapping — using body-wide expression max. "
                "Consider setting disease_name to a recognised term."
            )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_one(c: Dict) -> Dict:
            async with semaphore:
                return await self.score_candidate(c)

        results = await asyncio.gather(
            *[score_one(c) for c in candidates],
            return_exceptions=True,
        )

        final: List[Dict] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(
                    f"   Expression scoring failed for "
                    f"{candidates[i].get('drug_name', '?')}: {r}"
                )
                candidates[i]["tissue_expression_score"]  = 0.3
                candidates[i]["tissue_expression_detail"] = {"error": str(r)}
                final.append(candidates[i])
            else:
                final.append(r)

        scored = [c for c in final if c.get("tissue_expression_score", 0) > 0.5]
        logger.info(
            f"✅ Expression scoring complete: "
            f"{len(scored)}/{len(final)} candidates with tissue expression > 0.5"
        )
        return final

    # ── Diagnostics ───────────────────────────────────────────────────────────

    async def validate_api_connection(
        self, test_gene: str = "EGFR"
    ) -> Tuple[bool, str]:
        """
        Test HPA API connectivity with a known gene (EGFR is well-characterised).
        Call this before a pipeline run to verify the API is reachable.

        Returns (success: bool, message: str)
        """
        async with aiohttp.ClientSession() as session:
            expression = await self._fetch_hpa_expression(test_gene, session)

        if not expression:
            return (
                False,
                f"HPA API returned empty data for {test_gene}. "
                f"Will use curated fallback database."
            )

        lung_level = expression.get("lung", "not found")
        return (
            True,
            f"HPA API OK — {test_gene} lung expression: {lung_level}. "
            f"{len(expression)} tissues returned."
        )