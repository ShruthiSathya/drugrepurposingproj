"""
tissue_expression.py — Tissue Expression Scoring v3.0
=======================================================
Scores drug candidates based on whether their protein targets are expressed
in the disease-relevant tissue, using the OpenTargets Platform API.

WHY OPENTARGETS (NOT HPA)
--------------------------
HPA REST API problems:
  - search_download.php returns entire database dump, ignores gene symbol filter
  - Direct Ensembl JSON only returns tissue-ENRICHED genes in nTPM field
  - Ubiquitously expressed drug targets (ABL1, BCL2, MTOR, TP53) return {}
  - Result: ~90% of scored genes got 0.3 uncertain fallback — no real signal

OpenTargets advantages:
  - Returns full tissue expression profile (119 tissues for CRBN)
  - Integer levels -1..4 mapped from GTEx/HPA RNA-seq
  - nTPM values for precise quantification
  - Already used by this pipeline — consistent infrastructure
  - Reliable, well-maintained API

API
---
POST https://api.platform.opentargets.org/api/v4/graphql

Two queries used:
  1. search(queryString, entityNames=["target"]) — symbol → Ensembl ID
  2. target(ensemblId) { expressions { tissue { label } rna { level value } } }

SCORING
-------
OpenTargets RNA level scale (from GTEx/HPA):
  -1 = not detected
   0 = low
   1 = low-medium
   2 = medium
   3 = medium-high
   4 = high

Score mapping:
  4 → 1.00   3 → 0.75   2 → 0.50   1 → 0.25   0 → 0.10   -1 → 0.00

Aggregate: max across all target genes (drug works if ANY target expressed)
Tissue match: score against disease-relevant tissues; fall back to body-wide
              max if no tissue mapping or tissue labels not found.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

CACHE_DIR          = Path("/tmp/drug_repurposing_cache")
EXPR_CACHE_FILE    = CACHE_DIR / "tissue_expression_v3_cache.json"
ENSEMBL_CACHE_FILE = CACHE_DIR / "gene_ensembl_cache.json"
CACHE_TTL_SECS     = 14 * 24 * 3600   # 2 weeks
MAX_CONCURRENT     = 4

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# ─────────────────────────────────────────────────────────────────────────────
# RNA level → score
# ─────────────────────────────────────────────────────────────────────────────
LEVEL_SCORES: Dict[int, float] = {
    5:  1.00,
    4:  1.00,
    3:  0.75,
    2:  0.50,
    1:  0.25,
    0:  0.10,
    -1: 0.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# Disease → target tissue labels (must match OpenTargets tissue.label strings)
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_MAP: Dict[str, List[str]] = {
    # Oncology
    "multiple myeloma":               ["bone marrow", "plasma cell"],
    "leukemia":                       ["bone marrow", "blood", "lymph node"],
    "lymphoma":                       ["lymph node", "spleen", "blood"],
    "pancreatic":                     ["pancreas"],
    "pdac":                           ["pancreas"],
    "glioblastoma":                   ["cerebral cortex", "brain", "caudate nucleus"],
    "gbm":                            ["cerebral cortex", "brain"],
    "glioma":                         ["cerebral cortex", "brain"],
    "lung cancer":                    ["lung"],
    "nsclc":                          ["lung"],
    "breast cancer":                  ["breast"],
    "breast carcinoma":               ["breast"],
    "colorectal":                     ["colon", "rectum", "sigmoid colon"],
    "colon cancer":                   ["colon"],
    "ovarian":                        ["ovary"],
    "melanoma":                       ["skin"],
    "hepatocellular":                 ["liver"],
    "liver cancer":                   ["liver"],
    "renal cell":                     ["kidney"],
    "bladder cancer":                 ["urinary bladder"],
    "prostate cancer":                ["prostate gland"],
    "thyroid cancer":                 ["thyroid gland"],
    "gastric cancer":                 ["stomach"],
    "esophageal":                     ["esophagus"],
    # Autoimmune / Inflammatory
    "rheumatoid arthritis":           ["synovial membrane", "fibroblast", "macrophage"],
    "systemic lupus":                 ["kidney", "skin", "blood"],
    "lupus":                          ["kidney", "skin"],
    "inflammatory bowel":             ["colon", "small intestine", "ileum"],
    "crohn":                          ["small intestine", "ileum", "colon"],
    "ulcerative colitis":             ["colon", "sigmoid colon"],
    "psoriasis":                      ["skin", "epidermis"],
    "multiple sclerosis":             ["cerebral cortex", "brain", "spinal cord"],
    # Cardiovascular
    "heart failure":                  ["heart", "heart muscle", "cardiac muscle cell"],
    "cardiomyopathy":                 ["heart", "heart muscle"],
    "pulmonary arterial hypertension":["lung", "heart"],
    "pericarditis":                   ["heart"],
    "atherosclerosis":                ["aorta", "blood vessel"],
    # Neurological
    "parkinson":                      ["substantia nigra", "caudate nucleus", "brain"],
    "alzheimer":                      ["cerebral cortex", "hippocampus", "brain"],
    "epilepsy":                       ["hippocampus", "cerebral cortex", "brain"],
    "amyotrophic lateral sclerosis":  ["spinal cord", "motor neuron", "brain"],
    "als":                            ["spinal cord", "motor neuron"],
    "huntington":                     ["caudate nucleus", "cerebral cortex", "brain"],
    # Metabolic
    "type 2 diabetes":                ["pancreas", "adipose tissue", "liver", "skeletal muscle"],
    "diabetes":                       ["pancreas", "adipose tissue", "liver"],
    "obesity":                        ["adipose tissue"],
    "hypercholesterolemia":           ["liver"],
    "gout":                           ["kidney", "synovial membrane"],
    # Pulmonary
    "asthma":                         ["lung", "bronchus"],
    "copd":                           ["lung"],
    "cystic fibrosis":                ["lung", "pancreas"],
    # Renal
    "chronic kidney disease":         ["kidney"],
    "ckd":                            ["kidney"],
    # Rare / Genetic
    "tuberous sclerosis":             ["brain", "kidney"],
    "spinal muscular atrophy":        ["spinal cord", "motor neuron"],
    "sma":                            ["spinal cord"],
    # Generic anatomical fallbacks
    "brain":                          ["cerebral cortex", "brain", "hippocampus"],
    "liver":                          ["liver"],
    "kidney":                         ["kidney"],
    "lung":                           ["lung"],
    "heart":                          ["heart", "heart muscle"],
    "skin":                           ["skin"],
    "colon":                          ["colon"],
    "bone marrow":                    ["bone marrow"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Static gene symbol → Ensembl ID for the most common drug targets
# Saves an API round-trip for genes we already know
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_ENSEMBL: Dict[str, str] = {
    "KRAS":   "ENSG00000133703", "BRAF":   "ENSG00000157764",
    "EGFR":   "ENSG00000146648", "ERBB2":  "ENSG00000141736",
    "TP53":   "ENSG00000141510", "MYC":    "ENSG00000136997",
    "CDK4":   "ENSG00000135446", "CDK6":   "ENSG00000105810",
    "PIK3CA": "ENSG00000121879", "PTEN":   "ENSG00000171862",
    "ALK":    "ENSG00000171094", "MET":    "ENSG00000105976",
    "RET":    "ENSG00000165731", "MTOR":   "ENSG00000198793",
    "AKT1":   "ENSG00000142208", "ESR1":   "ENSG00000091831",
    "PGR":    "ENSG00000082175", "ABL1":   "ENSG00000097007",
    "BCL2":   "ENSG00000171791", "BRCA1":  "ENSG00000012048",
    "BRCA2":  "ENSG00000139618", "CRBN":   "ENSG00000141526",
    "CD274":  "ENSG00000120217", "PDCD1":  "ENSG00000188389",
    "CTLA4":  "ENSG00000163599", "CD19":   "ENSG00000177455",
    "MS4A1":  "ENSG00000156738", "SNCA":   "ENSG00000145335",
    "LRRK2":  "ENSG00000188906", "MAPT":   "ENSG00000186868",
    "APP":    "ENSG00000142192", "PSEN1":  "ENSG00000080815",
    "APOE":   "ENSG00000130203", "TREM2":  "ENSG00000095970",
    "HTT":    "ENSG00000197386", "SOD1":   "ENSG00000142168",
    "TARDBP": "ENSG00000120948", "NPPA":   "ENSG00000175206",
    "NPPB":   "ENSG00000120937", "MYH7":   "ENSG00000092054",
    "ACE":    "ENSG00000159640", "ADRB1":  "ENSG00000043591",
    "ADRB2":  "ENSG00000169252", "TNF":    "ENSG00000232810",
    "IL6":    "ENSG00000136244", "IL1B":   "ENSG00000125538",
    "JAK1":   "ENSG00000162434", "JAK2":   "ENSG00000096968",
    "STAT3":  "ENSG00000168610", "NLRP3":  "ENSG00000162711",
    "INSR":   "ENSG00000171105", "PPARG":  "ENSG00000132170",
    "PCSK9":  "ENSG00000169174", "HMGCR":  "ENSG00000113161",
    "LDLR":   "ENSG00000130164", "CFTR":   "ENSG00000001626",
    "SRC":    "ENSG00000197122", "BTK":    "ENSG00000010671",
    "VEGFA":  "ENSG00000112715", "KDR":    "ENSG00000128052",
    "DRD2":   "ENSG00000149295", "HTR2A":  "ENSG00000102468",
    "PRKAA1": "ENSG00000132356", "IDH1":   "ENSG00000138413",
    "IDH2":   "ENSG00000182054", "FLT3":   "ENSG00000122025",
    "KIT":    "ENSG00000157404", "PDGFRA": "ENSG00000134853",
    "FGFR1":  "ENSG00000077782", "FGFR2":  "ENSG00000066468",
    "FGFR3":  "ENSG00000068078", "AR":     "ENSG00000169083",
    "VHL":    "ENSG00000134086", "MDM2":   "ENSG00000135679",
    "NPM1":   "ENSG00000181163", "DNMT3A": "ENSG00000119772",
    "TET2":   "ENSG00000168769", "ASXL1":  "ENSG00000171456",
    "BCR":    "ENSG00000186716", "JAK3":   "ENSG00000105639",
    "IKBKB":  "ENSG00000104365", "NFKB1":  "ENSG00000109320",
    "MAP2K1": "ENSG00000169032", "MAP2K2": "ENSG00000126934",
    "MAPK1":  "ENSG00000100030", "MAPK3":  "ENSG00000102882",
    "HDAC1":  "ENSG00000116478", "HDAC2":  "ENSG00000196591",
    "DNMT1":  "ENSG00000130816", "EZH2":   "ENSG00000106462",
    "BRD4":   "ENSG00000141867", "MYD88":  "ENSG00000172936",
    "IRAK4":  "ENSG00000198001", "RIPK2":  "ENSG00000104312",
    "PTPN11": "ENSG00000179295", "SHP2":   "ENSG00000179295",
    "PRMT5":  "ENSG00000100150", "SMARCA4":"ENSG00000127616",
}


class TissueExpressionScorer:
    """
    Scores drug candidates by protein target expression in disease-relevant tissue.
    Uses OpenTargets GraphQL API — same API already used by this pipeline.
    """

    def __init__(self, disease_name: str = ""):
        self.disease_name    = disease_name.lower().strip()
        self._expr_cache: Dict    = {}
        self._ensembl_cache: Dict = {}
        self._load_caches()
        self._target_tissues = self._resolve_target_tissues()

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _load_caches(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for path, attr in [(EXPR_CACHE_FILE, "_expr_cache"),
                           (ENSEMBL_CACHE_FILE, "_ensembl_cache")]:
            if path.exists():
                try:
                    raw = json.loads(path.read_text())
                    setattr(self, attr, {
                        k: v for k, v in raw.items()
                        if v.get("fetched_at", 0) + CACHE_TTL_SECS > now
                    })
                    logger.debug(
                        f"Loaded {len(getattr(self, attr))} from {path.name}"
                    )
                except Exception as e:
                    logger.warning(f"Cache load failed ({path.name}): {e}")

    def _save_cache(self, path: Path, data: Dict) -> None:
        try:
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Cache save failed ({path.name}): {e}")

    # ── Tissue resolution ─────────────────────────────────────────────────────

    def _resolve_target_tissues(self) -> List[str]:
        if not self.disease_name:
            return []
        # Longest keyword match wins
        best_key, best_tissues = "", []
        for keyword, tissues in TISSUE_MAP.items():
            if keyword in self.disease_name and len(keyword) > len(best_key):
                best_key, best_tissues = keyword, tissues
        if best_tissues:
            logger.info(f"   Tissue mapping: '{self.disease_name}' → {best_tissues}")
            return best_tissues
        # Word-level fallback
        for word in self.disease_name.split():
            for keyword, tissues in TISSUE_MAP.items():
                if word in keyword or keyword in word:
                    logger.info(f"   Tissue mapping (partial '{word}'): {tissues}")
                    return tissues
        logger.warning(
            f"   No tissue mapping for '{self.disease_name}' — using body-wide max"
        )
        return []

    # ── Ensembl ID resolution ─────────────────────────────────────────────────

    async def _resolve_ensembl_id(
        self, gene_symbol: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """Resolve gene symbol → Ensembl ID via static map then OT search."""
        gene_upper = gene_symbol.upper()

        if gene_upper in KNOWN_ENSEMBL:
            return KNOWN_ENSEMBL[gene_upper]

        if gene_upper in self._ensembl_cache:
            return self._ensembl_cache[gene_upper].get("ensembl_id")

        query = """
        query GeneSearch($q: String!) {
          search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
            hits {
              object {
                ... on Target {
                  id
                  approvedSymbol
                }
              }
            }
          }
        }
        """
        try:
            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": query, "variables": {"q": gene_symbol}},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data  = await resp.json()
                hits  = (data.get("data", {})
                             .get("search", {})
                             .get("hits", []))
                if not hits:
                    return None
                obj    = hits[0].get("object", {})
                symbol = obj.get("approvedSymbol", "").upper()
                eid    = obj.get("id", "")
                if symbol != gene_upper or not eid:
                    return None
                self._ensembl_cache[gene_upper] = {
                    "ensembl_id": eid,
                    "fetched_at": time.time(),
                }
                self._save_cache(ENSEMBL_CACHE_FILE, self._ensembl_cache)
                return eid
        except Exception as e:
            logger.warning(f"   Ensembl lookup failed for {gene_symbol}: {e}")
            return None

    # ── Expression fetch ──────────────────────────────────────────────────────

    async def _fetch_expression(
        self, ensembl_id: str, session: aiohttp.ClientSession
    ) -> Dict[str, Dict]:
        """
        Fetch full tissue expression profile from OpenTargets.
        Returns {tissue_label_lower: {"level": int, "value": float}}
        """
        if ensembl_id in self._expr_cache:
            return self._expr_cache[ensembl_id].get("expression_data", {})

        query = """
        query TissueExpression($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            approvedSymbol
            expressions {
              tissue {
                label
              }
              rna {
                level
                value
              }
            }
          }
        }
        """
        try:
            async with session.post(
                OT_GRAPHQL_URL,
                json={"query": query, "variables": {"ensemblId": ensembl_id}},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"   OT API {resp.status} for {ensembl_id}"
                    )
                    return {}
                data   = await resp.json()
                target = data.get("data", {}).get("target")
                if not target:
                    return {}
                raw = target.get("expressions", [])
        except Exception as e:
            logger.warning(f"   OT fetch failed for {ensembl_id}: {e}")
            return {}

        expression: Dict[str, Dict] = {}
        for entry in raw:
            label = entry.get("tissue", {}).get("label", "").lower().strip()
            rna   = entry.get("rna") or {}
            level = rna.get("level")
            value = rna.get("value", 0.0)
            if label and level is not None:
                expression[label] = {"level": int(level), "value": float(value)}

        if expression:
            self._expr_cache[ensembl_id] = {
                "expression_data": expression,
                "fetched_at":      time.time(),
            }
            self._save_cache(EXPR_CACHE_FILE, self._expr_cache)
            logger.debug(f"   OT: {ensembl_id} → {len(expression)} tissues")

        return expression

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_expression(self, expression: Dict[str, Dict]) -> float:
        """Score expression dict against disease target tissues."""
        if not expression:
            return 0.0

        def level_score(label: str) -> float:
            entry = expression.get(label)
            if entry:
                return LEVEL_SCORES.get(entry["level"], 0.0)
            # Partial match: "bone marrow" matches "bone marrow cell"
            for tissue_label, data in expression.items():
                if label in tissue_label or tissue_label in label:
                    return LEVEL_SCORES.get(data["level"], 0.0)
            return None  # not found in expression data

        if self._target_tissues:
            scores = [
                s for t in self._target_tissues
                if (s := level_score(t)) is not None
            ]
            if scores:
                return max(scores)
            logger.debug("   Target tissues not in expression data — using body-wide max")

        # Body-wide max fallback
        all_scores = [LEVEL_SCORES.get(v["level"], 0.0) for v in expression.values()]
        return max(all_scores) if all_scores else 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def score_target_genes(
        self, gene_symbols: List[str]
    ) -> Tuple[float, Dict]:
        """
        Score a list of target genes by expression in disease tissue.

        Returns (aggregate_score, per_gene_breakdown)
        aggregate_score: float 0–1, max across all targets
        """
        if not gene_symbols:
            return 0.0, {}

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        per_gene: Dict[str, Dict] = {}

        async def score_one(gene: str) -> None:
            async with semaphore:
                async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(),
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as session:
                    ensembl_id = await self._resolve_ensembl_id(gene, session)
                    if not ensembl_id:
                        logger.debug(f"   {gene}: Ensembl ID not resolved")
                        per_gene[gene] = {"score": 0.0, "ensembl_id": None,
                                          "reason": "ensembl_not_found"}
                        return
                    expression = await self._fetch_expression(ensembl_id, session)

            if not expression:
                logger.debug(f"   {gene}: no expression data from OT")
                per_gene[gene] = {"score": 0.0, "ensembl_id": ensembl_id,
                                  "reason": "no_expression_data"}
                return

            score = self._score_expression(expression)

            # Compact summary for logging/reporting
            if self._target_tissues:
                summary = {}
                for t in self._target_tissues:
                    entry = expression.get(t)
                    if entry:
                        summary[t] = entry
                    else:
                        for label, data in expression.items():
                            if t in label or label in t:
                                summary[label] = data
            else:
                # Top 5 by level
                summary = dict(
                    sorted(expression.items(),
                           key=lambda x: x[1]["level"], reverse=True)[:5]
                )

            per_gene[gene] = {
                "ensembl_id":         ensembl_id,
                "score":              round(score, 4),
                "target_tissues":     self._target_tissues,
                "expression_summary": summary,
                "total_tissues":      len(expression),
            }

        await asyncio.gather(
            *[score_one(g) for g in set(gene_symbols)],
            return_exceptions=True,
        )

        if not per_gene:
            return 0.0, {}

        scores    = [v["score"] for v in per_gene.values()]
        aggregate = max(scores)

        logger.info(
            f"   Tissue scores: "
            f"{ {g: round(v['score'], 2) for g, v in per_gene.items()} } "
            f"→ aggregate={aggregate:.2f}"
        )
        return round(aggregate, 4), per_gene

    async def score_candidate(self, candidate: Dict) -> Dict:
        """Score a single drug candidate. Adds tissue_expression_score field."""
        gene_symbols = candidate.get("target_genes", [])
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]

        if not gene_symbols:
            logger.info(
                f"   {candidate.get('drug_name', '?')}: "
                f"no target genes — tissue score = 0.0"
            )
            candidate["tissue_expression_score"]  = 0.0
            candidate["tissue_expression_detail"] = {"warning": "no target genes"}
            return candidate

        agg_score, breakdown = await self.score_target_genes(gene_symbols)
        candidate["tissue_expression_score"]  = agg_score
        candidate["tissue_expression_detail"] = breakdown
        return candidate

    async def score_batch(
        self, candidates: List[Dict], max_concurrent: int = 4
    ) -> List[Dict]:
        """Score a batch of drug candidates."""
        logger.info(
            f"🧬 Tissue expression scoring: {len(candidates)} candidates "
            f"for '{self.disease_name}', tissues={self._target_tissues}"
        )
        if not self._target_tissues:
            logger.warning(
                "   ⚠️  No tissue mapping — using body-wide expression max"
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
                    f"   Scoring failed for "
                    f"{candidates[i].get('drug_name', '?')}: {r}"
                )
                candidates[i]["tissue_expression_score"]  = 0.0
                candidates[i]["tissue_expression_detail"] = {"error": str(r)}
                final.append(candidates[i])
            else:
                final.append(r)

        scored = [c for c in final if c.get("tissue_expression_score", 0) > 0.5]
        logger.info(
            f"✅ Tissue scoring complete: "
            f"{len(scored)}/{len(final)} candidates with score > 0.5"
        )
        return final

    async def close(self) -> None:
        """No-op — sessions are per-request."""
        pass

    # ── Diagnostics ───────────────────────────────────────────────────────────

    async def validate_api_connection(
        self, test_gene: str = "CRBN"
    ) -> Tuple[bool, str]:
        """Test OpenTargets API connectivity with a known gene."""
        ensembl = KNOWN_ENSEMBL.get(test_gene.upper(), "ENSG00000141526")
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(),
            timeout=aiohttp.ClientTimeout(total=15),
        ) as session:
            expression = await self._fetch_expression(ensembl, session)

        if not expression:
            return False, f"OpenTargets returned no expression data for {test_gene}"

        n   = len(expression)
        top = sorted(
            expression.items(), key=lambda x: x[1]["level"], reverse=True
        )[:3]
        top_str = ", ".join(f"{t}=level{d['level']}" for t, d in top)
        return True, f"OK — {test_gene}: {n} tissues, top: {top_str}"