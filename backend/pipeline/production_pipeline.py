import asyncio
import aiohttp
import logging
import math
import time
from typing import Dict, List, Optional

from .data_fetcher import ProductionDataFetcher
from .graph_builder import ProductionGraphBuilder
from .ppi_network import PPINetworkScorer, batch_ppi_scores
from .drug_similarity import DrugSimilarityScorer, build_reference_smiles, batch_similarity_scores
from .scorer import ProductionScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


class ProductionPipeline:
    """
    Main pipeline for drug repurposing analysis.
    All scoring is based on live API data — no hardcoded drug-disease knowledge.
    """

    def __init__(self):
        self.data_fetcher  = ProductionDataFetcher()
        self.graph_builder = ProductionGraphBuilder()
        self.scorer:  Optional[ProductionScorer] = None
        self.disease_cache: Dict = {}
        self._ppi_scorer = None
        self._sim_scorer = None
        self.drugs_cache:   Optional[List[Dict]] = None
        self._pubmed_cache: Dict[str, float] = {}
        self._pubmed_session: Optional[aiohttp.ClientSession] = None

    # ── PubMed helper ─────────────────────────────────────────────────────────
    async def _fetch_pubmed_score(self, drug_name: str, disease_name: str) -> float:
        key = f"{drug_name.lower()}|{disease_name.lower()}"
        if key in self._pubmed_cache:
            return self._pubmed_cache[key]

        try:
            if self._pubmed_session is None or self._pubmed_session.closed:
                self._pubmed_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )
            params = {
                "db":      "pubmed",
                "term":    f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]',
                "retmax":  "0",
                "retmode": "json",
            }
            async with self._pubmed_session.get(PUBMED_ESEARCH, params=params) as resp:
                if resp.status != 200:
                    self._pubmed_cache[key] = 0.0
                    return 0.0
                data  = await resp.json()
                count = int(data.get("esearchresult", {}).get("count", 0))
        except Exception as e:
            logger.debug(f"PubMed lookup failed for {drug_name}/{disease_name}: {e}")
            self._pubmed_cache[key] = 0.0
            return 0.0

        score = math.log10(count + 1) / math.log10(201)
        score = min(score, 1.0)
        self._pubmed_cache[key] = score

        if score > 0.05:
            logger.info(
                f"📚 PubMed: {drug_name} ↔ {disease_name} — "
                f"{count} hits → lit_score={score:.3f}"
            )
        return score

    # ── Public drug fetcher ───────────────────────────────────────────────────
    async def fetch_approved_drugs(self, limit: int = 3000) -> List[Dict]:
        if self.drugs_cache is None:
            self.drugs_cache = await self.data_fetcher.fetch_approved_drugs(limit=limit)
            logger.info(f"✅ Fetched {len(self.drugs_cache)} approved drugs (cached on pipeline)")
        else:
            logger.info(f"✅ Using cached drug data ({len(self.drugs_cache)} drugs)")
        return self.drugs_cache

    # ── generate_candidates() — callable by benchmark/validation scripts ──────
    async def generate_candidates(
        self,
        disease_data:     Dict,
        drugs_data:       List[Dict],
        min_score:        float = 0.0,
        fetch_pubmed:     bool  = False,
        fetch_ppi:        bool  = True,
        fetch_similarity: bool  = True,
        use_efo:          bool  = True,
        use_tissue:       bool  = True,
        use_polypharm:    bool  = True,
    ) -> List[Dict]:
        """
        Score all drugs against a disease and return candidate dicts.

        v6.1: FIX — tissue expression now blended at 15% weight instead of
        additive, preventing score inflation (everything → 1.0) for diseases
        where most genes are ubiquitously expressed (e.g. bone marrow in MM).
        """
        disease_name = disease_data["name"]

        # ── MODULE 1: EFO Ontology Expansion ──────────────────────────────────
        expander = None
        if use_efo:
            try:
                from efo_ontology import EFOOntologyExpander
                expander = EFOOntologyExpander(session=self.data_fetcher.session)
                disease_data = await expander.expand_disease_genes(disease_data)
                stats = disease_data.get("efo_expansion_stats", {})
                logger.info(
                    f"🌳 EFO: {stats.get('original_gene_count')} → "
                    f"{stats.get('total_gene_count')} genes "
                    f"(+{stats.get('new_genes_added')} from ontology tree)"
                )
            except Exception as e:
                logger.warning(f"EFO expansion failed (non-fatal): {e}")
            finally:
                if expander is not None:
                    try:
                        await expander.close()
                    except Exception:
                        pass

        disease_genes = disease_data.get("genes", [])

        # ── Build graph & scorer ──────────────────────────────────────────────
        graph  = self.graph_builder.build_graph(disease_data, drugs_data)
        scorer = ProductionScorer(graph)

        # ── PubMed scores ─────────────────────────────────────────────────────
        pubmed_score_map: Dict[str, float] = {}
        if fetch_pubmed:
            drugs_with_targets = [d for d in drugs_data if d.get("targets")]
            pubmed_tasks = [
                self._fetch_pubmed_score(d["name"], disease_name)
                for d in drugs_with_targets
            ]
            scores_list = await asyncio.gather(*pubmed_tasks, return_exceptions=True)
            pubmed_score_map = {
                d["name"]: (s if isinstance(s, float) else 0.0)
                for d, s in zip(drugs_with_targets, scores_list)
            }

        # ── PPI network proximity ─────────────────────────────────────────────
        ppi_score_map: Dict[str, float] = {}
        if fetch_ppi and disease_genes:
            if self._ppi_scorer is None:
                self._ppi_scorer = PPINetworkScorer()
            try:
                logger.info(f"🔗 Fetching STRING PPI scores for {disease_name}...")
                ppi_results = await batch_ppi_scores(
                    drugs_data=drugs_data,
                    disease_genes=disease_genes,
                    scorer=self._ppi_scorer,
                )
                ppi_score_map = {name: score for name, (score, _) in ppi_results.items()}
                n_nonzero = sum(1 for s in ppi_score_map.values() if s > 0)
                logger.info(f"✅ PPI scores: {n_nonzero}/{len(drugs_data)} drugs with network proximity")
            except Exception as e:
                logger.warning(f"PPI scoring failed (continuing without it): {e}")
        elif fetch_ppi and not disease_genes:
            logger.warning("PPI skipped: no disease genes available")

        # ── Drug-drug chemical similarity ─────────────────────────────────────
        sim_score_map: Dict[str, float] = {}
        if fetch_similarity:
            if self._sim_scorer is None:
                self._sim_scorer = DrugSimilarityScorer()
            try:
                ref_smiles, ref_names = build_reference_smiles(disease_name, drugs_data)
                if ref_smiles:
                    logger.info(f"🧪 Computing drug similarity vs {len(ref_smiles)} known {disease_name} drugs...")
                    sim_results = batch_similarity_scores(
                        drugs_data=drugs_data,
                        reference_smiles=ref_smiles,
                        reference_names=ref_names,
                        scorer=self._sim_scorer,
                    )
                    sim_score_map = {name: score for name, (score, _) in sim_results.items()}
                    n_hits = sum(1 for s in sim_score_map.values() if s > 0)
                    logger.info(f"✅ Similarity scores: {n_hits}/{len(drugs_data)} drugs above threshold")
                else:
                    logger.info(f"Drug similarity skipped: no reference drugs found for '{disease_name}'")
            except Exception as e:
                logger.warning(f"Drug similarity scoring failed (continuing without it): {e}")

        # ── MODULE 2: Tissue Expression ───────────────────────────────────────
        tissue_score_map: Dict[str, float] = {}
        if use_tissue:
            try:
                from .tissue_expression import TissueExpressionScorer
                tef = TissueExpressionScorer(disease_name=disease_name)
                _stub_candidates = [
                    {"name": d["name"], "drug_name": d["name"], "target_genes": d.get("targets", [])}
                    for d in drugs_data
                ]
                _scored = await tef.score_batch(_stub_candidates)
                tissue_score_map = {
                    c["name"]: c.get("tissue_expression_score", 0.0)
                    for c in _scored
                }
                n_tissue = sum(1 for s in tissue_score_map.values() if s > 0)
                logger.info(f"🧬 Tissue expression: {n_tissue}/{len(drugs_data)} drugs with non-zero score")
                try:
                    await tef.close()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Tissue expression scoring failed (non-fatal): {e}")

        # ── Score all drugs ───────────────────────────────────────────────────
        candidates = []
        for drug in drugs_data:
            drug_name    = drug["name"]
            lit_score    = pubmed_score_map.get(drug_name, 0.0)
            ppi_score    = ppi_score_map.get(drug_name, 0.0)
            sim_score    = sim_score_map.get(drug_name, 0.0)
            tissue_score = tissue_score_map.get(drug_name, 0.0)

            score, evidence = scorer.score_drug_disease_match(
                drug_name,
                disease_name,
                disease_data,
                drug,
                external_literature_score=lit_score,
                ppi_score=ppi_score,
                similarity_score=sim_score,
            )

            # FIX v6.1: Tissue expression blended at 15% weight, NOT additive.
            # Additive approach caused everything to clamp to 1.0 for blood
            # cancers where most genes are expressed in bone marrow.
            # Weighted blend preserves ranking while incorporating tissue signal.
            if tissue_score > 0:
                score = min(1.0, score * 0.85 + tissue_score * 0.15)

            if score >= min_score:
                candidates.append({
                    "name":                   drug_name,
                    "drug_name":              drug_name,
                    "drug_id":                drug.get("id", ""),
                    "score":                  score,
                    "confidence":             evidence["confidence"],
                    "shared_genes":           evidence["shared_genes"],
                    "shared_pathways":        evidence["shared_pathways"],
                    "explanation":            evidence["explanation"],
                    "indication":             drug.get("indication", ""),
                    "mechanism":              drug.get("mechanism", ""),
                    "gene_score":             evidence["gene_score"],
                    "pathway_score":          evidence["pathway_score"],
                    "ppi_score":              evidence["ppi_score"],
                    "similarity_score":       evidence["similarity_score"],
                    "mechanism_score":        evidence["mechanism_score"],
                    "literature_score":       evidence["literature_score"],
                    "tissue_expression_score": tissue_score,
                    "polypharmacology_score": 0.0,
                    "target_genes": [t.upper() for t in (drug.get("targets") or [])],
                })

        # ── MODULE 3: Polypharmacology Scoring ────────────────────────────────
        if use_polypharm and candidates:
            try:
                from .polypharmacology import PolypharmacologyScorer
                poly_scorer = PolypharmacologyScorer(disease_name=disease_name)
                candidates.sort(key=lambda c: c["score"], reverse=True)
                top_50 = candidates[:50]
                top_50 = poly_scorer.score_batch(top_50, disease_targets=disease_genes)

                for c in top_50:
                    poly = c.get("polypharmacology_score", 0.0)
                    if poly > 0:
                        c["score"] = min(1.0, c["score"] + poly * 0.3)

                top_50_names = {c["name"] for c in top_50}
                rest = [c for c in candidates if c["name"] not in top_50_names]
                candidates = top_50 + rest

                n_poly = sum(1 for c in top_50 if c.get("polypharmacology_score", 0) > 0)
                logger.info(f"💊 Polypharmacology: {n_poly}/50 top candidates with poly score")
            except Exception as e:
                logger.warning(f"Polypharmacology scoring failed (non-fatal): {e}")

        return candidates

    # ── Main entry point ──────────────────────────────────────────────────────
    async def analyze_disease(
        self,
        disease_name: str,
        top_n_for_trial: int = 10,
    ) -> Dict:
        """
        Enhanced pipeline analysis with all 4 new modules integrated.
        """
        logger.info(f"[1/9] Fetching disease data: {disease_name}")
        disease_data = await self.data_fetcher.fetch_disease_data(disease_name)

        if not disease_data:
            return {"error": f"Disease not found: {disease_name}"}

        logger.info(f"[2/9] EFO ontology expansion...")
        expander = None
        try:
            from efo_ontology import EFOOntologyExpander
            expander = EFOOntologyExpander(session=self.data_fetcher.session)
            disease_data = await expander.expand_disease_genes(disease_data)
            stats = disease_data.get("efo_expansion_stats", {})
            logger.info(
                f"     EFO: {stats.get('original_gene_count')} → "
                f"{stats.get('total_gene_count')} genes "
                f"(+{stats.get('new_genes_added')} from ontology tree)"
            )
        except Exception as e:
            logger.warning(f"     EFO expansion failed (non-fatal): {e}")
        finally:
            if expander is not None:
                try:
                    await expander.close()
                except Exception:
                    pass

        logger.info(f"[3/9] Fetching and scoring all drugs...")
        drugs_data = await self.fetch_approved_drugs(limit=3000)

        logger.info(f"[4/9] Running full scoring pipeline (PPI + similarity + tissue + poly)...")
        candidates = await self.generate_candidates(
            disease_data=disease_data,
            drugs_data=drugs_data,
            min_score=0.0,
            fetch_pubmed=False,
            use_efo=False,
            use_tissue=True,
            use_polypharm=True,
        )
        logger.info(f"     Scored {len(candidates)} candidates")

        candidates.sort(key=lambda c: c["score"], reverse=True)

        logger.info(f"[6/9] Safety filtering...")
        safe_candidates = [
            c for c in candidates
            if not c.get("safety_concerns") or len(c.get("safety_concerns", [])) < 3
        ]
        logger.info(f"     {len(safe_candidates)} candidates passed safety filter")

        top_50 = safe_candidates[:50]
        synergistic_pairs = []

        logger.info(f"[7/9] Polypharmacology complete (ran inside scoring pipeline)")

        top_50.sort(key=lambda c: c["score"], reverse=True)

        logger.info(f"[8/9] Running virtual clinical trials (top {top_n_for_trial})...")
        trial_results = []
        trial_report  = ""
        try:
            from .insilico_trial import InSilicoTrialSimulator
            top_for_trial = top_50[:top_n_for_trial]
            for c in top_for_trial:
                c["disease_genes"] = disease_data.get("genes", [])
            simulator     = InSilicoTrialSimulator(disease=disease_name, n_patients=200)
            trial_results = await simulator.run_batch(top_for_trial)
            trial_report  = simulator.generate_trial_report(trial_results)
            logger.info("     Virtual trials complete")
        except Exception as e:
            logger.warning(f"     Virtual trials failed (non-fatal): {e}")

        logger.info(f"[9/9] Assembling final report...")
        return {
            "disease":            disease_name,
            "disease_data":       disease_data,
            "top_candidates":     top_50[:20],
            "synergistic_pairs":  synergistic_pairs[:5],
            "trial_results":      trial_results,
            "trial_report":       trial_report,
            "pipeline_stats": {
                "total_candidates_evaluated":  len(candidates),
                "after_safety_filter":         len(safe_candidates),
                "efo_expansion": disease_data.get("efo_expansion_stats", {}),
            },
        }

    async def close(self):
        await self.data_fetcher.close()
        if self._pubmed_session and not self._pubmed_session.closed:
            await self._pubmed_session.close()
        if self._ppi_scorer:
            await self._ppi_scorer.close()
        logger.info("🔒 Pipeline closed")


# ── Aliases ───────────────────────────────────────────────────────────────────
RepurposingPipeline = ProductionPipeline


async def analyze(
    disease_name: str, min_score: float = 0.2, max_results: int = 20
) -> Dict:
    pipeline = ProductionPipeline()
    try:
        return await pipeline.analyze_disease(disease_name, min_score, max_results)
    finally:
        await pipeline.close()