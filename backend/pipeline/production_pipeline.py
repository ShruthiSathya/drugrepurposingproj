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

    # ── Public drug fetcher (delegates to data_fetcher, uses cache) ───────────
    async def fetch_approved_drugs(self, limit: int = 3000) -> List[Dict]:
        """
        Fetch approved drugs, using internal cache if already loaded.
        Exposed as a top-level method so external callers (RepoDB benchmark,
        validation runner) can access it without reaching into data_fetcher.
        """
        if self.drugs_cache is None:
            self.drugs_cache = await self.data_fetcher.fetch_approved_drugs(limit=limit)
            logger.info(f"✅ Fetched {len(self.drugs_cache)} approved drugs (cached on pipeline)")
        else:
            logger.info(f"✅ Using cached drug data ({len(self.drugs_cache)} drugs)")
        return self.drugs_cache

    # ── generate_candidates() — callable by benchmark/validation scripts ──────
    async def generate_candidates(
        self,
        disease_data:  Dict,
        drugs_data:    List[Dict],
        min_score:     float = 0.0,
        fetch_pubmed:  bool  = False,
        fetch_ppi:     bool  = True,
        fetch_similarity: bool = True,
    ) -> List[Dict]:
        """
        Score all drugs against a disease and return candidate dicts.

        v5: Now includes PPI network proximity and drug-drug chemical similarity.

        Parameters
        ----------
        disease_data : dict
            Output of data_fetcher.fetch_disease_data().
        drugs_data : list of dict
            Output of fetch_approved_drugs().
        min_score : float
            Minimum composite score to include (default 0.0 = return all).
        fetch_pubmed : bool
            If True, fetch live PubMed co-occurrence scores (slower).
        fetch_ppi : bool
            If True, fetch STRING PPI network proximity scores.
            Requires internet access to string-db.org (cached after first run).
            Set False to skip for faster runs (e.g. during development).
        fetch_similarity : bool
            If True, compute Tanimoto drug-drug chemical similarity.
            Requires SMILES strings in drugs_data (already present from ChEMBL).
            Fast — no API call needed.

        Returns
        -------
        list of dict with keys:
            name, drug_name, drug_id, score, confidence,
            shared_genes, shared_pathways, explanation,
            indication, mechanism,
            gene_score, pathway_score, ppi_score, similarity_score,
            mechanism_score, literature_score
        """
        graph  = self.graph_builder.build_graph(disease_data, drugs_data)
        scorer = ProductionScorer(graph)

        disease_name  = disease_data["name"]
        disease_genes = disease_data.get("genes", [])

        # ── PubMed scores ─────────────────────────────────────────────────────────
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

        # ── PPI network proximity ─────────────────────────────────────────────────
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
                ppi_score_map = {}
        else:
            if fetch_ppi and not disease_genes:
                logger.warning("PPI skipped: no disease genes available")

        # ── Drug-drug chemical similarity ─────────────────────────────────────────
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
                sim_score_map = {}

        # ── Score all drugs ───────────────────────────────────────────────────────
        candidates = []
        for drug in drugs_data:
            drug_name    = drug["name"]
            lit_score    = pubmed_score_map.get(drug_name, 0.0)
            ppi_score    = ppi_score_map.get(drug_name, 0.0)
            sim_score    = sim_score_map.get(drug_name, 0.0)

            score, evidence = scorer.score_drug_disease_match(
                drug_name,
                disease_name,
                disease_data,
                drug,
                external_literature_score=lit_score,
                ppi_score=ppi_score,
                similarity_score=sim_score,
            )

            if score >= min_score:
                candidates.append({
                    "name":             drug_name,
                    "drug_name":        drug_name,
                    "drug_id":          drug.get("id", ""),
                    "score":            score,
                    "confidence":       evidence["confidence"],
                    "shared_genes":     evidence["shared_genes"],
                    "shared_pathways":  evidence["shared_pathways"],
                    "explanation":      evidence["explanation"],
                    "indication":       drug.get("indication", ""),
                    "mechanism":        drug.get("mechanism", ""),
                    "gene_score":       evidence["gene_score"],
                    "pathway_score":    evidence["pathway_score"],
                    "ppi_score":        evidence["ppi_score"],
                    "similarity_score": evidence["similarity_score"],
                    "mechanism_score":  evidence["mechanism_score"],
                    "literature_score": evidence["literature_score"],
                })

        return candidates

    # ── Main entry point ──────────────────────────────────────────────────────
    async def analyze_disease(
        self,                    # FIX: was `pipeline` (not a valid method parameter)
        disease_name: str,
        top_n_for_trial: int = 10,
    ) -> Dict:
        """
        Enhanced pipeline analysis with all 4 new modules integrated.

        Parameters
        ----------
        disease_name : str
            Disease name, e.g. "pancreatic ductal adenocarcinoma"
        top_n_for_trial : int
            Number of top candidates to pass to in-silico trial simulator.

        Returns
        -------
        dict with all results including virtual trial outcomes.
        """

        # ── STEP 1: Fetch disease data (existing) ──────────────────────────────
        logger.info(f"[1/9] Fetching disease data: {disease_name}")
        disease_data = await self.data_fetcher.fetch_disease_data(disease_name)

        if not disease_data:
            return {"error": f"Disease not found: {disease_name}"}

        # ── STEP 2: EFO Ontology Expansion (NEW: Module 1) ────────────────────
        logger.info(f"[2/9] EFO ontology expansion...")
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
            try:
                await expander.close()
            except Exception:
                pass

        # ── STEP 3: Fetch drug candidates (existing) ──────────────────────────
        logger.info(f"[3/9] Fetching drug candidates...")
        candidates = await self.data_fetcher.fetch_drug_candidates(disease_data)
        logger.info(f"     Found {len(candidates)} candidates")

        # ── STEP 4: Base scoring (existing) ───────────────────────────────────
        logger.info(f"[4/9] Scoring candidates (base)...")
        candidates = self.scorer.score_candidates(
            candidates, disease_data
        )

        # ── STEP 5: Tissue Expression Scoring (NEW: Module 2) ─────────────────
        logger.info(f"[5/9] Tissue expression scoring...")
        try:
            from tissue_expression import TissueExpressionFilter
            tef = TissueExpressionFilter(
                cancer_type=disease_name,
                session=self.data_fetcher.session,
            )
            candidates = await tef.score_candidates(candidates)
            # Apply tissue score to composite
            for c in candidates:
                c["composite_score"] = max(0.0, min(1.0,
                    c.get("composite_score", 0) + c.get("tissue_expression_score", 0)
                ))
            logger.info(f"     Tissue scoring applied to {len(candidates)} candidates")
        except Exception as e:
            logger.warning(f"     Tissue scoring failed (non-fatal): {e}")
        finally:
            try:
                await tef.close()
            except Exception:
                pass

        # Sort by updated composite score
        candidates.sort(key=lambda c: c.get("composite_score", 0), reverse=True)

        # ── STEP 6: Safety filter (existing) ──────────────────────────────────
        logger.info(f"[6/9] Safety filtering...")
        safe_candidates = [
            c for c in candidates
            if not c.get("safety_concerns") or len(c.get("safety_concerns", [])) < 3
        ]
        logger.info(f"     {len(safe_candidates)} candidates passed safety filter")

        # ── STEP 7: Polypharmacology Scoring (NEW: Module 3) ──────────────────
        logger.info(f"[7/9] Polypharmacology scoring...")
        top_50 = safe_candidates[:50]
        try:
            from polypharmacology import PolypharmacologyScorer
            poly_scorer = PolypharmacologyScorer(
                disease_genes=disease_data.get("genes", []),
                gene_scores=disease_data.get("gene_scores", {}),
            )
            top_50 = await poly_scorer.score_candidates(top_50)

            # Find synergistic drug pairs
            synergistic_pairs = await poly_scorer.find_synergistic_pairs(top_50[:20])
            logger.info(
                f"     Top pair: "
                f"{synergistic_pairs[0]['drug_a']} + {synergistic_pairs[0]['drug_b']} "
                f"(synergy: {synergistic_pairs[0]['synergy_score']:.3f})"
                if synergistic_pairs else "     No pairs computed"
            )
        except Exception as e:
            logger.warning(f"     Polypharmacology scoring failed (non-fatal): {e}")
            synergistic_pairs = []

        # Re-sort with poly contribution
        top_50.sort(
            key=lambda c: (
                c.get("composite_score", 0) +
                c.get("polypharmacology_score", 0) * 0.3
            ),
            reverse=True,
        )

        # ── STEP 8: In-Silico Trial Simulation (NEW: Module 4) ────────────────
        logger.info(f"[8/9] Running virtual clinical trials (top {top_n_for_trial})...")
        trial_results = []
        trial_report  = ""
        try:
            from insilico_trial import InSilicoTrialSimulator

            # Inject disease genes into candidates for network effect calculation
            top_for_trial = top_50[:top_n_for_trial]
            for c in top_for_trial:
                c["disease_genes"] = disease_data.get("genes", [])

            simulator    = InSilicoTrialSimulator(disease=disease_name, n_patients=200)
            trial_results = await simulator.run_batch(top_for_trial)
            trial_report  = simulator.generate_trial_report(trial_results)

            logger.info("     Virtual trials complete")
        except Exception as e:
            logger.warning(f"     Virtual trials failed (non-fatal): {e}")

        # ── STEP 9: Assemble final results ────────────────────────────────────
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

    # FIX: close() was previously indented inside analyze_disease, making it
    # unreachable as a class method. Now correctly at class level.
    async def close(self):
        await self.data_fetcher.close()
        if self._pubmed_session and not self._pubmed_session.closed:
            await self._pubmed_session.close()
        if self._ppi_scorer:
            await self._ppi_scorer.close()
        logger.info("🔒 Pipeline closed")


# ── Aliases — both names resolve to the same class ───────────────────────────
# run_validation.py and repodb_benchmark.py import RepurposingPipeline.
# ProductionPipeline is the canonical name; RepurposingPipeline is the alias.
RepurposingPipeline = ProductionPipeline


async def analyze(
    disease_name: str, min_score: float = 0.2, max_results: int = 20
) -> Dict:
    pipeline = ProductionPipeline()
    try:
        return await pipeline.analyze_disease(disease_name, min_score, max_results)
    finally:
        await pipeline.close()