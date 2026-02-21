
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
    import asyncio
    import logging
    from typing import Dict, List

    logger = logging.getLogger(__name__)


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
        from .ppi_network import PPINetworkScorer, batch_ppi_scores
        from .drug_similarity import DrugSimilarityScorer, build_reference_smiles, batch_similarity_scores
        from .scorer import ProductionScorer

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
        self,
        disease_name: str,
        min_score:    float = 0.2,
        max_results:  int   = 20,
    ) -> Dict:
        start_time = time.time()

        logger.info("=" * 70)
        logger.info(f"🔬 STARTING ANALYSIS: {disease_name}")
        logger.info("=" * 70)

        # STEP 1: Disease data
        logger.info("\n📊 Step 1/5: Fetching disease data from OpenTargets...")
        disease_data = await self.data_fetcher.fetch_disease_data(disease_name)

        if not disease_data:
            logger.error(f"❌ Disease not found: {disease_name}")
            return {
                "success":    False,
                "error":      f"Disease '{disease_name}' not found in OpenTargets database",
                "suggestion": (
                    "Try the full canonical name, e.g. 'Parkinson disease' "
                    "or 'pulmonary arterial hypertension'"
                ),
            }

        logger.info(f"✅ Found: {disease_data['name']}")
        logger.info(f"   Genes:       {len(disease_data['genes'])}")
        logger.info(f"   Pathways:    {len(disease_data['pathways'])}")
        logger.info(f"   Rare disease:{disease_data.get('is_rare', False)}")

        # STEP 2: Approved drugs
        logger.info("\n💊 Step 2/5: Fetching approved drugs from ChEMBL...")
        drugs_data = await self.fetch_approved_drugs(limit=3000)

        # STEP 3: Build graph
        logger.info("\n🕸️  Step 3/5: Building knowledge graph...")
        graph       = self.graph_builder.build_graph(disease_data, drugs_data)
        graph_stats = self.graph_builder.get_graph_stats()
        logger.info(
            f"✅ Graph built: {graph_stats['total_nodes']} nodes, "
            f"{graph_stats['total_edges']} edges"
        )

        # STEP 4: Score with live PubMed
        logger.info("\n🎯 Step 4/5: Scoring drug-disease matches...")
        self.scorer = ProductionScorer(graph)

        drugs_with_targets = [d for d in drugs_data if d.get("targets")]
        logger.info(
            f"   Fetching PubMed co-occurrence scores for "
            f"{len(drugs_with_targets)} drugs..."
        )
        pubmed_tasks = [
            self._fetch_pubmed_score(d["name"], disease_data["name"])
            for d in drugs_with_targets
        ]
        pubmed_scores_list = await asyncio.gather(*pubmed_tasks, return_exceptions=True)
        pubmed_score_map   = {
            d["name"]: (s if isinstance(s, float) else 0.0)
            for d, s in zip(drugs_with_targets, pubmed_scores_list)
        }

        candidates = []
        for drug in drugs_data:
            lit_score = pubmed_score_map.get(drug["name"], 0.0)
            score, evidence = self.scorer.score_drug_disease_match(
                drug["name"],
                disease_data["name"],
                disease_data,
                drug,
                external_literature_score=lit_score,
            )

            if score >= min_score:
                candidates.append(
                    {
                        "name":            drug["name"],
                        "drug_name":       drug["name"],
                        "drug_id":         drug.get("id", ""),
                        "score":           score,
                        "confidence":      evidence["confidence"],
                        "shared_genes":    evidence["shared_genes"],
                        "shared_pathways": evidence["shared_pathways"],
                        "explanation":     evidence["explanation"],
                        "indication":      drug.get("indication", ""),
                        "mechanism":       drug.get("mechanism", ""),
                        "gene_score":      evidence["gene_score"],
                        "pathway_score":   evidence["pathway_score"],
                        "mechanism_score": evidence["mechanism_score"],
                        "literature_score":evidence["literature_score"],
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:max_results]

        logger.info(f"✅ Found {len(candidates)} candidates above threshold {min_score}")

        # STEP 5: Summary
        logger.info("\n📝 Step 5/5: Generating analysis summary...")
        elapsed = time.time() - start_time

        result = {
            "success": True,
            "disease": {
                "name":           disease_data["name"],
                "id":             disease_data.get("id", ""),
                "description":    disease_data.get("description", ""),
                "genes_count":    len(disease_data["genes"]),
                "pathways_count": len(disease_data["pathways"]),
                "is_rare":        disease_data.get("is_rare", False),
                "active_trials":  disease_data.get("active_trials_count", 0),
                "top_genes":      disease_data["genes"][:10],
            },
            "candidates": candidates,
            "metadata": {
                "total_drugs_analyzed": len(drugs_data),
                "candidates_found":     len(candidates),
                "min_score_threshold":  min_score,
                "graph_stats":          graph_stats,
                "analysis_time_seconds":round(elapsed, 2),
                "data_sources": [
                    "OpenTargets Platform (disease-gene associations)",
                    "ChEMBL (approved drugs — full max_phase=4 set)",
                    "DGIdb (drug-gene interactions)",
                    "PubMed E-utilities (literature co-occurrence)",
                    "ClinicalTrials.gov (active trials)",
                ],
            },
        }

        logger.info("=" * 70)
        logger.info("✅ ANALYSIS COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Disease: {disease_data['name']}")
        logger.info(f"Candidates found: {len(candidates)}")
        logger.info(f"Analysis time: {elapsed:.2f}s")

        if candidates:
            logger.info("\n🏆 Top 5 candidates:")
            for i, c in enumerate(candidates[:5], 1):
                logger.info(f"  {i}. {c['drug_name']}")
                logger.info(
                    f"     Score: {c['score']:.3f} ({c['confidence']} confidence)"
                )
                logger.info(f"     Shared genes: {len(c['shared_genes'])}")
                logger.info(f"     Shared pathways: {len(c['shared_pathways'])}")

        return result

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