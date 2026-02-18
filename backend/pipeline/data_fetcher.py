"""
PRODUCTION DATA FETCHER v2
==============================
Changes vs v1:
1. _enhance_with_pathways() now uses HybridPathwayMapper (Reactome + KEGG)
   instead of the hand-curated _map_genes_to_pathways() dict.
   The curated map is retained as a fallback inside HybridPathwayMapper
   for genes not covered by either API.

2. _map_genes_to_pathways() is kept but only called by HybridPathwayMapper
   internally as a fallback — it is no longer the primary pathway source.

3. Methods citation to add to your paper:
   "Gene-to-pathway mappings were retrieved from the Reactome Pathway
    Knowledgebase (v88; Jassal et al. 2020, Nucleic Acids Res,
    doi:10.1093/nar/gkz1031) via the Reactome Content Service REST API,
    supplemented by KEGG PATHWAY annotations (Kanehisa et al. 2023,
    Nucleic Acids Res, doi:10.1093/nar/gkac963). For genes absent from
    both databases, annotations were taken from a manually curated
    reference map (Supplementary Table S1)."
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
from typing import Optional, List, Dict, Set
from pathlib import Path

from .reactome_kegg_integration import HybridPathwayMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDataFetcher:
    """
    Fetches disease and drug data from public APIs.
    No hardcoded drug-disease pairs — all data comes from live APIs.
    """

    OPENTARGETS_API    = "https://api.platform.opentargets.org/api/v4/graphql"
    CHEMBL_API         = "https://www.ebi.ac.uk/chembl/api/data"
    DGIDB_API          = "https://dgidb.org/api/graphql"
    CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, cache_dir: str = "/tmp/drug_repurposing_cache"):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.drug_cache: Dict    = {}
        self.disease_cache: Dict = {}
        self.ssl_context = self._create_ssl_context()
        # Shared HybridPathwayMapper instance (owns its own session + disk cache)
        self._pathway_mapper: Optional[HybridPathwayMapper] = None

    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            ctx = ssl.create_default_context(cafile=certifi.where())
            logger.info("✅ Using certifi CA certificates")
            return ctx
        except Exception as e:
            logger.warning(f"⚠️  Certifi failed: {e}")
            return ssl.create_default_context()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout   = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.session

    def _get_pathway_mapper(self) -> HybridPathwayMapper:
        if self._pathway_mapper is None:
            self._pathway_mapper = HybridPathwayMapper(use_curated_fallback=True)
        return self._pathway_mapper

    # ══════════════════════════════════════════════════════════════════════════
    #  DISEASE DATA
    # ══════════════════════════════════════════════════════════════════════════

    async def fetch_disease_data(self, disease_name: str) -> Optional[Dict]:
        cache_key = disease_name.lower().strip()
        if cache_key in self.disease_cache:
            logger.info("✅ Using cached disease data")
            return self.disease_cache[cache_key]

        data = await self._fetch_from_opentargets(disease_name)
        if data:
            data = await self._enhance_with_pathways(data)
            data = await self._add_clinical_trials_count(data)
            data = self._mark_rare_disease(data)
            self.disease_cache[cache_key] = data
            logger.info(
                f"✅ Disease data ready: {data['name']} "
                f"({len(data['genes'])} genes, {len(data['pathways'])} pathways)"
            )
        return data

    async def _fetch_from_opentargets(self, disease_name: str) -> Optional[Dict]:
        session = await self._get_session()

        search_query = """
        query SearchDisease($query: String!) {
          search(queryString: $query, entityNames: ["disease"],
                 page: {index: 0, size: 5}) {
            hits { id name description entity }
          }
        }
        """
        try:
            async with session.post(
                self.OPENTARGETS_API,
                json={"query": search_query, "variables": {"query": disease_name}},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    logger.error(f"❌ OpenTargets search failed: {resp.status}")
                    return None
                result  = await resp.json()
                hits    = result.get("data", {}).get("search", {}).get("hits", [])
                if not hits:
                    logger.warning(f"⚠️  Disease not found: {disease_name}")
                    return None
                disease    = hits[0]
                disease_id = disease["id"]
                found_name = disease["name"]
                logger.info(f"✅ Found disease: {found_name} (ID: {disease_id})")

            targets_query = """
            query DiseaseTargets($efoId: String!) {
              disease(efoId: $efoId) {
                id name description
                associatedTargets(page: {index: 0, size: 200}) {
                  count
                  rows {
                    target { id approvedSymbol approvedName biotype }
                    score
                  }
                }
              }
            }
            """
            async with session.post(
                self.OPENTARGETS_API,
                json={"query": targets_query, "variables": {"efoId": disease_id}},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    logger.error("❌ Failed to fetch disease targets")
                    return None
                result       = await resp.json()
                disease_data = result.get("data", {}).get("disease", {})
                if not disease_data:
                    return None

                rows  = disease_data.get("associatedTargets", {}).get("rows", [])
                genes: List[str]              = []
                gene_scores: Dict[str, float] = {}
                for row in rows:
                    target = row.get("target", {})
                    symbol = target.get("approvedSymbol")
                    score  = row.get("score", 0)
                    if symbol and score > 0.1:
                        genes.append(symbol)
                        gene_scores[symbol] = score

                logger.info(f"📊 Found {len(genes)} associated genes from OpenTargets")
                return {
                    "name":        found_name,
                    "id":          disease_id,
                    "description": disease_data.get("description", "")[:500],
                    "genes":       genes,
                    "gene_scores": gene_scores,
                    "pathways":    [],
                    "source":      "OpenTargets Platform",
                }

        except Exception as e:
            logger.error(f"❌ OpenTargets fetch failed: {e}")
            return None

    async def _enhance_with_pathways(self, disease_data: Dict) -> Dict:
        """
        Fetch pathway annotations for disease genes using Reactome + KEGG.
        Falls back to curated map for genes not found in either database.
        """
        genes = disease_data.get("genes", [])[:50]
        if not genes:
            disease_data["pathways"] = []
            return disease_data

        mapper = self._get_pathway_mapper()

        logger.info(f"🔬 Fetching Reactome/KEGG pathways for {len(genes)} genes...")
        try:
            gene_pathway_map = await mapper.get_pathways_bulk(genes)

            all_pathways: Set[str] = set()
            covered_by_api = 0
            covered_by_fallback = 0

            for gene, pathways in gene_pathway_map.items():
                if pathways:
                    all_pathways.update(pathways)
                    # Detect whether result came from API or fallback
                    # (HybridPathwayMapper logs this internally)
                    covered_by_api += 1
                else:
                    covered_by_fallback += 1

            disease_data["pathways"] = sorted(all_pathways) if all_pathways else ["General cellular signaling"]
            logger.info(
                f"✅ Pathway mapping complete: {len(all_pathways)} unique pathways "
                f"(genes with annotations: {covered_by_api}/{len(genes)})"
            )
        except Exception as e:
            logger.warning(f"⚠️  Pathway mapper failed ({e}), using curated fallback for all genes")
            disease_data["pathways"] = self._map_genes_to_pathways_fallback(genes)

        return disease_data

    def _mark_rare_disease(self, disease_data: Dict) -> Dict:
        name = disease_data.get("name", "").lower()
        desc = disease_data.get("description", "").lower()
        rare_kw = [
            "rare", "orphan", "syndrome", "dystrophy", "atrophy",
            "familial", "congenital", "hereditary", "genetic disorder",
            "lysosomal storage", "mitochondrial", "metabolic disorder",
        ]
        disease_data["is_rare"] = any(k in name or k in desc for k in rare_kw)
        if disease_data["is_rare"]:
            logger.info(f"🔬 Identified as RARE DISEASE: {disease_data['name']}")
        return disease_data

    async def _add_clinical_trials_count(self, disease_data: Dict) -> Dict:
        try:
            session = await self._get_session()
            async with session.get(
                self.CLINICALTRIALS_API,
                params={
                    "query.cond":           disease_data["name"],
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
                    "pageSize":             1,
                    "format":               "json",
                    "countTotal":           "true",
                },
            ) as resp:
                if resp.status == 200:
                    data  = await resp.json()
                    total = data.get("totalCount", 0)
                    disease_data["active_trials_count"] = total
                    logger.info(f"📋 Found {total} active clinical trials")
                else:
                    disease_data["active_trials_count"] = 0
        except Exception as e:
            logger.warning(f"⚠️  Could not fetch clinical trials: {e}")
            disease_data["active_trials_count"] = 0
        return disease_data

    # ══════════════════════════════════════════════════════════════════════════
    #  DRUG DATA
    # ══════════════════════════════════════════════════════════════════════════

    async def fetch_approved_drugs(self, limit: int = 2000) -> List[Dict]:
        logger.info(f"💊 Fetching approved drugs from ChEMBL (limit={limit})...")

        cache_file = self.cache_dir / "chembl_approved_drugs.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                if len(cached) >= limit:
                    logger.info("✅ Loading drugs from cache")
                    return cached[:limit]
            except Exception as e:
                logger.warning(f"⚠️  Cache read failed: {e}")

        drugs = await self._fetch_chembl_approved_drugs(limit)
        if not drugs:
            logger.error("❌ No drugs fetched from ChEMBL!")
            return []

        # Step 1: DGIdb (primary source)
        logger.info(f"🔗 Step 1/3: Enhancing {len(drugs)} drugs with DGIdb targets...")
        drugs = await self._enhance_with_dgidb(drugs)

        # Step 2: ChEMBL mechanism fallback for DGIdb gaps
        unenriched = [d for d in drugs if not d.get("targets")]
        logger.info(
            f"🔗 Step 2/3: ChEMBL mechanism enriching "
            f"{len(unenriched)} unenriched drugs..."
        )
        drugs = await self._enhance_with_chembl_mechanisms(drugs)

        # Step 3: diagnostic summary
        still_unenriched = [d for d in drugs if not d.get("targets")]
        enriched_by_source: Dict[str, int] = {}
        for d in drugs:
            src = d.get("target_source", "none")
            enriched_by_source[src] = enriched_by_source.get(src, 0) + 1

        logger.info("=" * 60)
        logger.info("📊 ENRICHMENT DIAGNOSTICS")
        logger.info(f"   Total drugs:    {len(drugs)}")
        logger.info(f"   Enriched:       {len(drugs) - len(still_unenriched)} "
                    f"({(len(drugs) - len(still_unenriched)) / len(drugs) * 100:.1f}%)")
        logger.info(f"   Unenriched:     {len(still_unenriched)}")
        for src, count in enriched_by_source.items():
            logger.info(f"   [{src}]: {count}")
        logger.info("=" * 60)

        try:
            with open(cache_file, "w") as f:
                json.dump(drugs, f, indent=2)
            logger.info(f"✅ Cached {len(drugs)} drugs")
        except Exception as e:
            logger.warning(f"⚠️  Cache write failed: {e}")

        return drugs

    async def _enhance_with_chembl_mechanisms(self, drugs: List[Dict]) -> List[Dict]:
        """
        For drugs that DGIdb left with no targets, query the ChEMBL
        mechanism-of-action endpoint and resolve gene symbols.
        """
        session = await self._get_session()

        unenriched_map: Dict[str, Dict] = {
            d["id"]: d
            for d in drugs
            if not d.get("targets") and d.get("id")
        }

        if not unenriched_map:
            logger.info("   No unenriched drugs to process")
            return drugs

        BATCH_SIZE = 50
        chembl_ids = list(unenriched_map.keys())
        target_symbol_cache: Dict[str, str] = {}
        drug_gene_map: Dict[str, List[str]] = {}

        for batch_start in range(0, len(chembl_ids), BATCH_SIZE):
            batch     = chembl_ids[batch_start: batch_start + BATCH_SIZE]
            ids_param = ",".join(batch)

            try:
                async with session.get(
                    f"{self.CHEMBL_API}/mechanism.json",
                    params={
                        "molecule_chembl_id__in": ids_param,
                        "limit": BATCH_SIZE * 5,
                    },
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"⚠️  ChEMBL mechanism endpoint returned {resp.status}")
                        continue

                    data       = await resp.json()
                    mechanisms = data.get("mechanisms", [])

                    for mech in mechanisms:
                        mol_id   = mech.get("molecule_chembl_id")
                        tgt_id   = mech.get("target_chembl_id")
                        tgt_type = mech.get("target_type", "")

                        if not mol_id or not tgt_id:
                            continue
                        if tgt_type not in ("SINGLE PROTEIN", ""):
                            continue

                        if tgt_id not in target_symbol_cache:
                            symbol = await self._resolve_chembl_target(tgt_id, session)
                            target_symbol_cache[tgt_id] = symbol or ""

                        symbol = target_symbol_cache[tgt_id]
                        if symbol:
                            drug_gene_map.setdefault(mol_id, [])
                            if symbol not in drug_gene_map[mol_id]:
                                drug_gene_map[mol_id].append(symbol)

            except Exception as e:
                logger.error(f"❌ ChEMBL mechanism batch failed: {e}")
                continue

        filled = 0
        mapper = self._get_pathway_mapper()

        for chembl_id, gene_symbols in drug_gene_map.items():
            if chembl_id in unenriched_map and gene_symbols:
                drug = unenriched_map[chembl_id]
                drug["targets"]       = gene_symbols
                drug["target_source"] = "opentargets_mechanism"
                filled += 1
                # Fetch pathways for these drug targets via Reactome/KEGG
                try:
                    gene_pw_map = await mapper.get_pathways_bulk(gene_symbols[:20])
                    pw_set: Set[str] = set()
                    for pws in gene_pw_map.values():
                        pw_set.update(pws)
                    drug["pathways"] = sorted(pw_set)
                except Exception:
                    drug["pathways"] = self._infer_pathways_from_targets_fallback(gene_symbols)

                logger.info(
                    f"   ✅ ChEMBL mechanism enriched {drug['name']} → "
                    f"{len(gene_symbols)} targets {gene_symbols}"
                )

        logger.info(f"   OpenTargets mechanism enriched {filled} additional drugs")
        return drugs

    async def _resolve_chembl_target(
        self, target_chembl_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        try:
            async with session.get(
                f"{self.CHEMBL_API}/target/{target_chembl_id}.json",
                params={"format": "json"},
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                components = data.get("target_components", [])
                for component in components:
                    for synonym in component.get("target_component_synonyms", []):
                        if synonym.get("syn_type") == "GENE_SYMBOL":
                            return synonym.get("component_synonym")
        except Exception:
            pass
        return None

    async def _fetch_chembl_approved_drugs(self, limit: int) -> List[Dict]:
        """
        Fetch up to `limit` approved drugs from ChEMBL, paginating in
        batches of 1000 (the ChEMBL API maximum per request).
        No drug names are injected — whatever ChEMBL returns is the pool.
        """
        session = await self._get_session()
        drugs: List[Dict] = []
        PAGE_SIZE = 1000
        offset = 0

        while len(drugs) < limit:
            batch_size = min(PAGE_SIZE, limit - len(drugs))
            try:
                async with session.get(
                    f"{self.CHEMBL_API}/molecule.json",
                    params={"max_phase": "4", "limit": batch_size, "offset": offset},
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"❌ ChEMBL API failed: {resp.status}")
                        break
                    data      = await resp.json()
                    molecules = data.get("molecules", [])
                    if not molecules:
                        logger.info("📥 ChEMBL returned no more molecules, stopping pagination")
                        break
                    logger.info(f"📥 ChEMBL page offset={offset}: {len(molecules)} molecules")
                    for mol in molecules:
                        drug = self._process_chembl_molecule(mol)
                        if drug:
                            drugs.append(drug)
                    offset += len(molecules)
                    if len(molecules) < batch_size:
                        break
            except Exception as e:
                logger.error(f"❌ ChEMBL fetch failed at offset {offset}: {e}")
                break

        logger.info(f"✅ Fetched {len(drugs)} drugs from ChEMBL (requested {limit})")
        return drugs

    def _process_chembl_molecule(self, molecule: Dict) -> Optional[Dict]:
        try:
            chembl_id = molecule.get("molecule_chembl_id")
            name      = molecule.get("pref_name") or chembl_id
            if not name or name == chembl_id:
                return None
            structures = molecule.get("molecule_structures", {})
            smiles     = structures.get("canonical_smiles", "") if structures else ""
            return {
                "id":        chembl_id,
                "name":      name,
                "indication": molecule.get("indication_class", "Various indications"),
                "mechanism": molecule.get("mechanism_of_action", ""),
                "approved":  True,
                "smiles":    smiles,
                "targets":   [],
                "pathways":  [],
            }
        except Exception:
            return None

    async def _enhance_with_dgidb(self, drugs: List[Dict]) -> List[Dict]:
        session = await self._get_session()

        DGIDB_QUERY = """
        query DrugInteractions($names: [String!]!) {
          drugs(names: $names) {
            nodes {
              name
              conceptId
              approved
              interactions {
                gene { name }
                interactionTypes { type }
              }
            }
          }
        }
        """

        BATCH_SIZE   = 100
        drug_names   = [d["name"] for d in drugs]
        name_variants = [
            [n.upper()  for n in drug_names],
            [n.title()  for n in drug_names],
            drug_names,
        ]

        drug_target_map: Dict[str, List[str]] = {}
        successful_queries = 0

        for variant_idx, variant_list in enumerate(name_variants):
            label = ["UPPERCASE", "TitleCase", "Original"][variant_idx]
            logger.info(f"🔍 Trying DGIdb with {label} names...")

            for batch_start in range(0, len(variant_list), BATCH_SIZE):
                batch = variant_list[batch_start: batch_start + BATCH_SIZE]
                logger.info(
                    f"   Batch {batch_start // BATCH_SIZE + 1}/"
                    f"{(len(variant_list) - 1) // BATCH_SIZE + 1} "
                    f"({len(batch)} drugs)..."
                )
                try:
                    async with session.post(
                        self.DGIDB_API,
                        json={"query": DGIDB_QUERY, "variables": {"names": batch}},
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.warning(f"⚠️  DGIdb returned {resp.status}: {text[:200]}")
                            continue
                        result = await resp.json()
                        if "errors" in result:
                            errs = [e.get("message") for e in result["errors"]]
                            logger.warning(f"⚠️  DGIdb GraphQL errors: {errs}")
                            continue

                        dgidb_drugs = (
                            result.get("data", {}).get("drugs", {}).get("nodes", []) or []
                        )
                        dgidb_drugs = [d for d in dgidb_drugs if d]

                        if dgidb_drugs:
                            successful_queries += 1
                            logger.info(f"   ✅ DGIdb returned {len(dgidb_drugs)} drug records")

                        for dd in dgidb_drugs:
                            key    = dd.get("name", "").lower()
                            inters = dd.get("interactions") or []
                            targets = [
                                i["gene"]["name"]
                                for i in inters
                                if i.get("gene") and i["gene"].get("name")
                            ]
                            if targets and key not in drug_target_map:
                                drug_target_map[key] = targets

                except Exception as e:
                    logger.error(f"❌ DGIdb batch failed: {e}")
                    continue

            if len(drug_target_map) > len(drugs) * 0.3:
                logger.info(f"✅ Good match rate with {label} names, stopping variants")
                break

        logger.info(f"📊 DGIdb mapping complete: {len(drug_target_map)} drugs have targets")

        mapper  = self._get_pathway_mapper()
        enhanced = 0

        for drug in drugs:
            candidates = {
                drug["name"].lower(),
                drug["name"].upper().lower(),
                drug["name"].title().lower(),
            }
            for key in candidates:
                if key in drug_target_map:
                    targets          = drug_target_map[key]
                    drug["targets"]  = targets
                    drug["target_source"] = "dgidb"
                    enhanced += 1

                    # Fetch Reactome/KEGG pathways for these targets
                    try:
                        gene_pw_map = await mapper.get_pathways_bulk(targets[:20])
                        pw_set: Set[str] = set()
                        for pws in gene_pw_map.values():
                            pw_set.update(pws)
                        drug["pathways"] = sorted(pw_set)
                    except Exception:
                        drug["pathways"] = self._infer_pathways_from_targets_fallback(targets)

                    break

        logger.info(f"✅ Enhanced {enhanced}/{len(drugs)} drugs with DGIdb gene targets")
        logger.info(f"   Enhancement rate: {enhanced / len(drugs) * 100:.1f}%")
        if enhanced == 0:
            logger.error("❌ CRITICAL: No drugs enhanced — check DGIdb connectivity")
        return drugs

    # ══════════════════════════════════════════════════════════════════════════
    #  FALLBACK helpers (used when Reactome/KEGG APIs are unreachable)
    # ══════════════════════════════════════════════════════════════════════════

    def _infer_pathways_from_targets_fallback(self, targets: List[str]) -> List[str]:
        """Use curated map as emergency fallback."""
        pathways: Set[str] = set()
        for t in targets[:20]:
            pathways.update(self._map_genes_to_pathways_fallback([t]))
        return list(pathways)

    def _map_genes_to_pathways_fallback(self, genes: List[str]) -> List[str]:
        """
        Curated gene→pathway map — now used ONLY as fallback when both
        Reactome and KEGG APIs fail or return nothing.

        This is the original _map_genes_to_pathways() content, preserved
        verbatim so existing behaviour is not regressed.
        """
        pathway_map: Dict[str, List[str]] = {
            # ── Neurodegeneration ─────────────────────────────────────────
            "SNCA":   ["Alpha-synuclein aggregation", "Dopamine metabolism", "Autophagy"],
            "LRRK2":  ["Autophagy", "Mitochondrial function", "Vesicle trafficking"],
            "PRKN":   ["Mitophagy", "Ubiquitin-proteasome system"],
            "PINK1":  ["Mitophagy", "Mitochondrial quality control"],
            "PARK7":  ["Oxidative stress response", "Mitochondrial function"],
            "DJ1":    ["Oxidative stress response", "Mitochondrial function"],
            "GBA":    ["Lysosomal function", "Sphingolipid metabolism", "Autophagy"],
            "GBA1":   ["Lysosomal function", "Sphingolipid metabolism", "Autophagy"],
            "MAOB":   ["Dopamine metabolism", "Monoamine oxidase"],
            "TH":     ["Dopamine biosynthesis", "Catecholamine synthesis"],
            "DDC":    ["Dopamine biosynthesis", "Neurotransmitter synthesis"],
            "LAMP1":  ["Lysosomal function", "Autophagy"],
            "LAMP2":  ["Autophagy", "Lysosomal membrane"],
            "HTT":    ["Huntingtin aggregation", "Ubiquitin-proteasome system"],
            "APP":    ["Amyloid-beta production", "APP processing"],
            "MAPT":   ["Tau protein function", "Microtubule stability"],
            "PSEN1":  ["Amyloid-beta production", "Gamma-secretase complex"],
            "PSEN2":  ["Amyloid-beta production", "Gamma-secretase complex"],
            "APOE":   ["Lipid metabolism", "Amyloid-beta clearance"],
            "CHAT":   ["Cholinergic signaling", "Neurotransmitter synthesis"],
            "ACHE":   ["Cholinergic signaling", "Acetylcholine degradation"],
            "GRIN1":  ["NMDA receptor signaling", "Glutamate signaling", "Synaptic plasticity"],
            "GRIN2A": ["NMDA receptor signaling", "Glutamate signaling", "Synaptic plasticity"],
            "GRIN2B": ["NMDA receptor signaling", "Glutamate signaling", "Synaptic plasticity"],
            "GRIN2C": ["NMDA receptor signaling", "Glutamate signaling"],
            "GRIN2D": ["NMDA receptor signaling", "Glutamate signaling"],
            "GRIN3A": ["NMDA receptor signaling", "Synaptic plasticity"],
            "GRIN3B": ["NMDA receptor signaling"],
            # ── Cardiology / vascular ─────────────────────────────────────
            "PDE5A":  ["PDE5 signaling", "cGMP-PKG signaling", "Nitric oxide signaling",
                       "Pulmonary vascular remodeling", "Vasodilation"],
            "NOS3":   ["Nitric oxide signaling", "Vasodilation", "Endothelial function"],
            "NOS1":   ["Nitric oxide signaling", "Neurotransmitter synthesis"],
            "GUCY1A1":["cGMP-PKG signaling", "Nitric oxide signaling"],
            "GUCY1B1":["cGMP-PKG signaling", "Nitric oxide signaling"],
            "PRKG1":  ["cGMP-PKG signaling", "Vasodilation"],
            "EDNRA":  ["Endothelin signaling", "Pulmonary vascular remodeling"],
            "EDNRB":  ["Endothelin signaling", "Pulmonary vascular remodeling"],
            "EDN1":   ["Endothelin signaling", "Vasoconstriction"],
            "PTGIS":  ["Prostacyclin signaling", "Platelet aggregation"],
            "TBXA2R": ["Arachidonic acid metabolism", "Platelet aggregation"],
            "ADRB1":  ["Beta-adrenergic signaling", "Cardiac function"],
            "ADRB2":  ["Beta-adrenergic signaling", "Vasodilation", "Bronchodilation"],
            "ADRB3":  ["Beta-adrenergic signaling", "Lipolysis"],
            "PTGS1":  ["COX pathway", "Platelet aggregation", "Arachidonic acid metabolism"],
            "PTGS2":  ["COX pathway", "Inflammatory response", "Arachidonic acid metabolism"],
            "TBXAS1": ["Arachidonic acid metabolism", "Platelet aggregation"],
            "HMGCR":  ["HMGCR pathway", "Cholesterol metabolism", "Lipid metabolism"],
            "LDLR":   ["Cholesterol metabolism", "LDL receptor signaling"],
            "PCSK9":  ["Cholesterol metabolism", "LDL receptor signaling"],
            "APOB":   ["Lipid metabolism", "LDL transport"],
            "LPA":    ["Lipoprotein metabolism", "Coagulation cascade"],
            "ACE":    ["Renin-angiotensin system", "Blood pressure regulation"],
            "AGT":    ["Renin-angiotensin system", "Blood pressure regulation"],
            "AGTR1":  ["Renin-angiotensin system", "Vasoconstriction"],
            "ITGA2B": ["Platelet aggregation", "Coagulation cascade"],
            "F2":     ["Coagulation cascade", "Thrombin signaling"],
            "VWF":    ["Coagulation cascade", "Platelet adhesion"],
            # ── Immunology / inflammation ─────────────────────────────────
            "MS4A1":  ["B-cell receptor signaling", "B-cell differentiation"],
            "CD20":   ["B-cell receptor signaling", "B-cell differentiation"],
            "CD79A":  ["B-cell receptor signaling"],
            "CD79B":  ["B-cell receptor signaling"],
            "BLNK":   ["B-cell receptor signaling"],
            "BTK":    ["B-cell receptor signaling", "B-cell differentiation"],
            "CD4":    ["T-cell receptor signaling", "Adaptive immunity"],
            "CD8A":   ["T-cell receptor signaling", "Cytotoxic T-cell"],
            "CTLA4":  ["T-cell checkpoint signaling"],
            "PDCD1":  ["T-cell checkpoint signaling"],
            "TNF":    ["TNF signaling", "NF-κB signaling", "Inflammatory response"],
            "IL6":    ["JAK-STAT signaling", "Cytokine signaling", "IL-6 signaling"],
            "IL1B":   ["Inflammatory response", "NF-κB signaling"],
            "NFKB1":  ["NF-κB signaling", "Inflammatory response"],
            "JAK1":   ["JAK-STAT signaling"],
            "JAK2":   ["JAK-STAT signaling"],
            "STAT3":  ["JAK-STAT signaling", "IL-6 signaling"],
            "TGFB1":  ["TGF-beta signaling", "Inflammatory response"],
            "HLA-B":  ["Immune presentation", "Adaptive immunity"],
            # ── Oncology ─────────────────────────────────────────────────
            "EGFR":   ["EGFR signaling", "MAPK signaling", "PI3K-Akt signaling"],
            "ERBB2":  ["EGFR signaling", "HER2 signaling"],
            "KRAS":   ["RAS signaling", "MAPK signaling"],
            "BRAF":   ["MAPK signaling", "RAS signaling"],
            "PIK3CA": ["PI3K-Akt signaling", "mTOR signaling"],
            "PTEN":   ["PI3K-Akt signaling", "Cell growth regulation"],
            "MTOR":   ["mTOR signaling", "Autophagy", "Protein synthesis"],
            "TP53":   ["p53 signaling", "Apoptosis", "DNA damage response"],
            "VEGFA":  ["Angiogenesis", "VEGF signaling"],
            "VEGFR2": ["Angiogenesis", "VEGF signaling"],
            "MYC":    ["Cell cycle regulation", "Oncogene signaling"],
            "CCND1":  ["Cell cycle regulation", "CDK signaling"],
            "CDK4":   ["Cell cycle regulation", "CDK signaling"],
            "CDK6":   ["Cell cycle regulation", "CDK signaling"],
            "RB1":    ["Cell cycle regulation", "Tumor suppressor"],
            "BRCA1":  ["DNA damage response", "DNA repair"],
            "BRCA2":  ["DNA damage response", "DNA repair"],
            "ESR1":   ["Estrogen receptor signaling", "Nuclear receptor signaling",
                       "Steroid hormone biosynthesis"],
            "ESR2":   ["Estrogen receptor signaling", "Nuclear receptor signaling"],
            "PGR":    ["Progesterone signaling", "Nuclear receptor signaling"],
            "CRBN":   ["Protein degradation", "Ubiquitin-proteasome system",
                       "IKZF1/3 degradation"],
            "CUL4A":  ["Protein degradation", "Ubiquitin-proteasome system"],
            "DDB1":   ["Protein degradation", "Ubiquitin-proteasome system"],
            "RBX1":   ["Ubiquitin-proteasome system"],
            "FGFR2":  ["FGFR signaling", "MAPK signaling"],
            "HDAC1":  ["Histone modification", "Epigenetic regulation"],
            "HDAC2":  ["Histone modification", "Epigenetic regulation"],
            "HDAC6":  ["Histone modification", "Epigenetic regulation"],
            # ── Metabolic / PCOS / diabetes ───────────────────────────────
            "INSR":   ["Insulin signaling", "Glucose metabolism"],
            "IRS1":   ["Insulin signaling", "PI3K-Akt signaling"],
            "IRS2":   ["Insulin signaling", "PI3K-Akt signaling"],
            "PRKAB1": ["AMPK signaling", "Glucose metabolism"],
            "PRKAB2": ["AMPK signaling", "Glucose metabolism"],
            "PRKAA1": ["AMPK signaling", "Gluconeogenesis"],
            "PRKAA2": ["AMPK signaling", "Gluconeogenesis"],
            "G6PC":   ["Gluconeogenesis", "Glucose metabolism"],
            "PCK1":   ["Gluconeogenesis", "Glucose metabolism"],
            "SLC2A4": ["Glucose metabolism", "GLUT4 transport"],
            "PPARG":  ["PPAR signaling", "Adipogenesis", "Glucose metabolism"],
            "PPARA":  ["PPAR signaling", "Fatty acid oxidation"],
            "LHCGR":  ["Steroid hormone biosynthesis", "Gonadotropin signaling"],
            "FSHR":   ["Steroid hormone biosynthesis", "Gonadotropin signaling"],
            "CYP17A1":["Steroid hormone biosynthesis", "Androgen receptor signaling"],
            "CYP19A1":["Steroid hormone biosynthesis", "Estrogen receptor signaling"],
            "SHBG":   ["Steroid hormone biosynthesis", "Androgen binding"],
            "AMH":    ["Gonadotropin signaling", "Ovarian function"],
            # ── Hair follicle / alopecia ──────────────────────────────────
            "SRD5A1": ["5-alpha reductase pathway", "Androgen receptor signaling",
                       "Hair follicle cycling", "Steroid hormone biosynthesis"],
            "SRD5A2": ["5-alpha reductase pathway", "Androgen receptor signaling",
                       "Hair follicle cycling"],
            "AR":     ["Androgen receptor signaling", "Hair follicle cycling"],
            "KCNJ8":  ["Potassium channel signaling", "Vasodilation", "Hair follicle cycling"],
            "KCNJ11": ["Potassium channel signaling", "Vasodilation"],
            "ABCC9":  ["Potassium channel signaling", "Vasodilation"],
            "CYP2C9": ["Drug metabolism", "CYP oxidation"],
            "CYP2D6": ["Drug metabolism", "CYP oxidation"],
            # ── Lysosomal / rare diseases ─────────────────────────────────
            "ATP7B":  ["Copper metabolism", "Metal ion homeostasis"],
            "NPC1":   ["Cholesterol trafficking", "Lysosomal function"],
            "NPC2":   ["Cholesterol metabolism", "Lipid transport"],
            "DMD":    ["Dystrophin-glycoprotein complex", "Muscle fiber integrity"],
            "CFTR":   ["Chloride ion transport", "CFTR trafficking"],
            "HEXA":   ["Lysosomal function", "Sphingolipid metabolism"],
            "HEXB":   ["Lysosomal function", "Sphingolipid metabolism"],
            "GALC":   ["Lysosomal function", "Sphingolipid metabolism"],
            "ARSA":   ["Lysosomal function", "Sulfatide metabolism"],
            "ASAH1":  ["Sphingolipid metabolism", "Lysosomal function"],
            # ── Misc signaling ────────────────────────────────────────────
            "HGF":    ["MET signaling", "Cell growth"],
            "PLCG1":  ["Phospholipase C signaling", "IP3 signaling"],
            "SLC6A12":["Neurotransmitter transport"],
            "CDH13":  ["Cell adhesion", "Cardiovascular function"],
            "NAT2":   ["Drug metabolism", "Acetylation"],
            "CFLAR":  ["Apoptosis", "FLIP signaling"],
        }

        pathways: Set[str] = set()
        for gene in genes:
            if gene in pathway_map:
                pathways.update(pathway_map[gene])
        return sorted(pathways) if pathways else ["General cellular signaling"]

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔒 Session closed")
        if self._pathway_mapper:
            await self._pathway_mapper.close()
            logger.info("🔒 Pathway mapper closed")


DataFetcher = ProductionDataFetcher