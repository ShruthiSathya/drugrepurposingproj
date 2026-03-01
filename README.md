# Drug Repurposing Pipeline

A computational pipeline for systematic drug repurposing — given a disease, it ranks approved drugs by how likely they are to treat it, using gene target overlap, pathway proximity, PPI network topology, chemical similarity, and tissue expression as converging lines of evidence.

No hardcoded drug-disease knowledge. All signal comes from live APIs (OpenTargets, ChEMBL, DGIdb, STRING, Reactome, KEGG). Candidates are then filtered for clinical safety, scored for polypharmacology, and put through an in-silico virtual Phase 2 trial.

---

## How it works

The pipeline runs in roughly nine steps:

1. **Fetch disease data** from OpenTargets Platform — associated genes with evidence scores, active trial counts, and rare-disease flags
2. **EFO ontology expansion** — walks the disease ontology tree to pull in genes from related conditions, widening the target set without hardcoding anything
3. **Fetch drugs** from ChEMBL (all max-phase 4 compounds), then enrich targets via DGIdb, ChEMBL mechanisms, and OpenTargets drug queries. Biologics and key small molecules without API coverage get targets from primary pharmacological literature
4. **Score each drug** against the disease using a weighted six-component formula (see below)
5. **Tissue expression** — OpenTargets RNA-seq data confirms targets are actually expressed in the disease-relevant tissue
6. **Safety filter** — drops or flags drugs with known contraindications for the disease class, using both per-drug lookup and mechanism-class rules
7. **Polypharmacology scoring** — rewards drugs that hit resistance genes, cover multiple mechanism classes, and have broad disease-target coverage
8. **Virtual Phase 2 trial** — simulates 200 virtual patients with disease-specific parameters (stroma barrier, mutation heterogeneity, immune desert fraction, etc.) to estimate ORR and prioritize candidates
9. **Baseline comparison** — Jaccard, gene count, cosine (TF-IDF), network proximity, and text mining baselines are scored against the same validation set to quantify how much each pipeline component adds

---

## Scoring

The composite score is a weighted sum of six sub-scores, all in [0, 1]:

| Component | Weight | Source |
|---|---|---|
| Gene overlap | 35% | OpenTargets disease genes × drug targets from DGIdb/ChEMBL |
| Pathway overlap | 25% | Reactome + KEGG, with per-pathway importance weights |
| PPI network proximity | 20% | STRING v12 (combined score >= 400), BFS shortest path |
| Chemical similarity | 10% | Tanimoto/ECFP4 vs known drugs for the disease |
| Mechanism alignment | 5% | Keyword matching between drug mechanism and disease context |
| Literature signal | 5% | PubMed co-occurrence (optional, off by default for speed) |

Tissue expression and polypharmacology are applied as post-hoc adjustments on top of this base score.

---

## Project structure

```
backend/pipeline/
├── production_pipeline.py       # main entry point, orchestrates everything
├── data_fetcher.py              # ChEMBL, DGIdb, OpenTargets data fetching
├── scorer.py                    # weighted scoring formula + sensitivity analysis
├── ppi_network.py               # STRING PPI proximity scoring
├── drug_similarity.py           # Tanimoto/ECFP4 chemical similarity
├── reactome_kegg_integration.py # live Reactome + KEGG pathway queries
├── tissue_expression.py         # OpenTargets RNA expression scoring
├── drug_filter.py               # contraindication database + safety filtering
├── polypharmacology.py          # multi-target resistance bypass scoring
├── insilico_trial.py            # virtual Phase 2 clinical trial simulator
├── calibration.py               # Platt/isotonic score calibration + ECE/AUROC
├── baselines.py                 # Jaccard, network proximity, cosine baselines
├── graph_builder.py             # knowledge graph (disease-gene-drug-pathway)
└── clinical_validator.py        # ClinicalTrials.gov + PubMed + OpenFDA validation
```

---

## Setup

```bash
pip install -r backend/requirements.txt --break-system-packages
```

RDKit is optional but improves chemical similarity accuracy (ECFP4 vs lightweight SMILES hashing). Without it the pipeline falls back gracefully.

```bash
# With RDKit (recommended)
pip install rdkit --break-system-packages
```

---

## Usage

```python
import asyncio
from backend.pipeline.production_pipeline import ProductionPipeline

async def main():
    pipeline = ProductionPipeline()
    results = await pipeline.analyze_disease("pulmonary arterial hypertension")
    
    for c in results["top_candidates"][:10]:
        print(f"{c['drug_name']:25s}  score={c['score']:.3f}  genes={c['shared_genes'][:3]}")
    
    await pipeline.close()

asyncio.run(main())
```

Or if you want more control — fetch data once and score against multiple diseases:

```python
async def main():
    pipeline = ProductionPipeline()
    
    disease_data = await pipeline.data_fetcher.fetch_disease_data("rheumatoid arthritis")
    drugs_data   = await pipeline.fetch_approved_drugs(limit=3000)
    
    candidates = await pipeline.generate_candidates(
        disease_data=disease_data,
        drugs_data=drugs_data,
        min_score=0.1,
        fetch_ppi=True,
        fetch_similarity=True,
        use_tissue=True,
        use_polypharm=True,
    )
    
    await pipeline.close()
```

---

## Baselines

Five baselines are included for comparison in validation runs:

- **Jaccard** — set overlap between drug targets and disease genes, normalized by union size
- **Gene count** — raw shared gene count (log-normalized); shows that set-size bias correction matters
- **Network proximity** — Cheng et al. 2018 "closest" measure using a STRING-derived PPI graph; requires downloading STRING DB (~30 seconds first run, then cached)
- **Cosine similarity** — TF-IDF weighted gene-set vectors; IDF fit on the full drug pool
- **Text mining** — PubMed co-occurrence via NCBI E-utilities (rate-limited to 3 req/sec)

```python
from backend.pipeline.baselines import run_extended_baseline_comparison

metrics = await run_extended_baseline_comparison(
    pipeline=pipeline,
    test_cases=positive_cases,
    neg_cases=negative_cases,
    top_k=100,
    use_network_proximity=False,  # set True if STRING data available
)
```

---

## Score calibration

Raw composite scores are calibrated to probabilities using either Platt scaling (logistic regression) or isotonic regression. Calibration parameters are saved to disk and auto-loaded on subsequent runs.

```python
from backend.pipeline.calibration import ScoreCalibrator

cal = ScoreCalibrator(method="platt")
cal.fit(train_scores, train_labels)

summary = cal.calibration_summary(val_scores, val_labels)
print(f"AUROC: {summary['metrics']['auroc']:.3f}")
print(f"ECE:   {summary['metrics']['ece']:.4f}")
print(f"All passed: {summary['all_passed']}")
```

Pass criteria: AUROC >= 0.70, sensitivity >= 0.65, specificity >= 0.75, ECE < 0.15, Brier < 0.25.

---

## Virtual trials

The in-silico trial simulator generates 200 virtual patients with disease-specific biology (stroma density, immune infiltration, mutation heterogeneity, PK variability) and simulates tumor/disease dynamics over 6 treatment cycles. The network effect blends topology signal (40%) with the composite ML score (60%) — this prevents near-zero ORR for all candidates when drug targets do not directly overlap with the disease gene set.

Disease parameters are pre-defined for 30+ conditions. For any condition without specific parameters the pipeline falls back to generic defaults.

---

## External APIs used

| API | What we fetch | Auth |
|---|---|---|
| OpenTargets Platform | Disease genes, drug-target associations, tissue expression | None |
| ChEMBL REST | Approved drug pool, molecular properties, mechanisms | None |
| DGIdb GraphQL | Drug-gene interactions | None |
| STRING v12 | PPI edges for network proximity | None |
| Reactome Content Service | Pathway annotations per gene | None |
| KEGG REST | Supplementary pathway annotations | None |
| ClinicalTrials.gov v2 | Trial counts for diseases and drug-disease pairs | None |
| NCBI E-utilities | PubMed co-occurrence counts | None (3 req/sec) |
| OpenFDA | Adverse event counts | None |

All requests go through aiohttp with SSL via certifi. Responses are disk-cached to avoid redundant API calls across runs.

---

## Notes

**Drug target coverage:** Around 60-70% of approved drugs have targets from DGIdb or ChEMBL mechanisms. The remaining ~30% go through OpenTargets drug queries, then biologic/small-molecule literature fallbacks. Drugs with zero targets after all four steps receive scores only from chemical similarity and PPI proximity.

**Pathway annotations:** Reactome and KEGG are queried live. The curated fallback map in `data_fetcher.py` covers ~200 genes and is used only when both APIs return nothing for a gene.

**Scoring weights** were selected by grid search over a tuning split. The `weight_grid_search()` function in `scorer.py` enumerates the full search space. Sensitivity analysis (Spearman rank correlation under ±10% weight perturbation) is included in `scorer.py` as `sensitivity_analysis()`.

**Network effects in virtual trials:** The `_calculate_network_effect` method blends PPI topology with the composite ML score. The ^1.5 exponent on `network_effect` steepens differentiation between strong and weak matches and was calibrated against real-world ORR benchmarks (multiple myeloma / lenalidomide as reference point).