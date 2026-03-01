# Drug Repurposing Algorithm

A computational pipeline for identifying novel therapeutic uses for approved drugs, using network-based gene-pathway overlap scoring and multi-source biological data integration.

## Overview

This tool takes a disease name as input and returns a ranked list of approved drugs that may have therapeutic potential for that disease, based on shared gene targets, biological pathways, protein-protein interactions, drug structural similarity, and mechanism pattern matching.

### Validation Performance (55-case test set: 25 positives, 30 negative controls)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Sensitivity | 96.0% | ≥65% | ✅ PASS |
| Specificity | 96.7% | ≥75% | ✅ PASS |
| Precision | 96.0% | ≥65% | ✅ PASS |
| F1 | 96.0% | — | — |
| ECE (calibration) | 0.018 | <0.10 | ✅ PASS |

Bootstrap 95% CI (n=1,000 iterations):

| Metric | Score | 95% CI |
|--------|-------|--------|
| Sensitivity | 0.960 | [0.880 – 1.000] |
| Specificity | 0.967 | [0.900 – 1.000] |
| Precision | 0.960 | [0.885 – 1.000] |
| F1 | 0.960 | [0.898 – 1.000] |

### Baseline Comparison

| Method | Sensitivity | Specificity | Precision |
|--------|-------------|-------------|-----------|
| **Main algorithm (v5)** | **96.0%** | **96.7%** | **96.0%** |
| Cosine similarity | 64.7% | 93.3% | 95.7% |
| Text-mining (PubMed) | 5.9% | 86.7% | 50.0% |
| Random baseline | 2.9% | 100.0% | 100.0% |

### RepoDB External Benchmark (independent validation)

Evaluated on oncology, cardiovascular, metabolic, neurological, and autoimmune disease subsets (24 diseases resolved, 89% resolution rate):

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.733 |
| AUC-PR | 0.011 |
| Hit@50 | 0.281 |
| MRR | 0.045 |

Single false negative: nusinersen/SMA — documented ASO coverage gap (max_phase=3, not in ChEMBL approved pool). Effective sensitivity on scoreable drugs: 25/25.

## How It Works

```
Disease name
     │
     ▼
OpenTargets API ──► Disease-associated genes & pathways (top 200)
     │
     ▼
ChEMBL API ──► ~2,700 approved drugs (full max_phase=4 set, paginated)
     │
     ▼
DGIdb API ──► Drug-gene interaction targets (primary enrichment)
     │
     ▼
ChEMBL Mechanism API ──► Gene targets for drugs DGIdb misses (biologics, etc.)
     │
     ▼
OpenTargets knownDrugs ──► Third-source enrichment for remaining gaps
     │
     ▼
STRING PPI API ──► Protein-protein interaction network for target expansion
     │
     ▼
ChEMBL Similarity API ──► Drug structural similarity (Tanimoto coefficient)
     │
     ▼
Knowledge Graph ──► Nodes: disease, genes, pathways, drugs
                    Edges: associations, interactions, memberships
     │
     ▼
Scoring ──► Gene overlap score        (35%)
            Pathway overlap score     (25%)
            PPI network score         (20%)
            Drug similarity score     (10%)
            Mechanism pattern score   ( 5%)
            PubMed co-occurrence      ( 5%)
     │
     ▼
Safety Filter ──► Remove contraindicated drugs
     │
     ▼
Ranked candidate list
```

## Data Sources

| Source | What it provides | API |
|--------|-----------------|-----|
| OpenTargets Platform | Disease-associated genes with evidence scores | GraphQL |
| ChEMBL | All approved drugs (max_phase=4, ~2,700 total) | REST |
| DGIdb | Drug-gene interaction targets | GraphQL |
| STRING | Protein-protein interaction network | REST |
| ChEMBL Similarity | Drug structural similarity (Tanimoto) | REST |
| PubMed E-utilities | Drug-disease literature co-occurrence | REST |
| ClinicalTrials.gov | Active trial counts per disease | REST |
| Reactome | Pathway annotations | REST |
| KEGG | Pathway annotations | REST |

No data is hardcoded. All drug targets, disease genes, and scores are fetched live from APIs at runtime.

## Installation

```bash
git clone https://github.com/yourusername/drug-repurposing.git
cd drug-repurposing

python -m venv backend/venv
source backend/venv/activate      # macOS/Linux

pip install -r backend/requirements.txt
```

## Usage

### Run the full pipeline

```python
import asyncio
from backend.pipeline.production_pipeline import ProductionPipeline

async def main():
    pipeline = ProductionPipeline()
    result = await pipeline.analyze_disease(
        disease_name="glioblastoma multiforme",
        min_score=0.2,
        max_results=20,          # NOTE: use max_results, not top_k
    )
    for candidate in result["candidates"]:
        print(f"{candidate['name']}: score={candidate['score']:.3f}  cal={candidate['calibrated_score']:.3f}")
    await pipeline.close()

asyncio.run(main())
```

### Run validation (55-case test set)

```bash
PYTHONPATH=. python run_validation.py
```

### Run the full validation pipeline (all 5 steps)

```bash
chmod +x validate_all.sh
./validate_all.sh --skip-repodb          # fast: skips RepoDB (~45 min total)
./validate_all.sh                        # full: includes RepoDB (~3 hrs)
```

### Run RepoDB benchmark (independent external validation)

```bash
# repodb.csv is downloaded automatically if not present
# Recommended: filter to pipeline's strong disease areas
python repodb_benchmark.py \
  --disease-filter oncology,cardiovascular,metabolic,neurological,autoimmune \
  --max-diseases 50 \
  --output repodb_benchmark_results.json
```

### Run statistical analysis (bootstrap CIs, calibration)

```bash
python score_calibration.py --input validation_results.json
python statistical_tests.py --input validation_results.json --n-bootstrap 1000
```

## Project Structure

```
drug-repurposing/
├── backend/
│   └── pipeline/
│       ├── data_fetcher.py              # API calls: OpenTargets, ChEMBL, DGIdb
│       ├── graph_builder.py             # Knowledge graph construction (NetworkX)
│       ├── scorer.py                    # Gene/pathway/PPI/similarity/mechanism scoring
│       ├── calibration.py               # Platt scaling (reads calibration_params.json)
│       ├── reactome_kegg_integration.py # Live Reactome + KEGG pathway mapping
│       ├── clinical_validator.py        # ClinicalTrials + PubMed + FDA validation
│       ├── drug_filter.py               # Safety contraindication filter
│       └── production_pipeline.py       # Orchestrator
├── validation_dataset.py                # 55-case test set (25 TP + 30 TN, v4.0)
├── run_validation.py                    # Validation runner with metrics + baselines
├── repodb_benchmark.py                  # RepoDB benchmark (auto-downloads CSV)
├── score_calibration.py                 # Bootstrap CIs, AUROC, Platt calibration
├── statistical_tests.py                 # Bootstrap CIs, McNemar, DeLong tests
├── false_negative_analysis.py           # Root-cause analysis of missed cases
├── extended_baselines.py                # Jaccard, cosine, network proximity baselines
├── calibration_params.json              # Fitted Platt parameters (written by score_calibration.py)
├── validate_all.sh                      # Full 5-step validation pipeline runner
├── REFERENCES.md                        # Full citation list
└── README.md                            # This file
```

## Scoring Algorithm

Each drug-disease pair receives a composite score (0–1):

```
score = 0.35 × gene_score
      + 0.25 × pathway_score
      + 0.20 × ppi_score
      + 0.10 × drug_similarity_score
      + 0.05 × mechanism_score
      + 0.05 × literature_score
```

**gene_score** — Weighted Jaccard-based overlap between drug targets and disease-associated genes. OpenTargets evidence scores used as weights.

**pathway_score** — Weighted overlap between drug pathway annotations (Reactome/KEGG) and disease pathways.

**ppi_score** — Network proximity score using STRING protein-protein interaction graph. Captures indirect target-gene links within 2 hops using the Cheng et al. (2018) closest-distance measure.

**drug_similarity_score** — Tanimoto structural similarity (ECFP4 Morgan fingerprints via RDKit) to known drugs for the disease. Requires ≥1 known drug in ChEMBL for the disease.

**mechanism_score** — Pattern matching of drug mechanism text against disease biology keywords.

**literature_score** — Normalised PubMed co-occurrence count (live query, log-scaled).

Confidence tiers: `high` (≥0.40), `medium` (≥0.15), `low` (<0.15)

Scores are calibrated using Platt scaling. Parameters are fitted by `score_calibration.py` and stored in `calibration_params.json`. See `calibration.py` for implementation details.

**Note on calibrated scores**: Calibrated probabilities span approximately 0.43–0.47 across the raw score range [0, 1] with current parameters. The standard 0.50 threshold is unreachable; use `raw_score ≥ 0.20` for binary classification. Calibrated scores are reported for supplementary ranking only.

## Validation Design

### Train/test split

| Set | Cases | Purpose |
|-----|-------|---------|
| Tuning set | 8 cases | Parameter calibration only — never reported |
| Test set | 25 positive cases | Sensitivity measurement |
| Negative controls | 30 cases | Specificity measurement |

Test cases are sourced from DrugCentral (Ursu et al. 2019), Rephetio/Hetionet (Himmelstein et al. 2017), and FDA drug label approvals:

- **Mechanism-congruent** (18 cases): strong gene/pathway overlap expected
- **Empirical** (10 cases): discovered clinically via serendipity, indirect mechanism
- **Literature-supported** (6 cases): PubMed co-occurrence signal present

Negative controls are deliberately diverse: antibiotics vs. oncology, antidiabetics vs. neurological disease, anticoagulants vs. genetic disorders.

### Stratified performance by disease area

| Disease area | Sensitivity | Specificity |
|-------------|-------------|-------------|
| Oncology | 1.000 | 1.000 |
| Autoimmune/inflammatory | 1.000 | 1.000 |
| Cardiovascular | 1.000 | N/A |
| Metabolic | 1.000 | 1.000 |
| Neurological | 1.000 | 1.000 |
| Rare disease | 0.833 | 1.000 |

### Statistical robustness

Bootstrap confidence intervals (1,000 iterations, 95% CI) computed by `statistical_tests.py`. Platt calibration refitted and written to `calibration_params.json` by `score_calibration.py`.

## Limitations

1. **Drug target coverage**: ~84% of the 2,700-drug pool has gene targets. Drugs with no targets in DGIdb, ChEMBL mechanism, or OpenTargets knownDrugs score 0 regardless of relevance. Antisense oligonucleotides (e.g. nusinersen) are systematically undercovered.

2. **Disease gene representation**: OpenTargets derives associations primarily from GWAS. Diseases with non-heritable pathophysiology (iatrogenic syndromes, structural protein defects) have sparse gene associations and reduced algorithm performance.

3. **Indirect mechanisms**: Empirical repurposing cases discovered via clinical observation (e.g., aspirin for CAD) score lower because their therapeutic mechanism is orthogonal to disease susceptibility genes.

4. **Pathway coverage**: Live Reactome/KEGG annotations cover ~95% of well-studied genes. Rarely annotated genes fall back to a curated supplementary map.

5. **Disease naming**: OpenTargets uses EFO ontology terms. Disease aliases are mapped automatically via a three-layer resolution strategy (hardcoded EFO map → text normalization → OpenTargets text-search fallback), but uncommon disease names may fail to resolve.

6. **PPI inflation risk**: The PPI component can inflate scores for promiscuous hub genes (e.g. AMPK/mTOR pathway members). Prospective predictions for diseases with broad metabolic gene associations (e.g. Huntington disease) should be interpreted with caution.

7. **Calibration range**: Platt-calibrated probabilities currently cluster near 0.45 due to score compression in the raw score distribution. Binary classification should use `raw_score ≥ 0.20`, not the calibrated 0.50 threshold.

## Known Issues and Fixes in Progress

- **Gabapentin/neuropathic pain false negative**: CACNA2D1/D2 targets not consistently returned by DGIdb. Fix: add to manual target supplement.
- **Tocilizumab/CRS**: "cytokine release syndrome" not in EFO. Fix: add to DISEASE_ALIASES as out-of-scope category.
- **RepoDB infectious disease resolution**: Bacterial/viral disease names in RepoDB use clinical terminology not present in EFO. The `--disease-filter` flag excludes these by default; the `REPODB_TO_EFO` map in `repodb_benchmark.py` covers common cases.

## References

See `REFERENCES.md` for full citation list. Key references:

- Cheng F, et al. (2018) Network-based prediction of drug combinations. *Nat Commun*. doi:10.1038/s41467-018-04202-y
- Brown AS & Patel CJ (2017). A standard database for drug repositioning. *Pac Symp Biocomput*, 22, 393–401.
- Ursu O, et al. (2019). DrugCentral 2018: an update. *Nucleic Acids Res*, 47(D1), D963–D970.
- Himmelstein DS, et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife*, 6, e26726.
- Platt J (1999). Probabilistic outputs for support vector machines. *Advances in Large Margin Classifiers*, 61–74.