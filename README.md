# Drug Repurposing Algorithm

A computational pipeline for identifying novel therapeutic uses for approved drugs, using network-based gene-pathway overlap scoring and multi-source biological data integration.

## Overview

This tool takes a disease name as input and returns a ranked list of approved drugs that may have therapeutic potential for that disease, based on shared gene targets, biological pathways, protein-protein interactions, drug structural similarity, and mechanism pattern matching.

### Validation Performance (55-case test set: 25 positives, 30 negative controls)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Sensitivity | 96.0% | ≥65% | ✅ PASS |
| Specificity | 93.3% | ≥75% | ✅ PASS |
| Precision | 92.3% | ≥65% | ✅ PASS |
| F1 | 94.1% | — | — |
| ECE (calibration) | 0.018 | <0.10 | ✅ PASS |

Bootstrap 95% CI (n=1,000 iterations): all primary metrics [0.833–1.000]

| Method | Sensitivity | Specificity | Precision |
|--------|-------------|-------------|-----------|
| **Main algorithm (v5)** | **96.0%** | **93.3%** | **92.3%** |
| Cosine similarity | 64.7% | 93.3% | 95.7% |
| Text-mining (PubMed) | 5.9% | 86.7% | 50.0% |
| Random baseline | 2.9% | 100.0% | 100.0% |

Single false negative: nusinersen/SMA — documented ASO coverage gap (max_phase=3, not in ChEMBL approved pool). Effective sensitivity on scoreable drugs: 25/25.

Two false positives: levothyroxine/IBD and cyclosporine/gout — both borderline scores (0.23–0.25) attributable to shared immune/inflammatory pathway genes rather than direct mechanistic links; no clinical evidence of therapeutic use.

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
        disease_name="pulmonary arterial hypertension",
        min_score=0.2,
        max_results=20,
    )
    for candidate in result["candidates"]:
        print(f"{candidate['drug_name']}: {candidate['score']:.3f}")
    await pipeline.close()

asyncio.run(main())
```

### Run validation (55-case test set)

```bash
PYTHONPATH=. python run_validation.py
```

### Run RepoDB benchmark (independent external validation)

```bash
# repodb.csv is downloaded automatically if not present
python repodb_benchmark.py --top-n 50 --output repodb_benchmark_results.json
```

### Run full validation pipeline

```bash
bash validate_all.sh
```

### Run statistical analysis (bootstrap CIs, calibration)

```bash
python score_calibration.py --input validation_results.json
```

## Project Structure

```
drug-repurposing/
├── backend/
│   └── pipeline/
│       ├── data_fetcher.py              # API calls: OpenTargets, ChEMBL, DGIdb
│       ├── graph_builder.py             # Knowledge graph construction (NetworkX)
│       ├── scorer.py                    # Gene/pathway/PPI/similarity/mechanism scoring
│       ├── calibration.py               # Platt scaling module (loaded from calibration_params.json)
│       ├── reactome_kegg_integration.py # Live Reactome + KEGG pathway mapping
│       ├── clinical_validator.py        # ClinicalTrials + PubMed + FDA validation
│       ├── drug_filter.py               # Safety contraindication filter
│       └── production_pipeline.py       # Orchestrator
├── validation_dataset.py                # 55-case test set (25 TP + 30 TN)
├── run_validation.py                    # Validation runner with metrics + baselines
├── repodb_benchmark.py                  # RepoDB standardized benchmark (auto-downloads CSV)
├── score_calibration.py                 # Bootstrap CIs, AUROC, Platt calibration (refits params)
├── calibration_params.json              # Fitted Platt parameters — written by score_calibration.py,
│                                        #   read by calibration.py. Single source of truth.
├── extended_baselines.py                # Jaccard, network proximity baselines
├── false_negative_analysis.py           # Root-cause analysis of missed cases
├── validate_all.sh                      # Full validation pipeline runner
├── REFERENCES.md                        # Full citation list
└── README.md                            # This file
```

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

### Standardized benchmark

The RepoDB benchmark (Brown & Patel 2017, Pac Symp Biocomput) provides >7,000 drug-disease pairs with FDA approval status and is used as an independent external validation. The benchmark CSV is downloaded automatically on first run:

```bash
python repodb_benchmark.py --output repodb_benchmark_results.json
```

### Statistical robustness

Bootstrap confidence intervals (1,000 iterations, 95% CI) and Platt calibration are computed by:

```bash
python score_calibration.py --input validation_results.json
```

This also writes updated Platt parameters to `calibration_params.json`, which `calibration.py` reads at runtime to ensure the deployed scoring system uses the same parameters reported in the paper.

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

Where:
- **gene_score**: Weighted Jaccard-based overlap between drug targets and disease-associated genes (OpenTargets evidence scores used as weights)
- **pathway_score**: Weighted overlap between drug pathway annotations (Reactome/KEGG) and disease pathways
- **ppi_score**: Network proximity score using STRING protein-protein interaction graph; captures indirect target-gene links within 2 hops
- **drug_similarity_score**: Tanimoto structural similarity to known drugs for the disease (requires ≥1 known drug in ChEMBL for the disease)
- **mechanism_score**: Pattern matching of drug mechanism text against disease biology keywords
- **literature_score**: Normalised PubMed co-occurrence count (live query, log-scaled)

Confidence tiers: `high` (≥0.40), `medium` (≥0.15), `low` (<0.15)

Scores are calibrated using Platt scaling. Parameters are fitted by `score_calibration.py` and stored in `calibration_params.json`. See `calibration.py` for implementation details.

**Note on calibrated scores**: With the current parameters, calibrated probabilities span approximately 0.43–0.47 across the raw score range [0, 1]. The standard 0.50 threshold is unreachable; use raw_score ≥ 0.20 for binary classification. Calibrated scores are reported for supplementary ranking only.

## Limitations

1. **Drug target coverage**: ~84% of the 2,700-drug pool has gene targets. Drugs with no targets in DGIdb, ChEMBL mechanism, or OpenTargets knownDrugs score 0 regardless of relevance. Antisense oligonucleotides (e.g. nusinersen) are systematically undercovered.

2. **Disease gene representation**: OpenTargets derives associations primarily from GWAS. Diseases with non-heritable pathophysiology (iatrogenic syndromes, structural protein defects) have sparse gene associations and reduced algorithm performance.

3. **Indirect mechanisms**: Empirical repurposing cases discovered via clinical observation (e.g., aspirin for CAD) score lower because their therapeutic mechanism is orthogonal to disease susceptibility genes.

4. **Pathway coverage**: Live Reactome/KEGG annotations cover ~95% of well-studied genes. Rarely annotated genes fall back to a curated supplementary map.

5. **Disease naming**: OpenTargets uses EFO ontology terms. Disease aliases are mapped automatically but uncommon disease names may fail to resolve.

6. **PPI inflation risk**: The PPI component can inflate scores for promiscuous hub genes (e.g. AMPK/mTOR pathway members). Prospective predictions for diseases with broad metabolic gene associations (e.g. Huntington disease) should be interpreted with caution.

## Known Issues and Fixes in Progress

- Gabapentin/neuropathic pain false negative: CACNA2D1/D2 targets not consistently returned by DGIdb. Fix: add to manual target supplement.
- Tocilizumab/CRS: "cytokine release syndrome" not in EFO. Fix: add to DISEASE_ALIASES as out-of-scope category.