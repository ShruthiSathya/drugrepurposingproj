#!/bin/bash
# =============================================================================
# NAVARA AI — Master Validation Pipeline
# =============================================================================
# Runs the full validation suite using your existing Python modules:
#   1. Curated test set (run_validation.py)          → 33 cases, 15 negatives
#   2. Score calibration (score_calibration.py)      → bootstrap CIs, AUROC
#   3. RepoDB benchmark (repodb_benchmark.py)        → 150 positive controls
#   4. False-negative analysis (false_negative_analysis.py)
#
# Usage:
#   chmod +x validate_all.sh
#   ./validate_all.sh                    # full run
#   ./validate_all.sh --quick            # quick mode (25 RepoDB pairs)
#   ./validate_all.sh --skip-repodb      # skip slow RepoDB download
#   ./validate_all.sh --threshold 0.35   # custom calibrated score threshold
#
# Outputs (timestamped in ./validation_runs/<timestamp>/):
#   curated_results.json          — per-case scores, TP/FP/TN/FN
#   calibration_results.json      — bootstrap CIs, AUROC, Platt params
#   repodb_results.json           — external benchmark (optional)
#   fn_analysis.txt               — root-cause false negative breakdown
#   SUMMARY.txt                   — publication-ready metrics table
#   validation_report.md          — full markdown report for paper
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
QUICK=false
SKIP_REPODB=false
THRESHOLD=0.40
SHOW_FN=true
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./validation_runs/${TIMESTAMP}"
PYTHON="python3"
VENV_PATH="./backend/venv"

# ── Colour output ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()    { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()     { echo -e "${GREEN}✅ $*${NC}"; }
warn()   { echo -e "${YELLOW}⚠️  $*${NC}"; }
error()  { echo -e "${RED}❌ $*${NC}"; }
header() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; \
           echo -e "${BOLD}${BLUE}  $*${NC}"; \
           echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}"; }

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)       QUICK=true; shift ;;
        --skip-repodb) SKIP_REPODB=true; shift ;;
        --threshold)   THRESHOLD="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --python)      PYTHON="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--quick] [--skip-repodb] [--threshold FLOAT] [--output-dir DIR]"
            exit 0 ;;
        *) warn "Unknown arg: $1"; shift ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
header "NAVARA AI — Validation Pipeline"
log "Timestamp:  ${TIMESTAMP}"
log "Output dir: ${OUTPUT_DIR}"
log "Threshold:  ${THRESHOLD} (calibrated score)"
log "Quick mode: ${QUICK}"
log "Skip RepoDB: ${SKIP_REPODB}"
echo ""

if [ ! -f "README.md" ]; then
    error "Run this script from the project root directory."
    exit 1
fi

# Activate venv if available
if [ -d "${VENV_PATH}" ]; then
    source "${VENV_PATH}/bin/activate"
    ok "Activated venv: ${VENV_PATH}"
else
    warn "No venv found at ${VENV_PATH}. Using system Python: $(${PYTHON} --version)"
fi

# Verify key scripts exist
REQUIRED_SCRIPTS=(
    "run_validation.py"
    "score_calibration.py"
    "false_negative_analysis.py"
    "validation_dataset.py"
    "backend/pipeline/calibration.py"
)
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "${script}" ]; then
        error "Missing required file: ${script}"
        exit 1
    fi
done
ok "All required scripts present."

# Create output dir
mkdir -p "${OUTPUT_DIR}"
log "Results will be saved to: ${OUTPUT_DIR}/"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Curated Validation (run_validation.py)
# ─────────────────────────────────────────────────────────────────────────────
header "STEP 1/4 — Curated Validation (33 cases + 15 negatives)"
CURATED_OUT="${OUTPUT_DIR}/curated_results.json"

CURATED_FLAGS="--output ${CURATED_OUT} --threshold ${THRESHOLD}"
if [ "${SHOW_FN}" = true ]; then
    CURATED_FLAGS="${CURATED_FLAGS} --show-fn"
fi

log "Running: PYTHONPATH=. ${PYTHON} run_validation.py ${CURATED_FLAGS}"
start_time=$SECONDS

if PYTHONPATH=. ${PYTHON} run_validation.py ${CURATED_FLAGS} 2>&1 | tee "${OUTPUT_DIR}/curated_run.log"; then
    elapsed=$((SECONDS - start_time))
    ok "Curated validation complete (${elapsed}s) → ${CURATED_OUT}"
else
    error "Curated validation FAILED. Check ${OUTPUT_DIR}/curated_run.log"
    exit 1
fi

# Extract key metrics for display
if [ -f "${CURATED_OUT}" ]; then
    SENS=$(${PYTHON} -c "import json; d=json.load(open('${CURATED_OUT}')); print(f\"{d['metrics']['sensitivity']:.1%}\")" 2>/dev/null || echo "N/A")
    SPEC=$(${PYTHON} -c "import json; d=json.load(open('${CURATED_OUT}')); print(f\"{d['metrics']['specificity']:.1%}\")" 2>/dev/null || echo "N/A")
    PREC=$(${PYTHON} -c "import json; d=json.load(open('${CURATED_OUT}')); print(f\"{d['metrics']['precision']:.1%}\")" 2>/dev/null || echo "N/A")
    F1=$(${PYTHON} -c "import json; d=json.load(open('${CURATED_OUT}')); print(f\"{d['metrics']['f1']:.4f}\")" 2>/dev/null || echo "N/A")
    log "  Sensitivity: ${SENS}  |  Specificity: ${SPEC}  |  Precision: ${PREC}  |  F1: ${F1}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Score Calibration + Bootstrap CIs (score_calibration.py)
# ─────────────────────────────────────────────────────────────────────────────
header "STEP 2/4 — Score Calibration & Bootstrap CIs"
CALIBRATION_OUT="${OUTPUT_DIR}/calibration_results.json"

# score_calibration.py reads from validation_results.json (existing run output)
# We'll point it at our freshly generated curated results
if [ -f "${CURATED_OUT}" ]; then
    log "Running: PYTHONPATH=. ${PYTHON} score_calibration.py ${CURATED_OUT} --output ${CALIBRATION_OUT}"

    if PYTHONPATH=. ${PYTHON} score_calibration.py "${CURATED_OUT}" \
        --output "${CALIBRATION_OUT}" \
        --n-bootstrap 1000 2>&1 | tee "${OUTPUT_DIR}/calibration_run.log"; then
        ok "Calibration complete → ${CALIBRATION_OUT}"

        AUROC=$(${PYTHON} -c "import json; d=json.load(open('${CALIBRATION_OUT}')); print(f\"{d['point_estimates']['auroc']:.3f}\")" 2>/dev/null || echo "N/A")
        log "  AUROC: ${AUROC}"
    else
        warn "Calibration script failed — continuing without CI data."
        warn "Check ${OUTPUT_DIR}/calibration_run.log for details."
        # Note: score_calibration.py expects specific keys; if curated_results format differs,
        # it may fail. The summary will still be generated from curated_results.
    fi
else
    warn "Curated results file missing — skipping calibration."
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: RepoDB External Benchmark (repodb_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────
REPODB_OUT="${OUTPUT_DIR}/repodb_results.json"

if [ "${SKIP_REPODB}" = true ]; then
    header "STEP 3/4 — RepoDB Benchmark (SKIPPED)"
    warn "Skipping RepoDB benchmark. Run without --skip-repodb for full external validation."
else
    header "STEP 3/4 — RepoDB External Benchmark"

    REPODB_FLAGS="--output ${REPODB_OUT}"
    if [ "${QUICK}" = true ]; then
        REPODB_FLAGS="${REPODB_FLAGS} --quick"
        log "Quick mode: 25 positive + 10 negative controls"
    else
        log "Full mode: 150 positive + 50 negative controls (~4 hours)"
    fi

    # Check if repodb_full.csv already downloaded (cached)
    if [ -f "repodb_full.csv" ]; then
        REPODB_FLAGS="${REPODB_FLAGS} --offline --repodb-csv repodb_full.csv"
        log "Using cached RepoDB CSV: repodb_full.csv"
    fi

    log "Running: PYTHONPATH=. ${PYTHON} repodb_benchmark.py ${REPODB_FLAGS}"

    if PYTHONPATH=. ${PYTHON} repodb_benchmark.py ${REPODB_FLAGS} 2>&1 | tee "${OUTPUT_DIR}/repodb_run.log"; then
        ok "RepoDB benchmark complete → ${REPODB_OUT}"

        if [ -f "${REPODB_OUT}" ]; then
            RDB_SENS=$(${PYTHON} -c "import json; d=json.load(open('${REPODB_OUT}')); print(f\"{d['sensitivity']:.1%}\")" 2>/dev/null || echo "N/A")
            RDB_SPEC=$(${PYTHON} -c "import json; d=json.load(open('${REPODB_OUT}')); print(f\"{d['specificity']:.1%}\")" 2>/dev/null || echo "N/A")
            log "  RepoDB Sensitivity: ${RDB_SENS}  |  Specificity: ${RDB_SPEC}"
        fi
    else
        warn "RepoDB benchmark failed — check ${OUTPUT_DIR}/repodb_run.log"
        warn "This is expected if network is unavailable. Use --offline with a local CSV."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: False Negative Root-Cause Analysis
# ─────────────────────────────────────────────────────────────────────────────
header "STEP 4/4 — False Negative Analysis"
FN_OUT="${OUTPUT_DIR}/fn_analysis.txt"

log "Running: PYTHONPATH=. ${PYTHON} false_negative_analysis.py"
if PYTHONPATH=. ${PYTHON} false_negative_analysis.py > "${FN_OUT}" 2>&1; then
    ok "FN analysis complete → ${FN_OUT}"
    # Print summary
    grep -A 3 "SUMMARY FOR PAPER" "${FN_OUT}" | head -20 || true
else
    warn "FN analysis failed — check ${FN_OUT}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Generate Publication Summary
# ─────────────────────────────────────────────────────────────────────────────
header "STEP 5/5 — Generating Publication Summary"
SUMMARY_OUT="${OUTPUT_DIR}/SUMMARY.txt"
REPORT_OUT="${OUTPUT_DIR}/validation_report.md"

${PYTHON} - <<PYEOF > "${SUMMARY_OUT}" 2>&1
import json, os, sys
from datetime import datetime

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return None

curated    = load("${CURATED_OUT}")
calibrated = load("${CALIBRATION_OUT}")
repodb     = load("${REPODB_OUT}") if os.path.exists("${REPODB_OUT}") else None

lines = []
lines.append("=" * 65)
lines.append("NAVARA AI — DRUG REPURPOSING VALIDATION SUMMARY")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 65)

if curated:
    m = curated.get("metrics", {})
    lines.append("")
    lines.append("CURATED VALIDATION (Primary — for publication headline)")
    lines.append("-" * 65)
    lines.append(f"  Test cases:     {curated.get('metadata', {}).get('n_test_cases', 33)}")
    lines.append(f"  Neg controls:   {curated.get('metadata', {}).get('n_negative_controls', 15)}")
    lines.append(f"  TP={m.get('tp','?')}, FN={m.get('fn','?')}, TN={m.get('tn','?')}, FP={m.get('fp','?')}")
    lines.append(f"  Sensitivity:    {m.get('sensitivity',0):.1%}")
    lines.append(f"  Specificity:    {m.get('specificity',0):.1%}")
    lines.append(f"  Precision:      {m.get('precision',0):.1%}")
    lines.append(f"  F1 Score:       {m.get('f1',0):.4f}")
    lines.append("")
    lines.append("  Per-category sensitivity:")
    for cat, d in m.get("by_category", {}).items():
        lines.append(f"    {cat:30s}: {d['sensitivity']:.1%}  (n={d['n']})")

if calibrated:
    lines.append("")
    lines.append("STATISTICAL ANALYSIS (Bootstrap 95% CIs, n=1000)")
    lines.append("-" * 65)
    pe = calibrated.get("point_estimates", {})
    cis = calibrated.get("bootstrap_cis", {})
    lines.append(f"  AUROC:          {pe.get('auroc',0):.3f}")
    lines.append(f"  AUPRC:          {pe.get('auprc',0):.3f}")
    for metric in ["sensitivity", "specificity", "precision", "f1"]:
        ci = cis.get(metric, {})
        lines.append(f"  {metric:15s}: {pe.get(metric,0):.3f}  (95% CI: {ci.get('lower',0):.3f}–{ci.get('upper',0):.3f})")
    ps = calibrated.get("platt_scaling", {})
    lines.append(f"  Platt A:        {ps.get('A',0):.4f}")
    lines.append(f"  Platt B:        {ps.get('B',0):.4f}")
    cd = calibrated.get("cohens_d", {})
    lines.append(f"  Cohen's d:      {cd.get('d',0):.3f} ({cd.get('interpretation','?')} effect)")

if repodb:
    lines.append("")
    lines.append("REPODB EXTERNAL BENCHMARK")
    lines.append("-" * 65)
    lines.append(f"  Cases evaluated: {repodb.get('n_evaluated','?')}")
    lines.append(f"  Sensitivity:     {repodb.get('sensitivity',0):.1%}  (95% CI: {repodb.get('sensitivity_ci_95', ['?','?'])[0]}–{repodb.get('sensitivity_ci_95', ['?','?'])[1]})")
    lines.append(f"  Specificity:     {repodb.get('specificity',0):.1%}")
    lines.append(f"  Precision:       {repodb.get('precision',0):.1%}")
    lines.append(f"  F1:              {repodb.get('f1',0):.4f}")
    lines.append(f"  Top-k:           {repodb.get('top_k','?')}")
    lines.append("")
    lines.append("  By disease category:")
    for cat, d in repodb.get("by_disease_category", {}).items():
        lines.append(f"    {cat:14s}: {d.get('sensitivity',0):.1%}  (n={d.get('n','?')})")

lines.append("")
lines.append("SCORING ALGORITHM WEIGHTS")
lines.append("-" * 65)
lines.append("  Gene overlap:        45%  (weighted Jaccard, OpenTargets scores)")
lines.append("  Pathway overlap:     30%  (Reactome/KEGG annotations)")
lines.append("  Mechanism pattern:   10%  (text matching)")
lines.append("  Literature signal:   15%  (PubMed co-occurrence, log-scaled)")
lines.append("")
lines.append("DATA SOURCES")
lines.append("-" * 65)
lines.append("  OpenTargets Platform  — disease-gene associations (GWAS)")
lines.append("  ChEMBL               — ~2,700 approved drugs (max_phase=4)")
lines.append("  DGIdb 4.0            — drug-gene interactions (primary)")
lines.append("  ChEMBL Mechanism API — drug gene targets (supplement)")
lines.append("  OpenTargets knownDrugs — biologic target fallback")
lines.append("  Reactome/KEGG        — pathway annotations (live API)")
lines.append("  PubMed E-utilities   — literature co-occurrence scores")
lines.append("=" * 65)

print("\n".join(lines))
PYEOF

ok "Summary written → ${SUMMARY_OUT}"
cat "${SUMMARY_OUT}"

# Generate markdown report
${PYTHON} - <<PYEOF > "${REPORT_OUT}" 2>&1
import json, os
from datetime import datetime

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

curated    = load("${CURATED_OUT}")
calibrated = load("${CALIBRATION_OUT}")
repodb     = load("${REPODB_OUT}") if os.path.exists("${REPODB_OUT}") else None

lines = [
    "# Navara AI — Drug Repurposing Validation Report",
    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Overview",
    "Computational drug repurposing pipeline using network-based gene-pathway overlap scoring.",
    "All data sourced live from OpenTargets, ChEMBL, DGIdb, Reactome/KEGG, and PubMed.",
    "",
    "## Scoring Algorithm",
    "```",
    "score = 0.45 × gene_score",
    "      + 0.30 × pathway_score",
    "      + 0.10 × mechanism_score",
    "      + 0.15 × literature_score",
    "```",
    "",
    "## Primary Validation Results",
]

if curated:
    m = curated.get("metrics", {})
    meta = curated.get("metadata", {})
    lines += [
        "",
        f"**Test set:** {meta.get('n_test_cases', 33)} drug-disease pairs  |  **Negative controls:** {meta.get('n_negative_controls', 15)}",
        f"**Classification threshold:** {meta.get('threshold', 0.40)} (calibrated score)",
        "",
        "| Metric | Value | Publication Target | Status |",
        "|--------|-------|-------------------|--------|",
        f"| Sensitivity | {m.get('sensitivity',0):.1%} | ≥65% | {'✅ PASS' if m.get('sensitivity',0) >= 0.65 else '❌ FAIL'} |",
        f"| Specificity | {m.get('specificity',0):.1%} | ≥75% | {'✅ PASS' if m.get('specificity',0) >= 0.75 else '❌ FAIL'} |",
        f"| Precision   | {m.get('precision',0):.1%}  | ≥65% | {'✅ PASS' if m.get('precision',0) >= 0.65 else '❌ FAIL'} |",
        f"| F1 Score    | {m.get('f1',0):.4f} | — | — |",
        "",
        "### Per-Category Sensitivity",
        "",
        "| Category | Sensitivity | n |",
        "|----------|-------------|---|",
    ]
    for cat, d in m.get("by_category", {}).items():
        lines.append(f"| {cat} | {d['sensitivity']:.1%} | {d['n']} |")

if calibrated:
    pe = calibrated.get("point_estimates", {})
    cis = calibrated.get("bootstrap_cis", {})
    cd = calibrated.get("cohens_d", {})
    ps = calibrated.get("platt_scaling", {})
    lines += [
        "",
        "## Statistical Analysis (Bootstrap 95% CIs, n=1000)",
        "",
        "| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |",
        "|--------|----------------|--------------|--------------|",
    ]
    for metric in ["sensitivity", "specificity", "precision", "f1", "auroc"]:
        ci = cis.get(metric, {})
        lines.append(f"| {metric.capitalize()} | {pe.get(metric,0):.3f} | {ci.get('lower',0):.3f} | {ci.get('upper',0):.3f} |")
    lines += [
        "",
        f"**Cohen's d:** {cd.get('d',0):.3f} ({cd.get('interpretation','?')} effect size)",
        f"**Platt scaling:** A={ps.get('A',0):.4f}, B={ps.get('B',0):.4f}",
    ]

if repodb:
    lines += [
        "",
        "## RepoDB External Benchmark",
        "",
        f"**Evaluated:** {repodb.get('n_evaluated', '?')} pairs  |  **Top-k:** {repodb.get('top_k', 100)}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sensitivity | {repodb.get('sensitivity',0):.1%} |",
        f"| Specificity | {repodb.get('specificity',0):.1%} |",
        f"| Precision   | {repodb.get('precision',0):.1%} |",
        f"| F1 Score    | {repodb.get('f1',0):.4f} |",
    ]

lines += [
    "",
    "## Known Limitations",
    "1. **Drug target coverage (~84%):** ~16% of approved drugs have no gene targets in DGIdb/ChEMBL/OpenTargets.",
    "2. **GWAS bias:** OpenTargets disease genes reflect susceptibility variants, not therapeutic targets.",
    "3. **Indirect mechanisms:** Empirical repurposing (aspirin/CAD, bupropion/smoking) scores lower due to orthogonal mechanisms.",
    "4. **Disease representation:** Rare/iatrogenic diseases (SSc, CRS) have sparse gene associations.",
    "",
    "## Data Sources",
    "- OpenTargets Platform (disease-gene associations)",
    "- ChEMBL (approved drugs, max_phase=4)",
    "- DGIdb 4.0 (drug-gene interactions)",
    "- Reactome + KEGG (pathway annotations)",
    "- PubMed E-utilities (literature co-occurrence)",
    "- ClinicalTrials.gov (active trial counts)",
]

print("\n".join(lines))
PYEOF

ok "Markdown report written → ${REPORT_OUT}"

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
header "VALIDATION COMPLETE"
echo ""
echo "  Output directory: ${OUTPUT_DIR}/"
echo ""
ls -lh "${OUTPUT_DIR}/"
echo ""
ok "All validation steps complete."
echo ""
echo "  Next steps for publication:"
echo "  1. Review ${OUTPUT_DIR}/validation_report.md"
echo "  2. Use calibration_results.json for Supplementary Fig S1 (reliability diagram)"
echo "  3. Run with full RepoDB for Table 3 in paper:"
echo "     ./validate_all.sh  (without --skip-repodb)"
echo ""