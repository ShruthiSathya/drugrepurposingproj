#!/usr/bin/env bash
# =============================================================================
# validate_all.sh — Full Validation Pipeline
# =============================================================================
# Runs all four validation steps in sequence, checking exit codes between
# steps. If any step fails, the pipeline halts with a non-zero exit code.
#
# Steps:
#   1. Curated validation (run_validation.py)       → validation_results.json
#   2. Calibration analysis (score_calibration.py)  → calibration_results.json
#   3. RepoDB benchmark (repodb_benchmark.py)        → repodb_benchmark_results.json
#   4. False negative analysis (false_negative_analysis.py) → fn_analysis.json
#
# FIXES vs previous version
# -------------------------
# 1. n_test_cases comment: Updated to n=33 (was n=34).
#    Tocilizumab/CRS moved to OUT_OF_SCOPE in validation_dataset.py v3.
#
# 2. Metrics extraction: .metrics.sensitivity path is correct for the JSON
#    structure written by run_validation.py (sensitivity is under .metrics,
#    not at top level). Also added calibration metrics extraction.
#
# Usage:
#   chmod +x validate_all.sh
#   ./validate_all.sh [--repodb-file repodb.csv] [--skip-repodb]
# =============================================================================

set -euo pipefail

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No color

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}   $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}  $*${NC}"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ─── Argument parsing ─────────────────────────────────────────────────────────
REPODB_FILE="repodb.csv"
SKIP_REPODB=0

for arg in "$@"; do
    case $arg in
        --repodb-file=*)  REPODB_FILE="${arg#*=}" ;;
        --skip-repodb)    SKIP_REPODB=1            ;;
    esac
done

# ─── Prerequisite checks ─────────────────────────────────────────────────────
log_section "PREREQUISITE CHECKS"

if ! command -v python3 &>/dev/null; then
    log_error "python3 not found. Please install Python 3.9+."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    log_warn "jq not found — metrics extraction at end will be skipped."
    HAS_JQ=0
else
    HAS_JQ=1
fi

# Check required Python scripts
for script in run_validation.py score_calibration.py false_negative_analysis.py; do
    if [[ ! -f "$script" ]]; then
        log_error "Missing script: $script"
        exit 1
    fi
done

if [[ $SKIP_REPODB -eq 0 ]] && [[ ! -f repodb_benchmark.py ]]; then
    log_warn "repodb_benchmark.py not found. Skipping RepoDB step."
    SKIP_REPODB=1
fi

log_ok "Prerequisites satisfied"
echo ""

START_TIME=$(date +%s)

# =============================================================================
# STEP 1: Curated Validation
# =============================================================================
# FIX 1: Updated n=33 (was n=34). Tocilizumab/CRS removed from test set in
# validation_dataset.py v3. The dataset constant N_TEST_CASES = 33.
# =============================================================================
log_section "STEP 1/4: Curated Validation (n=33 test cases)"

python3 run_validation.py --output validation_results.json

if [[ ! -f validation_results.json ]]; then
    log_error "Step 1 FAILED — validation_results.json not created"
    exit 1
fi

log_ok "Step 1 complete — validation_results.json created"

# Quick check: confirm n_test_cases is 33
if [[ $HAS_JQ -eq 1 ]]; then
    N_CASES=$(jq -r '.header.n_test_cases // "unknown"' validation_results.json)
    if [[ "$N_CASES" == "33" ]]; then
        log_ok "  n_test_cases = $N_CASES (correct)"
    elif [[ "$N_CASES" == "unknown" ]]; then
        log_warn "  n_test_cases field not found in header"
    else
        log_warn "  n_test_cases = $N_CASES (expected 33 — check validation_dataset.py)"
    fi
fi

# =============================================================================
# STEP 2: Score Calibration Analysis
# =============================================================================
log_section "STEP 2/4: Score Calibration Analysis"

python3 score_calibration.py \
    --input  validation_results.json \
    --output calibration_results.json

if [[ ! -f calibration_results.json ]]; then
    log_error "Step 2 FAILED — calibration_results.json not created"
    exit 1
fi

log_ok "Step 2 complete — calibration_results.json created"

# =============================================================================
# STEP 3: RepoDB Benchmark
# =============================================================================
log_section "STEP 3/4: RepoDB Benchmark"

if [[ $SKIP_REPODB -eq 1 ]]; then
    log_warn "RepoDB benchmark SKIPPED (--skip-repodb flag or missing repodb.csv)"
    log_warn "To run: ./validate_all.sh --repodb-file path/to/repodb.csv"
elif [[ ! -f "$REPODB_FILE" ]]; then
    log_warn "RepoDB file not found: $REPODB_FILE"
    log_warn "Download from: https://unmtid-shinyapps.net/shiny/repodb/"
    log_warn "Skipping RepoDB step."
else
    python3 repodb_benchmark.py \
        --repodb-file "$REPODB_FILE" \
        --top-n 50 \
        --output repodb_benchmark_results.json

    if [[ ! -f repodb_benchmark_results.json ]]; then
        log_error "Step 3 FAILED — repodb_benchmark_results.json not created"
        exit 1
    fi
    log_ok "Step 3 complete — repodb_benchmark_results.json created"
fi

# =============================================================================
# STEP 4: False Negative Analysis
# =============================================================================
log_section "STEP 4/4: False Negative Analysis"

python3 false_negative_analysis.py \
    --input  validation_results.json \
    --output fn_analysis.json

if [[ ! -f fn_analysis.json ]]; then
    log_error "Step 4 FAILED — fn_analysis.json not created"
    exit 1
fi

log_ok "Step 4 complete — fn_analysis.json created"

# =============================================================================
# SUMMARY
# =============================================================================
log_section "VALIDATION PIPELINE COMPLETE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "  Total time: ${ELAPSED}s"
echo ""
echo "  Output files:"
echo "    validation_results.json"
echo "    calibration_results.json"
[[ $SKIP_REPODB -eq 0 ]] && echo "    repodb_benchmark_results.json"
echo "    fn_analysis.json"
echo ""

# ─── Metrics extraction ───────────────────────────────────────────────────────
# FIX 2: Metrics are under '.metrics.*' (not top-level) in validation_results.json.
# Extract from the correct nested path.
if [[ $HAS_JQ -eq 1 ]]; then
    log_section "KEY METRICS"

    echo ""
    echo "  CURATED VALIDATION (validation_results.json)"
    # FIX: Use .metrics.sensitivity not .sensitivity
    SENSITIVITY=$(jq -r '.metrics.sensitivity // "N/A"' validation_results.json)
    SPECIFICITY=$(jq -r '.metrics.specificity // "N/A"' validation_results.json)
    PRECISION=$(jq -r '.metrics.precision   // "N/A"' validation_results.json)
    F1=$(jq -r '.metrics.f1          // "N/A"' validation_results.json)
    TP=$(jq -r '.metrics.tp          // "N/A"' validation_results.json)
    FN=$(jq -r '.metrics.fn          // "N/A"' validation_results.json)

    echo "    Sensitivity (recall): $SENSITIVITY"
    echo "    Specificity:          $SPECIFICITY"
    echo "    Precision:            $PRECISION"
    echo "    F1:                   $F1"
    echo "    TP/FN:                $TP/$FN"
    echo ""

    echo "  CALIBRATION (calibration_results.json)"
    # Platt parameters
    PLATT_A=$(jq -r '.platt_parameters.A // "N/A"' calibration_results.json)
    PLATT_B=$(jq -r '.platt_parameters.B // "N/A"' calibration_results.json)
    ECE=$(jq -r '.ece // "N/A"' calibration_results.json)
    D_TP_FN=$(jq -r '.effect_sizes.cohens_d_tp_vs_fn // "N/A"' calibration_results.json)
    D_TP_FP=$(jq -r '.effect_sizes.cohens_d_tp_vs_fp // "N/A"' calibration_results.json)

    echo "    Platt A:              $PLATT_A"
    echo "    Platt B:              +$PLATT_B  (positive value — check this is NOT negative)"
    echo "    ECE:                  $ECE"
    echo "    Cohen's d (TP vs FN): $D_TP_FN"
    echo "    Cohen's d (TP vs FP): $D_TP_FP"
    echo ""

    if [[ $SKIP_REPODB -eq 0 ]] && [[ -f repodb_benchmark_results.json ]]; then
        echo "  REPODB BENCHMARK (repodb_benchmark_results.json)"
        AUC_ROC=$(jq -r '.summary.global_auc_roc // "N/A"' repodb_benchmark_results.json)
        AUC_PR=$(jq -r '.summary.global_auc_pr  // "N/A"' repodb_benchmark_results.json)
        HIT50=$(jq -r '.summary."hit_at_50"     // "N/A"' repodb_benchmark_results.json)
        MRR=$(jq -r '.summary.mrr               // "N/A"' repodb_benchmark_results.json)

        echo "    Global AUC-ROC:       $AUC_ROC"
        echo "    Global AUC-PR:        $AUC_PR"
        echo "    Hit@50:               $HIT50"
        echo "    MRR:                  $MRR"
        echo ""
    fi

    echo "  FN ANALYSIS (fn_analysis.json)"
    FN_RATE=$(jq -r '.summary.fn_rate_percent // "N/A"' fn_analysis.json)
    FN_N=$(jq -r '.summary.n_false_negatives  // "N/A"' fn_analysis.json)
    echo "    False negative rate:  ${FN_RATE}%  (n=${FN_N})"
    echo ""

    # Sanity check: flag if Platt B is negative (means the fix wasn't applied)
    PLATT_B_NUM=$(jq -r '.platt_parameters.B' calibration_results.json 2>/dev/null || echo "0")
    if (( $(echo "$PLATT_B_NUM < 0" | bc -l 2>/dev/null || echo "0") )); then
        log_error "  ALERT: Platt B is negative ($PLATT_B_NUM)! Expected positive. Check calibration.py."
    else
        log_ok "  Platt B = +$PLATT_B (positive — correct)"
    fi

else
    log_warn "jq not available — skipping metrics extraction"
    log_warn "Install jq (apt install jq / brew install jq) for automatic metric display"
fi

echo ""
log_ok "All validation steps completed successfully"
echo ""