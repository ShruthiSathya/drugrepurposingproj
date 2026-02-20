#!/usr/bin/env bash
# =============================================================================
# validate_all.sh — Full Validation Pipeline
# =============================================================================
# Runs all four validation steps in sequence, checking exit codes between
# steps. If any step fails, the pipeline halts with a non-zero exit code.
#
# Steps:
#   1. Curated validation (run_validation.py)              → validation_results.json
#   2. Calibration analysis (score_calibration.py)         → calibration_results.json
#   3. RepoDB benchmark (repodb_benchmark.py)              → repodb_benchmark_results.json
#   4. False negative analysis (false_negative_analysis.py)→ fn_analysis.json
#
# FIXES vs previous version
# -------------------------
# FIX 1: n_test_cases comment corrected to n=33 (was n=34).
#        Tocilizumab/CRS moved to OUT_OF_SCOPE in validation_dataset.py v3.
#
# FIX 2: Metrics extraction uses .metrics.sensitivity (correct nested path).
#        Old version used top-level .sensitivity which does not exist.
#
# FIX 3: Platt B sanity check now uses python3 for numeric comparison.
#        Old version used `bc` which may not be installed and silently
#        returned 0 (always passing the check even with negative B).
#
# Usage:
#   chmod +x validate_all.sh
#   ./validate_all.sh
#   ./validate_all.sh --repodb-file path/to/repodb.csv
#   ./validate_all.sh --skip-repodb
# =============================================================================

set -euo pipefail

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}   $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ─── Argument parsing ─────────────────────────────────────────────────────────
REPODB_FILE="repodb.csv"
SKIP_REPODB=0

for arg in "$@"; do
    case $arg in
        --repodb-file=*)  REPODB_FILE="${arg#*=}" ;;
        --skip-repodb)    SKIP_REPODB=1 ;;
    esac
done

# ─── Prerequisite checks ─────────────────────────────────────────────────────
log_section "PREREQUISITE CHECKS"

if ! command -v python3 &>/dev/null; then
    log_error "python3 not found. Please install Python 3.9+."
    exit 1
fi
log_ok "python3 found: $(python3 --version 2>&1)"

if ! command -v jq &>/dev/null; then
    log_warn "jq not found — metrics extraction at end will be skipped."
    log_warn "Install: apt install jq   OR   brew install jq"
    HAS_JQ=0
else
    log_ok "jq found: $(jq --version)"
    HAS_JQ=1
fi

for script in run_validation.py score_calibration.py false_negative_analysis.py; do
    if [[ ! -f "$script" ]]; then
        log_error "Missing required script: $script"
        log_error "Run this script from the repo root directory."
        exit 1
    fi
done
log_ok "All required scripts present"

if [[ $SKIP_REPODB -eq 0 ]] && [[ ! -f repodb_benchmark.py ]]; then
    log_warn "repodb_benchmark.py not found — RepoDB step will be skipped."
    SKIP_REPODB=1
fi

echo ""
START_TIME=$(date +%s)

# =============================================================================
# STEP 1: Curated Validation (n=33)
# =============================================================================
# FIX 1: n=33, not n=34. Tocilizumab/CRS removed in validation_dataset.py v3.
# =============================================================================
log_section "STEP 1/4: Curated Validation (n=33 test cases)"

python3 run_validation.py --output validation_results.json

if [[ ! -f validation_results.json ]]; then
    log_error "Step 1 FAILED — validation_results.json was not created."
    exit 1
fi
log_ok "Step 1 complete — validation_results.json written"

# Quick sanity: confirm n_test_cases matches expected value
if [[ $HAS_JQ -eq 1 ]]; then
    N_CASES=$(jq -r '.header.n_test_cases // "unknown"' validation_results.json)
    if [[ "$N_CASES" == "33" ]]; then
        log_ok "  n_test_cases = $N_CASES ✓"
    elif [[ "$N_CASES" == "unknown" ]]; then
        log_warn "  n_test_cases not found in .header — check run_validation.py output format"
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
    log_error "Step 2 FAILED — calibration_results.json was not created."
    exit 1
fi
log_ok "Step 2 complete — calibration_results.json written"

# =============================================================================
# STEP 3: RepoDB Benchmark (optional)
# =============================================================================
log_section "STEP 3/4: RepoDB Benchmark"

if [[ $SKIP_REPODB -eq 1 ]]; then
    log_warn "RepoDB benchmark SKIPPED."
    log_warn "To run: ./validate_all.sh --repodb-file path/to/repodb.csv"
    log_warn "Download RepoDB: https://unmtid-shinyapps.net/shiny/repodb/"
elif [[ ! -f "$REPODB_FILE" ]]; then
    log_warn "RepoDB file not found: $REPODB_FILE"
    log_warn "Download: https://unmtid-shinyapps.net/shiny/repodb/"
    log_warn "Skipping RepoDB step."
    SKIP_REPODB=1
else
    python3 repodb_benchmark.py \
        --repodb-file "$REPODB_FILE" \
        --top-n 50 \
        --output repodb_benchmark_results.json

    if [[ ! -f repodb_benchmark_results.json ]]; then
        log_error "Step 3 FAILED — repodb_benchmark_results.json was not created."
        exit 1
    fi
    log_ok "Step 3 complete — repodb_benchmark_results.json written"
fi

# =============================================================================
# STEP 4: False Negative Analysis
# =============================================================================
log_section "STEP 4/4: False Negative Analysis"

python3 false_negative_analysis.py \
    --input  validation_results.json \
    --output fn_analysis.json

if [[ ! -f fn_analysis.json ]]; then
    log_error "Step 4 FAILED — fn_analysis.json was not created."
    exit 1
fi
log_ok "Step 4 complete — fn_analysis.json written"

# =============================================================================
# SUMMARY
# =============================================================================
log_section "VALIDATION PIPELINE COMPLETE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "  Total elapsed: ${ELAPSED}s"
echo ""
echo "  Output files created:"
echo "    ✓ validation_results.json"
echo "    ✓ calibration_results.json"
if [[ $SKIP_REPODB -eq 0 ]]; then
    echo "    ✓ repodb_benchmark_results.json"
else
    echo "    – repodb_benchmark_results.json  (skipped)"
fi
echo "    ✓ fn_analysis.json"
echo ""

# =============================================================================
# METRICS EXTRACTION
# =============================================================================
if [[ $HAS_JQ -eq 0 ]]; then
    log_warn "jq not available — install it to see automatic metric display."
    echo ""
    log_ok "All validation steps completed successfully."
    exit 0
fi

log_section "KEY METRICS SUMMARY"

# ── Curated validation metrics ────────────────────────────────────────────────
# FIX 2: metrics live under .metrics.* not at top level
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  CURATED VALIDATION  (validation_results.json)  │"
echo "  └─────────────────────────────────────────────────┘"

SENSITIVITY=$(jq -r '.metrics.sensitivity // "N/A"' validation_results.json)
SPECIFICITY=$(jq -r '.metrics.specificity // "N/A"' validation_results.json)
PRECISION=$(jq   -r '.metrics.precision   // "N/A"' validation_results.json)
F1=$(jq          -r '.metrics.f1          // "N/A"' validation_results.json)
TP=$(jq          -r '.metrics.tp          // "N/A"' validation_results.json)
FN=$(jq          -r '.metrics.fn          // "N/A"' validation_results.json)
TN=$(jq          -r '.metrics.tn          // "N/A"' validation_results.json)
FP=$(jq          -r '.metrics.fp          // "N/A"' validation_results.json)
N_POS=$(jq       -r '.header.n_positive_cases // "N/A"' validation_results.json)
N_NEG=$(jq       -r '.header.n_negative_cases // "N/A"' validation_results.json)

echo "    Sensitivity (recall) : $SENSITIVITY   (TP=$TP / n_pos=$N_POS)"
echo "    Specificity          : $SPECIFICITY   (TN=$TN / n_neg=$N_NEG)"
echo "    Precision            : $PRECISION"
echo "    F1                   : $F1"
echo "    False positives      : $FP"
echo "    False negatives      : $FN"

# Threshold check — warn if sensitivity below publication target
SENS_OK=$(python3 -c "
try:
    s = float('$SENSITIVITY')
    print('yes' if s >= 0.65 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
if [[ "$SENS_OK" == "yes" ]]; then
    log_ok "  Sensitivity $SENSITIVITY >= 0.65 publication threshold ✓"
elif [[ "$SENS_OK" == "no" ]]; then
    log_warn "  Sensitivity $SENSITIVITY < 0.65 — below publication threshold"
    log_warn "  Run: python3 apply_patch.py   then re-run validation"
fi

# ── Calibration metrics ───────────────────────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  CALIBRATION         (calibration_results.json) │"
echo "  └─────────────────────────────────────────────────┘"

PLATT_A=$(jq  -r '.platt_parameters.A          // "N/A"' calibration_results.json)
PLATT_B=$(jq  -r '.platt_parameters.B          // "N/A"' calibration_results.json)
ECE=$(jq      -r '.ece                          // "N/A"' calibration_results.json)
D_TP_FN=$(jq  -r '.effect_sizes.cohens_d_tp_vs_fn // "N/A"' calibration_results.json)
D_TP_FP=$(jq  -r '.effect_sizes.cohens_d_tp_vs_fp // "N/A"' calibration_results.json)
RAW_THRESH=$(jq -r '.classification_threshold.raw_equivalent // "N/A"' calibration_results.json)

echo "    Platt A              : $PLATT_A"
echo "    Platt B              : $PLATT_B"
echo "    Raw score threshold  : $RAW_THRESH  (calibrated = 0.40)"
echo "    ECE                  : $ECE   (target < 0.15)"
echo "    Cohen's d (TP vs FN) : $D_TP_FN"
echo "    Cohen's d (TP vs FP) : $D_TP_FP"

# FIX 3: Use python3 for numeric comparison — bc may not be installed and
# silently returns exit code 0 (wrong), causing the check to always pass.
PLATT_B_RAW=$(jq -r '.platt_parameters.B' calibration_results.json 2>/dev/null || echo "0")

IS_NEGATIVE=$(python3 - <<PYEOF 2>/dev/null
try:
    b = float("$PLATT_B_RAW")
    print("yes" if b < 0 else "no")
except Exception:
    print("unknown")
PYEOF
)

if [[ "$IS_NEGATIVE" == "yes" ]]; then
    echo ""
    log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_error "  CRITICAL: Platt B = $PLATT_B_RAW  (NEGATIVE — BUG NOT FIXED)"
    log_error "  Open: backend/pipeline/calibration.py"
    log_error "  Find:    _PLATT_B: float = -0.42"
    log_error "  Change:  _PLATT_B: float = +0.42"
    log_error "  Then re-run: ./validate_all.sh"
    log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
elif [[ "$IS_NEGATIVE" == "no" ]]; then
    log_ok "  Platt B = +$PLATT_B_RAW (positive — correct) ✓"
else
    log_warn "  Could not parse Platt B value: '$PLATT_B_RAW'"
fi

# ECE check
ECE_OK=$(python3 -c "
try:
    e = float('$ECE')
    print('yes' if e < 0.15 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
if [[ "$ECE_OK" == "yes" ]]; then
    log_ok "  ECE $ECE < 0.15 ✓"
elif [[ "$ECE_OK" == "no" ]]; then
    log_warn "  ECE $ECE >= 0.15 — calibration may need refit"
fi

# ── RepoDB metrics ────────────────────────────────────────────────────────────
if [[ $SKIP_REPODB -eq 0 ]] && [[ -f repodb_benchmark_results.json ]]; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    echo "  │  REPODB BENCHMARK    (repodb_benchmark_results.json)        │"
    echo "  └─────────────────────────────────────────────────────────────┘"

    AUC_ROC=$(jq -r '.summary.global_auc_roc  // "N/A"' repodb_benchmark_results.json)
    AUC_PR=$(jq  -r '.summary.global_auc_pr   // "N/A"' repodb_benchmark_results.json)
    HIT50=$(jq   -r '.summary."hit_at_50"     // "N/A"' repodb_benchmark_results.json)
    MRR=$(jq     -r '.summary.mrr             // "N/A"' repodb_benchmark_results.json)
    N_DIS=$(jq   -r '.summary.n_diseases_tested // "N/A"' repodb_benchmark_results.json)
    N_DRUGS=$(jq -r '.summary.n_drugs         // "N/A"' repodb_benchmark_results.json)

    echo "    Diseases tested      : $N_DIS"
    echo "    Drugs in pool        : $N_DRUGS"
    echo "    Global AUC-ROC       : $AUC_ROC"
    echo "    Global AUC-PR        : $AUC_PR"
    echo "    Hit@50               : $HIT50"
    echo "    MRR                  : $MRR"
fi

# ── False negative analysis ───────────────────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  FALSE NEGATIVE ANALYSIS  (fn_analysis.json)    │"
echo "  └─────────────────────────────────────────────────┘"

FN_RATE=$(jq -r '.summary.fn_rate_percent    // "N/A"' fn_analysis.json)
FN_N=$(jq    -r '.summary.n_false_negatives  // "N/A"' fn_analysis.json)
FN_POS=$(jq  -r '.summary.n_positive_cases   // "N/A"' fn_analysis.json)
CATS=$(jq    -r '.summary.category_breakdown // {} | to_entries[] | "      \(.key): \(.value)"' fn_analysis.json 2>/dev/null || echo "      (could not parse)")

echo "    False negatives      : $FN_N / $FN_POS"
echo "    FN rate              : ${FN_RATE}%"
echo "    Breakdown by category:"
echo "$CATS"

# ── Final pass/fail summary ───────────────────────────────────────────────────
echo ""
log_section "PASS / FAIL CHECKLIST"
echo ""

# Helper: print pass or fail line
check() {
    local label="$1"
    local result="$2"   # "pass" or "fail" or "warn" or "skip"
    local detail="$3"
    case "$result" in
        pass) echo -e "  ${GREEN}[PASS]${NC} $label — $detail" ;;
        fail) echo -e "  ${RED}[FAIL]${NC} $label — $detail" ;;
        warn) echo -e "  ${YELLOW}[WARN]${NC} $label — $detail" ;;
        skip) echo -e "  ${BLUE}[SKIP]${NC} $label — $detail" ;;
    esac
}

# Sensitivity >= 65%
SENS_RESULT="fail"
[[ "$SENS_OK" == "yes" ]] && SENS_RESULT="pass"
[[ "$SENS_OK" == "unknown" ]] && SENS_RESULT="warn"
check "Sensitivity ≥ 65%" "$SENS_RESULT" "$SENSITIVITY"

# Specificity >= 75%
SPEC_OK=$(python3 -c "
try:
    s = float('$SPECIFICITY')
    print('yes' if s >= 0.75 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
SPEC_RESULT="fail"
[[ "$SPEC_OK" == "yes" ]] && SPEC_RESULT="pass"
[[ "$SPEC_OK" == "unknown" ]] && SPEC_RESULT="warn"
check "Specificity ≥ 75%" "$SPEC_RESULT" "$SPECIFICITY"

# Precision >= 65%
PREC_OK=$(python3 -c "
try:
    p = float('$PRECISION')
    print('yes' if p >= 0.65 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
PREC_RESULT="fail"
[[ "$PREC_OK" == "yes" ]] && PREC_RESULT="pass"
[[ "$PREC_OK" == "unknown" ]] && PREC_RESULT="warn"
check "Precision ≥ 65%" "$PREC_RESULT" "$PRECISION"

# Platt B positive
PLATT_B_RESULT="fail"
[[ "$IS_NEGATIVE" == "no" ]] && PLATT_B_RESULT="pass"
[[ "$IS_NEGATIVE" == "unknown" ]] && PLATT_B_RESULT="warn"
check "Platt B is positive" "$PLATT_B_RESULT" "B = $PLATT_B_RAW"

# ECE < 0.15
ECE_RESULT="fail"
[[ "$ECE_OK" == "yes" ]] && ECE_RESULT="pass"
[[ "$ECE_OK" == "unknown" ]] && ECE_RESULT="warn"
check "ECE < 0.15" "$ECE_RESULT" "$ECE"

# RepoDB run
if [[ $SKIP_REPODB -eq 0 ]] && [[ -f repodb_benchmark_results.json ]]; then
    check "RepoDB benchmark" "pass" "AUC-ROC=$AUC_ROC  Hit@50=$HIT50"
else
    check "RepoDB benchmark" "skip" "run with --repodb-file to enable"
fi

echo ""

# ── Next steps if any failures ────────────────────────────────────────────────
if [[ "$SENS_RESULT" == "fail" ]] || [[ "$PLATT_B_RESULT" == "fail" ]]; then
    echo "  ─────────────────────────────────────────────────────────"
    echo "  RECOMMENDED NEXT STEPS:"
    if [[ "$PLATT_B_RESULT" == "fail" ]]; then
        echo "  1. Fix Platt B in backend/pipeline/calibration.py (change -0.42 → +0.42)"
    fi
    if [[ "$SENS_RESULT" == "fail" ]]; then
        echo "  2. Run: python3 apply_patch.py  (adds missing drug targets)"
        echo "     Then: rm -f /tmp/drug_repurposing_cache/chembl_approved_drugs.json"
        echo "     Then: ./validate_all.sh"
    fi
    echo "  ─────────────────────────────────────────────────────────"
    echo ""
fi

log_ok "Validation pipeline finished. $(date)"
echo ""