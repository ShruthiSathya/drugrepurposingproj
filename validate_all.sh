#!/usr/bin/env bash
# =============================================================================
# validate_all.sh — Full Validation Pipeline v2
# =============================================================================
# Runs all five validation steps in sequence.
#
# FIXES vs v1
# -----------
# FIX 1: n_test_cases comment corrected to n=55 (v4.0 dataset).
#
# FIX 2: Platt B sign check REMOVED / CORRECTED.
#   Previously the script flagged negative B as a CRITICAL ERROR. This was
#   wrong. B is the intercept and has no sign constraint — a negative B is
#   a valid fitted value. The check is now removed entirely.
#   The CORRECT check is that A must be NEGATIVE (higher raw score →
#   higher calibrated probability). Only A is checked.
#
# FIX 3: Metrics extraction uses .metrics.sensitivity (correct nested path).
#
# FIX 4: Step 5 (statistical_tests.py) added for bootstrap CIs.
#
# FIX 5: RepoDB step now uses --max-diseases 20 to limit benchmark to first
#   20 diseases. Full RepoDB (~1500 diseases) takes many hours. Run with
#   --full-repodb to disable the limit.
#
# Usage:
#   chmod +x validate_all.sh
#   ./validate_all.sh
#   ./validate_all.sh --repodb-file path/to/repodb.csv
#   ./validate_all.sh --skip-repodb
#   ./validate_all.sh --full-repodb      # removes the 20-disease limit
# =============================================================================

set -euo pipefail

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

REPODB_FILE="repodb.csv"
SKIP_REPODB=0
MAX_DISEASES=20        # FIX 5: default to 20 diseases for speed

for arg in "$@"; do
    case $arg in
        --repodb-file=*)  REPODB_FILE="${arg#*=}" ;;
        --skip-repodb)    SKIP_REPODB=1 ;;
        --full-repodb)    MAX_DISEASES="" ;;   # empty = no limit
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
    HAS_JQ=0
else
    log_ok "jq found: $(jq --version)"
    HAS_JQ=1
fi

for script in run_validation.py score_calibration.py false_negative_analysis.py statistical_tests.py; do
    if [[ ! -f "$script" ]]; then
        log_warn "Missing script: $script — step will be skipped."
    fi
done
log_ok "Script check complete"

START_TIME=$(date +%s)

# =============================================================================
# STEP 1: Curated Validation (n=55, v4.0 dataset)
# =============================================================================
log_section "STEP 1/5: Curated Validation (n=55 test cases, v4.0)"

python3 run_validation.py --output validation_results.json

if [[ ! -f validation_results.json ]]; then
    log_error "Step 1 FAILED — validation_results.json not created."
    exit 1
fi
log_ok "Step 1 complete — validation_results.json written"

if [[ $HAS_JQ -eq 1 ]]; then
    N_CASES=$(jq -r '.header.n_test_cases // "unknown"' validation_results.json)
    if [[ "$N_CASES" == "55" ]]; then
        log_ok "  n_test_cases = $N_CASES ✓"
    else
        log_warn "  n_test_cases = $N_CASES (expected 55 — check validation_dataset.py)"
    fi
fi

# =============================================================================
# STEP 2: Score Calibration Analysis
# =============================================================================
log_section "STEP 2/5: Score Calibration Analysis"

python3 score_calibration.py \
    --input  validation_results.json \
    --output calibration_results.json

if [[ ! -f calibration_results.json ]]; then
    log_error "Step 2 FAILED — calibration_results.json not created."
    exit 1
fi
log_ok "Step 2 complete — calibration_results.json written"

# ─── FIX 2: Check A is NEGATIVE (not B) ──────────────────────────────────────
if [[ $HAS_JQ -eq 1 ]]; then
    PLATT_A_RAW=$(jq -r '.platt_parameters.A' calibration_results.json 2>/dev/null || echo "0")
    PLATT_B_RAW=$(jq -r '.platt_parameters.B' calibration_results.json 2>/dev/null || echo "0")
    ECE=$(jq -r '.ece // "N/A"' calibration_results.json)

    A_IS_NEGATIVE=$(python3 -c "
try:
    a = float('$PLATT_A_RAW')
    print('yes' if a < 0 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

    if [[ "$A_IS_NEGATIVE" == "yes" ]]; then
        log_ok "  Platt A = $PLATT_A_RAW (negative — correct orientation) ✓"
    elif [[ "$A_IS_NEGATIVE" == "no" ]]; then
        log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_error "  CRITICAL: Platt A = $PLATT_A_RAW (POSITIVE — BUG)"
        log_error "  A must be negative: higher raw score → higher calibrated prob."
        log_error "  Fix: backend/pipeline/calibration.py — ensure _PLATT_A < 0"
        log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    else
        log_warn "  Could not parse Platt A value: '$PLATT_A_RAW'"
    fi

    log_info "  Platt B = $PLATT_B_RAW (intercept — sign is unconstrained)"
    log_info "  ECE     = $ECE"

    THRESH_RAW=$(jq -r '.classification_threshold.raw_equivalent_50 // "N/A"' calibration_results.json)
    if [[ "$THRESH_RAW" != "N/A" ]]; then
        THRESH_OK=$(python3 -c "
try:
    t = float('$THRESH_RAW')
    print('yes' if t <= 1.0 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
        if [[ "$THRESH_OK" == "no" ]]; then
            log_warn "  Calibrated threshold raw equivalent = $THRESH_RAW (> 1.0 — UNREACHABLE)"
            log_warn "  Use raw_score >= 0.20 for binary classification."
        else
            log_ok "  Calibrated threshold raw equivalent = $THRESH_RAW ✓"
        fi
    fi
fi

# =============================================================================
# STEP 3: RepoDB Benchmark (first 20 diseases by default)
# =============================================================================
log_section "STEP 3/5: RepoDB Benchmark"

if [[ $SKIP_REPODB -eq 1 ]]; then
    log_warn "RepoDB benchmark SKIPPED (--skip-repodb)."
else
    # FIX 5: Build the --max-diseases flag conditionally
    if [[ -n "$MAX_DISEASES" ]]; then
        MAX_DISEASES_FLAG="--max-diseases $MAX_DISEASES"
        log_info "RepoDB: limiting to first $MAX_DISEASES diseases (use --full-repodb to disable)"
    else
        MAX_DISEASES_FLAG=""
        log_info "RepoDB: running on ALL diseases (full benchmark — this will take hours)"
    fi

    python3 repodb_benchmark.py \
        --repodb-file "$REPODB_FILE" \
        --top-n 50 \
        --output repodb_benchmark_results.json \
        $MAX_DISEASES_FLAG

    if [[ ! -f repodb_benchmark_results.json ]]; then
        log_error "Step 3 FAILED."
        exit 1
    fi
    log_ok "Step 3 complete — repodb_benchmark_results.json written"
fi

# =============================================================================
# STEP 4: False Negative Analysis
# =============================================================================
log_section "STEP 4/5: False Negative Analysis"

python3 false_negative_analysis.py \
    --input  validation_results.json \
    --output fn_analysis.json

if [[ ! -f fn_analysis.json ]]; then
    log_error "Step 4 FAILED."
    exit 1
fi
log_ok "Step 4 complete — fn_analysis.json written"

# =============================================================================
# STEP 5: Statistical Tests (bootstrap CIs, McNemar, DeLong)
# =============================================================================
log_section "STEP 5/5: Statistical Tests (Bootstrap CI, McNemar, DeLong)"

if [[ ! -f statistical_tests.py ]]; then
    log_warn "statistical_tests.py not found — skipping."
    log_warn "Copy statistical_tests.py to this directory to enable."
else
    python3 statistical_tests.py \
        --input  validation_results.json \
        --output statistical_results.json \
        --n-bootstrap 1000 \
        --seed 42

    if [[ ! -f statistical_results.json ]]; then
        log_error "Step 5 FAILED."
        exit 1
    fi
    log_ok "Step 5 complete — statistical_results.json written"
fi

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
    echo "    ✓ repodb_benchmark_results.json  (first ${MAX_DISEASES:-ALL} diseases)"
else
    echo "    – repodb_benchmark_results.json  (skipped)"
fi
echo "    ✓ fn_analysis.json"
if [[ -f statistical_results.json ]]; then
    echo "    ✓ statistical_results.json"
fi
echo ""

if [[ $HAS_JQ -eq 0 ]]; then
    log_ok "All validation steps completed. Install jq for metric display."
    exit 0
fi

# =============================================================================
# METRICS DISPLAY
# =============================================================================
log_section "KEY METRICS SUMMARY"

echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  CURATED VALIDATION  (validation_results.json)      │"
echo "  └─────────────────────────────────────────────────────┘"

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

echo "    Sensitivity  : $SENSITIVITY   (TP=$TP / n_pos=$N_POS)"
echo "    Specificity  : $SPECIFICITY   (TN=$TN / n_neg=$N_NEG)"
echo "    Precision    : $PRECISION"
echo "    F1           : $F1"
echo "    FP           : $FP"
echo "    FN           : $FN"

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
fi

# Bootstrap CIs
if [[ -f statistical_results.json ]]; then
    echo ""
    echo "  ┌──────────────────────────────────────────────────────┐"
    echo "  │  BOOTSTRAP 95% CI  (statistical_results.json)        │"
    echo "  └──────────────────────────────────────────────────────┘"
    SENS_LO=$(jq -r '.bootstrap_ci.sensitivity.ci_lower // "N/A"' statistical_results.json)
    SENS_HI=$(jq -r '.bootstrap_ci.sensitivity.ci_upper // "N/A"' statistical_results.json)
    SPEC_LO=$(jq -r '.bootstrap_ci.specificity.ci_lower // "N/A"' statistical_results.json)
    SPEC_HI=$(jq -r '.bootstrap_ci.specificity.ci_upper // "N/A"' statistical_results.json)
    PREC_LO=$(jq -r '.bootstrap_ci.precision.ci_lower   // "N/A"' statistical_results.json)
    PREC_HI=$(jq -r '.bootstrap_ci.precision.ci_upper   // "N/A"' statistical_results.json)
    F1_LO=$(jq   -r '.bootstrap_ci.f1.ci_lower          // "N/A"' statistical_results.json)
    F1_HI=$(jq   -r '.bootstrap_ci.f1.ci_upper          // "N/A"' statistical_results.json)
    echo "    Sensitivity  : $SENSITIVITY  [95% CI: $SENS_LO – $SENS_HI]"
    echo "    Specificity  : $SPECIFICITY  [95% CI: $SPEC_LO – $SPEC_HI]"
    echo "    Precision    : $PRECISION  [95% CI: $PREC_LO – $PREC_HI]"
    echo "    F1           : $F1  [95% CI: $F1_LO – $F1_HI]"
fi

# Calibration
echo ""
echo "  ┌──────────────────────────────────────────────────────┐"
echo "  │  CALIBRATION  (calibration_results.json)             │"
echo "  └──────────────────────────────────────────────────────┘"
echo "    Platt A   : $PLATT_A_RAW  (must be negative)"
echo "    Platt B   : $PLATT_B_RAW  (no sign constraint)"
echo "    ECE       : $ECE"

# RepoDB
if [[ $SKIP_REPODB -eq 0 ]] && [[ -f repodb_benchmark_results.json ]]; then
    echo ""
    echo "  ┌──────────────────────────────────────────────────────┐"
    echo "  │  REPODB BENCHMARK  (repodb_benchmark_results.json)   │"
    echo "  └──────────────────────────────────────────────────────┘"
    N_DISEASES_TESTED=$(jq -r '.summary.n_diseases_tested // "N/A"' repodb_benchmark_results.json)
    AUC_ROC=$(jq -r '.summary.global_auc_roc // "N/A"' repodb_benchmark_results.json)
    AUC_PR=$(jq  -r '.summary.global_auc_pr  // "N/A"' repodb_benchmark_results.json)
    HIT50=$(jq   -r '.summary."hit_at_50"    // "N/A"' repodb_benchmark_results.json)
    MRR=$(jq     -r '.summary.mrr            // "N/A"' repodb_benchmark_results.json)
    echo "    Diseases  : $N_DISEASES_TESTED tested"
    echo "    AUC-ROC   : $AUC_ROC"
    echo "    AUC-PR    : $AUC_PR"
    echo "    Hit@50    : $HIT50"
    echo "    MRR       : $MRR"
fi

# =============================================================================
# PASS / FAIL CHECKLIST
# =============================================================================
log_section "PASS / FAIL CHECKLIST"
echo ""

check() {
    local label="$1"
    local result="$2"
    local detail="$3"
    case "$result" in
        pass) echo -e "  ${GREEN}[PASS]${NC} $label — $detail" ;;
        fail) echo -e "  ${RED}[FAIL]${NC} $label — $detail" ;;
        warn) echo -e "  ${YELLOW}[WARN]${NC} $label — $detail" ;;
        skip) echo -e "  ${BLUE}[SKIP]${NC} $label — $detail" ;;
    esac
}

SENS_RESULT="fail"
[[ "$SENS_OK" == "yes" ]] && SENS_RESULT="pass"
[[ "$SENS_OK" == "unknown" ]] && SENS_RESULT="warn"
check "Sensitivity ≥ 65%" "$SENS_RESULT" "$SENSITIVITY"

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

A_RESULT="fail"
[[ "$A_IS_NEGATIVE" == "yes" ]] && A_RESULT="pass"
[[ "$A_IS_NEGATIVE" == "unknown" ]] && A_RESULT="warn"
check "Platt A is negative (correct orientation)" "$A_RESULT" "A = $PLATT_A_RAW"

ECE_OK=$(python3 -c "
try:
    e = float('$ECE')
    print('yes' if e < 0.15 else 'no')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
ECE_RESULT="fail"
[[ "$ECE_OK" == "yes" ]] && ECE_RESULT="pass"
[[ "$ECE_OK" == "unknown" ]] && ECE_RESULT="warn"
check "ECE < 0.15" "$ECE_RESULT" "$ECE"

if [[ -f statistical_results.json ]]; then
    check "Bootstrap CI computed" "pass" "statistical_results.json present"
else
    check "Bootstrap CI computed" "skip" "statistical_tests.py not found"
fi

if [[ $SKIP_REPODB -eq 0 ]] && [[ -f repodb_benchmark_results.json ]]; then
    check "RepoDB benchmark" "pass" "AUC-ROC=$AUC_ROC  Hit@50=$HIT50  (n=$N_DISEASES_TESTED diseases)"
else
    check "RepoDB benchmark" "skip" "run without --skip-repodb to enable"
fi

echo ""
log_ok "Validation pipeline finished. $(date)"
echo ""