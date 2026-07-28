#!/usr/bin/env bash
# run_h2o_all.sh: run the H2O AutoML pipeline sequentially across all datasets.
# Requires Java 8+ on PATH (check with: java -version).
# Usage: bash run_h2o_all.sh [--skip-running]
#
# --skip-running  skip the first config (h2o_electricity) if it is already
#                 running in the background from a previous session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The scripts/ directory is one level below the project root. Always run from the root.
cd "$(dirname "$SCRIPT_DIR")"

LOG_DIR="logs/h2o_runs"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="$LOG_DIR/summary.txt"
> "$SUMMARY_FILE"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

bar() {
  printf "${CYAN}══════════════════════════════════════════════════════════════${RESET}\n"
}

# ── dataset list ──────────────────────────────────────────────────────────────
# Each entry: "config_file  display_name  approx_minutes"
declare -a DATASETS=(
  "configs/h2o/h2o_electricity.yaml    Electricity   30"
  "configs/h2o/h2o_m4_monthly.yaml     M4_Monthly    20"
  "configs/h2o/h2o_etth1.yaml          ETTh1         30"
  "configs/h2o/h2o_etth2.yaml          ETTh2         30"
  "configs/h2o/h2o_ettm1.yaml          ETTm1         40"
  "configs/h2o/h2o_cable_demand.yaml   Cable_Demand  20"
)

SKIP_FIRST=false
if [[ "${1:-}" == "--skip-running" ]]; then
  SKIP_FIRST=true
fi

TOTAL=${#DATASETS[@]}
DONE=0
FAILED=0
START_ALL=$(date +%s)

bar
printf "${BOLD}  H2O AutoML full dataset sweep${RESET}\n"
printf "  Started : $(date '+%Y-%m-%d %H:%M:%S')\n"
printf "  Datasets: $TOTAL\n"
bar
echo ""

# ── helper: human-readable duration ──────────────────────────────────────────
elapsed() {
  local secs=$1
  printf "%dm%02ds" $((secs/60)) $((secs%60))
}

# ── main loop ─────────────────────────────────────────────────────────────────
IDX=0
for entry in "${DATASETS[@]}"; do
  read -r config name approx_min <<< "$entry"
  IDX=$((IDX + 1))

  if [[ "$SKIP_FIRST" == "true" && "$IDX" -eq 1 ]]; then
    printf "${YELLOW}  [%d/%d] %-20s SKIPPED (already running)${RESET}\n" \
      "$IDX" "$TOTAL" "$name"
    echo "$name  SKIPPED" >> "$SUMMARY_FILE"
    continue
  fi

  LOG_FILE="$LOG_DIR/${name}.log"

  bar
  printf "${BOLD}  [%d/%d] %s${RESET}\n" "$IDX" "$TOTAL" "$name"
  printf "  Config : %s\n" "$config"
  printf "  Log    : %s\n" "$LOG_FILE"
  printf "  Est.   : ~%s min\n" "$approx_min"
  printf "  Start  : $(date '+%H:%M:%S')\n"
  bar

  T0=$(date +%s)

  # Run and tee output so it shows live in terminal AND is saved to log
  if python src/h2o_pipeline.py --config "$config" 2>&1 | tee "$LOG_FILE"; then
    T1=$(date +%s)
    DUR=$((T1 - T0))
    printf "\n${GREEN}  ✓ %-20s  done in $(elapsed $DUR)${RESET}\n\n" "$name"
    echo "$name  OK  $(elapsed $DUR)" >> "$SUMMARY_FILE"
    DONE=$((DONE + 1))
  else
    T1=$(date +%s)
    DUR=$((T1 - T0))
    printf "\n${RED}  ✗ %-20s  FAILED after $(elapsed $DUR), see %s${RESET}\n\n" \
      "$name" "$LOG_FILE"
    echo "$name  FAILED  $(elapsed $DUR)" >> "$SUMMARY_FILE"
    FAILED=$((FAILED + 1))
  fi
done

# ── final summary ─────────────────────────────────────────────────────────────
END_ALL=$(date +%s)
TOTAL_DUR=$((END_ALL - START_ALL))

bar
printf "${BOLD}  SUMMARY${RESET}\n"
printf "  Finished : $(date '+%Y-%m-%d %H:%M:%S')\n"
printf "  Total    : $(elapsed $TOTAL_DUR)\n"
printf "  Passed   : ${GREEN}%d${RESET}   Failed: ${RED}%d${RESET}\n" "$DONE" "$FAILED"
bar
echo ""
cat "$SUMMARY_FILE"
echo ""

if [[ "$FAILED" -gt 0 ]]; then
  exit 1
fi
