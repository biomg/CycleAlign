#!/usr/bin/env bash
# ============================================================================
# CycleAlign example runner
# ----------------------------------------------------------------------------
# Reproduces the Kowalczyk 2015 (mouse HSC, 1,434 cells, 94 genes) result on
# a small CPU configuration so that new users can verify the pipeline end-to-end.
#
#   - Conv encoder (default), 30 seeds, leaf gene order, kNN smoothing (k=8)
#   - MLP encoder variant is also run for comparison
#
# NOTE: 30 seeds × 1000 epochs is the full paper configuration and takes a
# while on CPU (~2.5 h per encoder on a single core). For a quick smoke test,
# reduce --runs to 5 and --epochs to 200.
# ============================================================================

set -euo pipefail

# --- resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SCRIPT_DIR/data"

# --- configuration ---------------------------------------------------------
RUNS="${RUNS:-30}"          # override with: RUNS=5 bash example/run_example.sh
EPOCHS="${EPOCHS:-1000}"
ENCODER="${ENCODER:-conv}"  # conv | mlp

echo "=============================================================="
echo " CycleAlign example — Kowalczyk 2015 (mouse HSC)"
echo "   data    : $DATA_DIR"
echo "   encoder : $ENCODER"
echo "   runs    : $RUNS  epochs : $EPOCHS"
echo "=============================================================="

python "$REPO_DIR/Multi-seed_phase_alignment_conv_reg.py" \
    --data-dir "$DATA_DIR" \
    --genes "$DATA_DIR/panel.txt" \
    --runs "$RUNS" \
    --epochs "$EPOCHS" \
    --encoder "$ENCODER" \
    --gene-order leaf \
    --smooth 8 \
    --outdir "$SCRIPT_DIR/output_${ENCODER}_r${RUNS}"

echo "Done. Results written to $SCRIPT_DIR/output_${ENCODER}_r${RUNS}/"
