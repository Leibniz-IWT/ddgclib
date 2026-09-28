#!/bin/bash
# Run 3D dynamic cases on the GPU server.
#
# Usage:
#   bash cases_dynamic/run_3d_cases_gpu.sh [--quick]
#
# The --quick flag runs with n_refine=1 for fast validation.
# Without it, runs with default refinement (n_refine=2).
#
# Prerequisites:
#   pip install -e .   (from project root)
#   conda activate ddg

set -euo pipefail
cd "$(dirname "$0")/.."

QUICK="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="cases_dynamic/logs/${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo "=============================================="
echo "  3D Dynamic Cases — GPU Server Run"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Logs: ${LOGDIR}/"
if [ "$QUICK" = "--quick" ]; then
    echo "  Mode: QUICK (n_refine=1)"
    REFINE_FLAG="--n-refine 1"
    N_STEPS_FLAG="--n-steps 500"
else
    echo "  Mode: FULL (default refinement)"
    REFINE_FLAG=""
    N_STEPS_FLAG=""
fi
echo "=============================================="

# --- 1. Hydrostatic 3D ---
echo ""
echo "[1/2] Running Hydrostatic_3D..."
if [ "$QUICK" = "--quick" ]; then
    # Hydrostatic_3D doesn't have CLI args, create a patched temp copy
    TMPSCRIPT=$(mktemp /tmp/hydrostatic_3d_XXXX.py)
    sed 's/n_refine = 2/n_refine = 1/;s/n_trav = 15/n_trav = 5/' \
        cases_dynamic/Hydrostatic_column/Hydrostatic_3D.py > "$TMPSCRIPT"
    python "$TMPSCRIPT" 2>&1 | tee "${LOGDIR}/hydrostatic_3d.log"
    rm -f "$TMPSCRIPT"
else
    python -m cases_dynamic.Hydrostatic_column.Hydrostatic_3D \
        2>&1 | tee "${LOGDIR}/hydrostatic_3d.log"
fi
echo "  -> Hydrostatic_3D complete. Figures in cases_dynamic/Hydrostatic_column/fig/"

# --- 2. Hagen-Poiseuille 3D ---
echo ""
echo "[2/2] Running Hagen_Poiseuile_3D..."
python cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py \
    ${REFINE_FLAG} ${N_STEPS_FLAG} \
    2>&1 | tee "${LOGDIR}/hagen_poiseuille_3d.log"
echo "  -> Hagen_Poiseuile_3D complete. Figures in cases_dynamic/Hagen_Poiseuile_3D/fig/"

echo ""
echo "=============================================="
echo "  All 3D cases complete."
echo "  Logs saved to: ${LOGDIR}/"
echo "=============================================="
