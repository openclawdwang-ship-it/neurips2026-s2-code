#!/bin/bash
# =============================================================================
# RunPod one-shot: Train FNO on Darcy 64x64 + full calibration
# Budget: ~30 min on RTX 3090/4090. Produces publishable results.
# =============================================================================
set -euo pipefail

# Fix: RunPod images may only have python3
PYTHON=$(command -v python3 || command -v python)
echo "Using: $PYTHON"

PROJECT_DIR="${PROJECT_DIR:-/workspace/S2-conformal-uq}"
cd "${PROJECT_DIR}"

# Ensure deps
$PYTHON -m pip install -q neuraloperator torch h5py numpy 2>/dev/null || true

echo "============================================="
echo "S2 Campaign: FNO Darcy 64x64 (neuralop data)"
echo "============================================="
echo ""

# ---- Phase 1: Train FNO ----
if [ -f "checkpoints/fno_darcy_64/best.pt" ]; then
    echo "[SKIP] FNO checkpoint exists."
else
    echo "[1/3] Training FNO on Darcy 64x64 (500 epochs)..."
    $PYTHON scripts/train.py \
        --model fno --pde darcy --resolution 64 \
        --data_source neuralop \
        --epochs 500 --lr 1e-3 --weight_decay 1e-4 \
        --n_train 800 --n_cal 100 --n_test 100 \
        --save_dir ./checkpoints \
        2>&1 | tee logs/train_fno_darcy_64.log
fi
echo ""

# ---- Phase 2: Train TFNO (cross-architecture comparison) ----
if [ -f "checkpoints/tfno_darcy_64/best.pt" ]; then
    echo "[SKIP] TFNO checkpoint exists."
else
    echo "[2/3] Training TFNO on Darcy 64x64 (500 epochs)..."
    $PYTHON scripts/train.py \
        --model tfno --pde darcy --resolution 64 \
        --data_source neuralop \
        --epochs 500 --lr 1e-3 --weight_decay 1e-4 \
        --n_train 800 --n_cal 100 --n_test 100 \
        --save_dir ./checkpoints \
        2>&1 | tee logs/train_tfno_darcy_64.log
fi
echo ""

# ---- Phase 3: Full calibration (all scores + MC dropout) ----
echo "[3/3] Calibrating ALL scores..."

for MODEL in fno tfno; do
    if [ -f "checkpoints/${MODEL}_darcy_64/best.pt" ]; then
        echo ""
        echo "--- Calibrating $MODEL ---"
        $PYTHON scripts/calibrate.py \
            --model $MODEL --pde darcy --resolution 64 \
            --data_source neuralop \
            --checkpoint_dir ./checkpoints \
            --output_dir ./results \
            --alpha 0.1 \
            --mc_dropout --mc_n_passes 20 \
            --n_train 800 --n_cal 100 --n_test 100 \
            2>&1 | tee logs/calibrate_${MODEL}_darcy_64.log
    else
        echo "WARN: No checkpoint for $MODEL, skipping calibration."
    fi
done

echo ""
echo "============================================="
echo "Done! Results in ./results/"
echo "============================================="
echo ""

# Show summary
if [ -f "results/fno_darcy_64/calibration_results.json" ]; then
    echo "=== FNO Results ==="
    cat results/fno_darcy_64/calibration_results.json
fi
echo ""
if [ -f "results/tfno_darcy_64/calibration_results.json" ]; then
    echo "=== TFNO Results ==="
    cat results/tfno_darcy_64/calibration_results.json
fi

echo ""
echo "To pull results locally:"
echo "  runpodctl receive (or scp)"
echo "  Files: results/ checkpoints/ logs/"
