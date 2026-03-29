#!/bin/bash
# =============================================================================
# Model Training Script — Train Once, Then Iterate on CP
# =============================================================================
# Following PRINCIPLES.md: models are trained ONCE and frozen.
# All subsequent experiments modify ONLY the conformal prediction layer.
#
# Budget: ~30 min per model-PDE pair on RTX 3090
# Total:  ~3-4 hours for all priority configurations
# =============================================================================

set -euo pipefail

# Use python3 (RunPod images may not have 'python' symlink)
PYTHON=$(command -v python3 || command -v python)
echo "Using: $PYTHON"

PROJECT_DIR="${PROJECT_DIR:-/workspace/S2-conformal-uq}"
cd "${PROJECT_DIR}"

DATA_DIR="./data/pdebench"
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "Model Training Campaign"
echo "============================================="
echo ""

# Check data exists
for f in "2D_DarcyFlow_beta1.0_Train.hdf5" "1D_Burgers_Sols_Nu0.001.hdf5"; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "ERROR: Missing ${f}. Run: bash scripts/download_data.sh"
        exit 1
    fi
done

# ---- Phase 1: Priority models (Week 1) ----
echo "=== Phase 1: Priority models ==="
echo ""

# FNO on Darcy (primary development target)
echo "[1/6] FNO on Darcy 64x64..."
if [ -f "${SAVE_DIR}/fno_darcy_64/best.pt" ]; then
    echo "  Checkpoint exists. Skipping."
else
    $PYTHON scripts/train.py \
        --model fno --pde darcy --resolution 64 \
        --epochs 100 --lr 1e-3 \
        --n_train 800 --n_cal 100 --n_test 100 \
        --data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" \
        2>&1 | tee "${LOG_DIR}/train_fno_darcy_64.log"
fi
echo ""

# TFNO on Darcy (cheaper, for cross-architecture comparison)
echo "[2/6] TFNO on Darcy 64x64..."
if [ -f "${SAVE_DIR}/tfno_darcy_64/best.pt" ]; then
    echo "  Checkpoint exists. Skipping."
else
    $PYTHON scripts/train.py \
        --model tfno --pde darcy --resolution 64 \
        --epochs 100 --lr 1e-3 \
        --n_train 800 --n_cal 100 --n_test 100 \
        --data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" \
        2>&1 | tee "${LOG_DIR}/train_tfno_darcy_64.log"
fi
echo ""

# DeepONet on Darcy (negative control for spectral bias hypothesis)
echo "[3/6] DeepONet on Darcy 64x64..."
if [ -f "${SAVE_DIR}/deeponet_darcy_64/best.pt" ]; then
    echo "  Checkpoint exists. Skipping."
else
    $PYTHON scripts/train.py \
        --model deeponet --pde darcy --resolution 64 \
        --epochs 200 --lr 5e-4 \
        --n_train 800 --n_cal 100 --n_test 100 \
        --data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" \
        2>&1 | tee "${LOG_DIR}/train_deeponet_darcy_64.log"
fi
echo ""

# FNO on Burgers 1D (fast 1D validation)
echo "[4/6] FNO on Burgers 1D (resolution 128)..."
if [ -f "${SAVE_DIR}/fno_burgers_128/best.pt" ]; then
    echo "  Checkpoint exists. Skipping."
else
    $PYTHON scripts/train.py \
        --model fno --pde burgers --resolution 128 \
        --epochs 100 --lr 1e-3 \
        --n_train 800 --n_cal 100 --n_test 100 \
        --data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" \
        2>&1 | tee "${LOG_DIR}/train_fno_burgers_128.log"
fi
echo ""

# ---- Phase 2: Rollout models (Week 2) ----
echo "=== Phase 2: Rollout models ==="

# FNO on Navier-Stokes (for Spectral ACI rollout experiments)
NS_FILE="2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
if [ -f "${DATA_DIR}/${NS_FILE}" ]; then
    echo "[5/6] FNO on Navier-Stokes 64x64..."
    if [ -f "${SAVE_DIR}/fno_navier_stokes_64/best.pt" ]; then
        echo "  Checkpoint exists. Skipping."
    else
        $PYTHON scripts/train.py \
            --model fno --pde navier_stokes --resolution 64 \
            --epochs 100 --lr 1e-3 \
            --n_train 800 --n_cal 100 --n_test 100 \
            --data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" \
            2>&1 | tee "${LOG_DIR}/train_fno_ns_64.log"
    fi

    echo ""
    echo "[6/6] TFNO on Navier-Stokes 64x64..."
    if [ -f "${SAVE_DIR}/tfno_navier_stokes_64/best.pt" ]; then
        echo "  Checkpoint exists. Skipping."
    else
        $PYTHON scripts/train.py \
            --model tfno --pde navier_stokes --resolution 64 \
            --epochs 100 --lr 1e-3 \
            --n_train 800 --n_cal 100 --n_test 100 \
            --data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" \
            2>&1 | tee "${LOG_DIR}/train_tfno_ns_64.log"
    fi
else
    echo "[5-6/6] Navier-Stokes data not found. Skipping NS training."
    echo "  Download first: bash scripts/download_data.sh"
fi

echo ""
echo "============================================="
echo "Training complete!"
echo "============================================="
echo ""
echo "Checkpoints:"
ls -la "${SAVE_DIR}"/*/best.pt 2>/dev/null || echo "  (none found)"
echo ""
echo "Next: bash scripts/run_all.sh"
