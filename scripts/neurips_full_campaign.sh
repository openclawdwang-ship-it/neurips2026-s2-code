#!/bin/bash
# =============================================================================
# NeurIPS Full Campaign: Publication-quality S2 experiments
# =============================================================================
# Produces: Table 1 (multi-model), Table 2 (ablation), all figures data
# Runtime: ~4-5h on RTX A4000/3090
# =============================================================================
set -euo pipefail

PYTHON=$(command -v python3 || command -v python)
echo "Using: $PYTHON"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"

PROJECT_DIR="${PROJECT_DIR:-.}"
cd "${PROJECT_DIR}"

$PYTHON -m pip install -q neuraloperator 2>/dev/null || true

SEEDS=(42 123 456 789 1024)
RESULTS_DIR="./results"
LOG_DIR="./logs"
CKPT_DIR="./checkpoints"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

echo "============================================="
echo "NeurIPS S2 Full Campaign"
echo "Seeds: ${SEEDS[*]}"
echo "============================================="

# train.py internally saves to: {save_dir}/{model}_{pde}_{resolution}/best.pt
# calibrate.py looks at:        {checkpoint_dir}/{model}_{pde}_{resolution}/best.pt
# So we use --save_dir/--checkpoint_dir = {base}/seed{SEED}
# and --output_dir = {results}/seed{SEED}

# =============================================================================
# PHASE 1: Train all models (Darcy 64x64, 5 seeds each)
# =============================================================================
echo ""
echo "=== PHASE 1: Training ==="

for SEED in "${SEEDS[@]}"; do
    SEED_CKPT="${CKPT_DIR}/seed${SEED}"
    for MODEL in fno tfno deeponet; do
        CKPT_FILE="${SEED_CKPT}/${MODEL}_darcy_64/best.pt"
        if [ -f "${CKPT_FILE}" ]; then
            echo "[SKIP] ${MODEL} seed=${SEED} exists"
            continue
        fi
        LR=1e-3
        [ "$MODEL" = "deeponet" ] && LR=5e-4
        echo "[TRAIN] ${MODEL} darcy 64 seed=${SEED}..."
        $PYTHON scripts/train.py \
            --model $MODEL --pde darcy --resolution 64 \
            --data_source neuralop \
            --epochs 500 --lr $LR --weight_decay 1e-4 \
            --n_train 800 --n_cal 100 --n_test 100 \
            --seed $SEED \
            --save_dir "${SEED_CKPT}" \
            2>&1 | tee "${LOG_DIR}/train_${MODEL}_seed${SEED}.log"
    done
done

# =============================================================================
# PHASE 2: Calibrate all models × all scores (Table 1)
# =============================================================================
echo ""
echo "=== PHASE 2: Calibration (Table 1) ==="

for SEED in "${SEEDS[@]}"; do
    SEED_CKPT="${CKPT_DIR}/seed${SEED}"
    SEED_OUT="${RESULTS_DIR}/seed${SEED}"
    for MODEL in fno tfno deeponet; do
        CKPT_FILE="${SEED_CKPT}/${MODEL}_darcy_64/best.pt"
        RESULT_FILE="${SEED_OUT}/${MODEL}_darcy_64/calibration_results.json"
        if [ ! -f "${CKPT_FILE}" ]; then
            echo "[SKIP] No checkpoint for ${MODEL} seed=${SEED}"
            continue
        fi
        if [ -f "${RESULT_FILE}" ]; then
            echo "[SKIP] ${MODEL} seed=${SEED} results exist"
            continue
        fi
        echo "[CAL] ${MODEL} seed=${SEED}..."
        $PYTHON scripts/calibrate.py \
            --model $MODEL --pde darcy --resolution 64 \
            --data_source neuralop \
            --checkpoint_dir "${SEED_CKPT}" \
            --output_dir "${SEED_OUT}" \
            --alpha 0.1 \
            --mc_dropout --mc_n_passes 20 \
            --n_train 800 --n_cal 100 --n_test 100 \
            --seed $SEED \
            2>&1 | tee "${LOG_DIR}/cal_${MODEL}_seed${SEED}.log"
    done
done

# =============================================================================
# PHASE 3: Alpha ablation (Table 2a) - FNO only, 5 seeds
# =============================================================================
echo ""
echo "=== PHASE 3: Alpha Ablation ==="

for ALPHA in 0.05 0.10 0.15 0.20; do
    for SEED in "${SEEDS[@]}"; do
        SEED_CKPT="${CKPT_DIR}/seed${SEED}"
        SEED_OUT="${RESULTS_DIR}/alpha${ALPHA}/seed${SEED}"
        RESULT_FILE="${SEED_OUT}/fno_darcy_64/calibration_results.json"
        if [ -f "${RESULT_FILE}" ]; then
            echo "[SKIP] alpha=${ALPHA} seed=${SEED}"
            continue
        fi
        echo "[ABL] alpha=${ALPHA} seed=${SEED}..."
        $PYTHON scripts/calibrate.py \
            --model fno --pde darcy --resolution 64 \
            --data_source neuralop \
            --checkpoint_dir "${SEED_CKPT}" \
            --output_dir "${SEED_OUT}" \
            --alpha $ALPHA \
            --no_mc_dropout \
            --n_train 800 --n_cal 100 --n_test 100 \
            --seed $SEED \
            2>&1 | tee "${LOG_DIR}/abl_alpha${ALPHA}_seed${SEED}.log"
    done
done

# =============================================================================
# PHASE 4: Calibration set size ablation (Table 2b)
# =============================================================================
echo ""
echo "=== PHASE 4: n_cal Ablation ==="

for NCAL in 50 100 200 500; do
    NTRAIN=$((1000 - NCAL - 100))
    for SEED in "${SEEDS[@]}"; do
        NCAL_CKPT="${CKPT_DIR}/ncal${NCAL}/seed${SEED}"
        NCAL_OUT="${RESULTS_DIR}/ncal${NCAL}/seed${SEED}"
        CKPT_FILE="${NCAL_CKPT}/fno_darcy_64/best.pt"
        RESULT_FILE="${NCAL_OUT}/fno_darcy_64/calibration_results.json"

        if [ ! -f "${CKPT_FILE}" ]; then
            echo "[TRAIN] fno ncal=${NCAL} seed=${SEED}..."
            $PYTHON scripts/train.py \
                --model fno --pde darcy --resolution 64 \
                --data_source neuralop \
                --epochs 500 --lr 1e-3 --weight_decay 1e-4 \
                --n_train $NTRAIN --n_cal $NCAL --n_test 100 \
                --seed $SEED \
                --save_dir "${NCAL_CKPT}" \
                2>&1 | tee "${LOG_DIR}/train_ncal${NCAL}_seed${SEED}.log"
        fi

        if [ -f "${RESULT_FILE}" ]; then
            echo "[SKIP] ncal=${NCAL} seed=${SEED}"
            continue
        fi
        echo "[CAL] ncal=${NCAL} seed=${SEED}..."
        $PYTHON scripts/calibrate.py \
            --model fno --pde darcy --resolution 64 \
            --data_source neuralop \
            --checkpoint_dir "${NCAL_CKPT}" \
            --output_dir "${NCAL_OUT}" \
            --alpha 0.1 \
            --no_mc_dropout \
            --n_train $NTRAIN --n_cal $NCAL --n_test 100 \
            --seed $SEED \
            2>&1 | tee "${LOG_DIR}/cal_ncal${NCAL}_seed${SEED}.log"
    done
done

echo ""
echo "============================================="
echo "CAMPAIGN COMPLETE"
echo "============================================="
echo "Results in: ${RESULTS_DIR}/"
find "${RESULTS_DIR}" -name "calibration_results.json" | wc -l
echo "result files generated."
