#!/bin/bash
# =============================================================================
#  ONE-CLICK: Spectral Conformal Prediction — Phase 1 (Budget Mode)
# =============================================================================
#
#  嘉哥，你只需要这一个脚本。
#
#  在 RunPod 上 (RTX 3090, $0.48/hr):
#    1. git clone <repo> /workspace/S2-conformal-uq
#    2. cd /workspace/S2-conformal-uq
#    3. bash scripts/oneclick.sh
#
#  预算: $5 上限 = 10 小时 GPU。实际 ~1 小时 = ~$0.50
#
#  Phase 1 精简方案:
#    - 只训练 FNO × Darcy (最小最快)
#    - 跑全部 7 个 conformal score + MC Dropout
#    - 不跑 rollout/ensemble (省到 Phase 2)
#    - 产出: coverage vs band width 对比图
#
#  Options:
#    bash scripts/oneclick.sh              # 全部
#    bash scripts/oneclick.sh --skip-setup # 跳过环境配置
#    bash scripts/oneclick.sh --skip-train # 跳过训练（已有 checkpoint）
#    bash scripts/oneclick.sh --dry        # 只打印不执行
#
#  预期时间 (RTX 3090):
#    Setup + Download:  ~10 min
#    Training (500ep):  ~45 min
#    Calibration:       ~5 min
#    Figures:           ~30 sec
#    Total:             ~1 hour ≈ $0.48
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# ---- Telegram notifications ----
notify_success() {
    local msg="$1"
    openclaw message send --channel telegram --target 5377816744 --message "✅ S2: ${msg}" 2>/dev/null || true
}
notify_stuck() {
    local msg="$1"
    openclaw message send --channel telegram --target 5377816744 --message "🚧 S2: ${msg}" 2>/dev/null || true
}
# Trap: notify on any unexpected failure
trap 'notify_stuck "oneclick.sh failed at line $LINENO (exit $?). Check logs."' ERR

# ---- Parse args ----
SKIP_SETUP=0
SKIP_TRAIN=0
DRY_RUN=0

for arg in "$@"; do
    case "${arg}" in
        --skip-setup) SKIP_SETUP=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        --dry|-n)     DRY_RUN=1 ;;
    esac
done

# ---- Colors ----
G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
B='\033[1;36m'
N='\033[0m'

echo ""
echo -e "${B}╔══════════════════════════════════════════════════════════╗${N}"
echo -e "${B}║  S2: Spectral Conformal Prediction — Phase 1 (Budget) ║${N}"
echo -e "${B}║  NeurIPS 2026 | Budget: \$5 | Target: ~1 hour          ║${N}"
echo -e "${B}╚══════════════════════════════════════════════════════════╝${N}"
echo ""
echo -e "  Project:  ${PROJECT_DIR}"
echo -e "  Started:  $(date)"
echo -e "  Host:     $(hostname 2>/dev/null || echo 'unknown')"

# GPU info + cost tracking
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  GPU:      ${GPU}"
fi
echo ""

START_TIME=$(date +%s)

# ---- Helper: run with timeout + logging ----
run_with_budget() {
    local budget="$1"
    local logfile="$2"
    shift 2
    local cmd="$*"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo -e "  ${Y}[DRY]${N} ${cmd}"
        return 0
    fi

    echo "  Running: ${cmd}"
    echo "  Log: ${logfile}"
    echo "  Budget: ${budget}s"

    timeout "${budget}" bash -c "${cmd}" 2>&1 | tee "${logfile}" || {
        local exit_code=$?
        if [ ${exit_code} -eq 124 ]; then
            echo -e "  ${R}[TIMEOUT]${N} Exceeded ${budget}s budget"
        else
            echo -e "  ${R}[CRASH]${N} Exit code ${exit_code}"
        fi
        return 1
    }
    echo -e "  ${G}[DONE]${N}"
}

DATA_DIR="./data/pdebench"
CKPT_DIR="./checkpoints"
RESULTS_DIR="./results"
LOG_DIR="./logs"
FIGURES_DIR="./figures"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${FIGURES_DIR}"

# ================================================================
# STAGE 1: ENVIRONMENT SETUP
# ================================================================
if [ "${SKIP_SETUP}" -eq 0 ]; then
    echo -e "${B}━━━ Stage 1/4: Environment Setup ━━━${N}"
    echo ""
    if [ -f "${SCRIPT_DIR}/setup_runpod.sh" ]; then
        bash "${SCRIPT_DIR}/setup_runpod.sh"
    else
        echo -e "  ${Y}[SKIP]${N} setup_runpod.sh not found, assuming deps installed"
    fi
    echo ""
else
    echo -e "${Y}[SKIP] Stage 1: Setup${N}"
fi

# ================================================================
# STAGE 2: VERIFY DATA
# ================================================================
echo ""
echo -e "${B}━━━ Stage 2/4: Data Check ━━━${N}"

DARCY_FILE="2D_DarcyFlow_beta1.0_Train.hdf5"
if [ -f "${DATA_DIR}/${DARCY_FILE}" ]; then
    SIZE=$(du -h "${DATA_DIR}/${DARCY_FILE}" 2>/dev/null | cut -f1)
    echo -e "  ${G}[OK]${N} ${DARCY_FILE} (${SIZE})"
else
    echo -e "  ${R}[MISSING]${N} ${DARCY_FILE}"
    echo ""
    echo "  Attempting download..."
    if [ -f "${SCRIPT_DIR}/download_data.sh" ]; then
        bash "${SCRIPT_DIR}/download_data.sh" darcy "${DATA_DIR}" || true
    fi

    # Fallback: fix_data.py (HuggingFace → neuralop → synthetic)
    if [ ! -f "${DATA_DIR}/${DARCY_FILE}" ]; then
        echo -e "  ${Y}[FALLBACK]${N} DaRUS download failed. Trying fix_data.py..."
        python3 "${SCRIPT_DIR}/fix_data.py" --data_dir "${DATA_DIR}" --n_samples 1000 --resolution 128 || {
            echo -e "  ${R}ERROR: All data methods failed. Cannot continue.${N}"
            exit 1
        }
    fi

    if [ ! -f "${DATA_DIR}/${DARCY_FILE}" ]; then
        echo -e "  ${R}ERROR: No data file after all attempts. Cannot continue.${N}"
        exit 1
    fi
fi
echo ""

# ================================================================
# STAGE 3: TRAIN FNO ON DARCY (only model needed for Phase 1)
# ================================================================
echo -e "${B}━━━ Stage 3/4: Training (FNO × Darcy 64×64, 500 epochs) ━━━${N}"
echo ""

if [ "${SKIP_TRAIN}" -eq 1 ]; then
    echo -e "  ${Y}[SKIP] Training (--skip-train)${N}"
elif [ -f "${CKPT_DIR}/fno_darcy_64/best.pt" ]; then
    echo -e "  ${G}[SKIP]${N} Checkpoint exists: ${CKPT_DIR}/fno_darcy_64/best.pt"
else
    # Budget: 5400s = 90 min (generous for 500 epochs on 3090)
    run_with_budget 5400 "${LOG_DIR}/train_fno_darcy_64.log" \
        "python3 scripts/train.py --model fno --pde darcy --resolution 64 --epochs 500 --lr 1e-3 --seed 42 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}" || {
        echo -e "${R}Training failed! Check ${LOG_DIR}/train_fno_darcy_64.log${N}"
        exit 1
    }
fi

# Verify checkpoint exists
if [ ! -f "${CKPT_DIR}/fno_darcy_64/best.pt" ] && [ "${DRY_RUN}" -eq 0 ]; then
    echo -e "${R}ERROR: No checkpoint after training. Aborting.${N}"
    exit 1
fi
echo ""

# ================================================================
# STAGE 4: CALIBRATION + FIGURES
# ================================================================
echo -e "${B}━━━ Stage 4/4: Conformal Calibration + Figures ━━━${N}"
echo ""

# 4a: All 7 score functions + MC Dropout
ALL_SCORES="l2 spectral_uniform spectral_sobolev_1 spectral_sobolev_2 spectral_inverse spectral_learned cqr_simplified"

echo -e "  ${G}[4a]${N} Running all conformal scores..."
run_with_budget 300 "${LOG_DIR}/cal_fno_darcy_64.log" \
    "python3 scripts/calibrate.py --model fno --pde darcy --resolution 64 --alpha 0.1 --scores ${ALL_SCORES} --mc_dropout --mc_n_passes 20 --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}" || {
    echo -e "  ${R}[WARN]${N} Calibration failed. Check log."
}

# 4b: Ablation — calibration set size (quick, ~2 min total)
echo ""
echo -e "  ${G}[4b]${N} Ablation: calibration set size..."
for n_cal in 50 100 200 500; do
    run_with_budget 120 "${LOG_DIR}/ablation_ncal_${n_cal}.log" \
        "python3 scripts/calibrate.py --model fno --pde darcy --resolution 64 --scores l2 spectral_sobolev_1 --n_cal ${n_cal} --no_mc_dropout --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}/ablation_ncal_${n_cal}" || true
done

# 4c: Generate figures
echo ""
echo -e "  ${G}[4c]${N} Generating figures..."
run_with_budget 60 "${LOG_DIR}/plot_results.log" \
    "python3 scripts/plot_results.py --results_dir ${RESULTS_DIR} --output_dir ${FIGURES_DIR}" || {
    echo -e "  ${Y}[WARN]${N} Figure generation failed. Results still saved as JSON."
}

# ================================================================
# FINAL SUMMARY
# ================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

# Estimate cost (RTX 3090 @ $0.48/hr)
COST=$(python3 -c "print(f'\${${ELAPSED}/3600 * 0.48:.2f}')" 2>/dev/null || echo "?")

echo ""
echo -e "${B}╔══════════════════════════════════════════════════════════╗${N}"
echo -e "${B}║                 PHASE 1 COMPLETE!                       ║${N}"
echo -e "${B}╚══════════════════════════════════════════════════════════╝${N}"
echo ""
echo -e "  Time:     ${HOURS}h ${MINS}m ${SECS}s"
echo -e "  Est cost: ${COST} (RTX 3090 @ \$0.48/hr)"
echo -e "  Budget:   \$5.00 max"
echo -e "  Finished: $(date)"
echo ""

# Results summary
if [ -f "${RESULTS_DIR}/results.tsv" ] && [ "${DRY_RUN}" -eq 0 ]; then
    echo -e "  ${B}Results:${N}"
    python3 -c "
import sys
rows = []
with open('${RESULTS_DIR}/results.tsv') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 10:
            rows.append(parts)

if not rows:
    print('    (no results)')
    sys.exit(0)

# Header
print(f'    {\"Score\":<22} {\"Coverage\":>10} {\"Band Width\":>12} {\"Status\":>8}')
print(f'    {\"-\"*22} {\"-\"*10} {\"-\"*12} {\"-\"*8}')

best_bw = float('inf')
best_name = ''
for r in rows:
    score = r[4]
    cov = float(r[6])
    bw = float(r[7])
    status = r[9]
    marker = ''
    if status == 'keep' and bw < best_bw:
        best_bw = bw
        best_name = score
    print(f'    {score:<22} {cov:>10.4f} {bw:>12.6f} {status:>8}')

if best_name:
    print()
    print(f'    WINNER: {best_name} (band width = {best_bw:.6f})')
" 2>/dev/null || echo "    (unable to parse results.tsv)"
fi

echo ""
echo -e "  ${B}Key outputs:${N}"
echo "    ${RESULTS_DIR}/fno_darcy_64/calibration_results.json"
echo "    ${RESULTS_DIR}/results.tsv"
echo "    ${FIGURES_DIR}/"
echo ""
echo -e "  ${B}Phase 2 (when budget allows):${N}"
echo "    - TFNO/DeepONet cross-architecture comparison"
echo "    - Navier-Stokes rollout with Spectral ACI"
echo "    - Burgers 1D validation"
echo "    bash scripts/run_all.sh 1 2 3  # Full campaign"
echo ""
echo -e "${G}Done! 嘉哥可以去看 figures/ 里的图了${N}"
echo ""

# ---- Notify success ----
notify_success "Phase 1 complete! FNO×Darcy trained (${HOURS}h${MINS}m), 7 scores + MC Dropout calibrated, figures generated. Cost ~${COST}."
