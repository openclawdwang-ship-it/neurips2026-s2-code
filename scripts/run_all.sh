#!/bin/bash
# =============================================================================
# Master Experiment Orchestrator
# =============================================================================
# End-to-end: Train → Calibrate → Rollout → Ablations → Figures
#
# Follows PRINCIPLES.md:
#   - Fixed time budgets per experiment
#   - Coverage >= 1-α-0.02 hard constraint
#   - Results logged to results.tsv (append-only)
#   - Crash recovery: skip and continue
#
# Usage:
#   bash scripts/run_all.sh              # Full campaign (all phases)
#   bash scripts/run_all.sh 1            # Phase 1: Training only
#   bash scripts/run_all.sh 2            # Phase 2: Static conformal
#   bash scripts/run_all.sh 3            # Phase 3: Rollout (Spectral ACI)
#   bash scripts/run_all.sh 4            # Phase 4: Ablations
#   bash scripts/run_all.sh 5            # Phase 5: Figures
#   bash scripts/run_all.sh 2 3 5        # Multiple phases
#   bash scripts/run_all.sh --dry        # Show what would run
#
# Expected times (RTX 3090):
#   Phase 1: ~8-10 hours   (train 6 models, 500 epochs each)
#   Phase 2: ~5 minutes    (7 score functions × 4 configs)
#   Phase 3: ~5 minutes    (2 rollout configs)
#   Phase 4: ~10 minutes   (ablations)
#   Phase 5: ~30 seconds   (figures)
#   Total:   ~10 hours
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="./data/pdebench"
CKPT_DIR="./checkpoints"
RESULTS_DIR="./results"
LOG_DIR="./logs"
ALPHA=0.1

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${CKPT_DIR}"

# ---- Parse args ----
DRY_RUN=0
PHASES=()
for arg in "$@"; do
    if [ "${arg}" = "--dry" ] || [ "${arg}" = "-n" ]; then
        DRY_RUN=1
    elif [[ "${arg}" =~ ^[1-5]$ ]]; then
        PHASES+=("${arg}")
    fi
done

# Default: all phases
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(1 2 3 4 5)
fi

# ---- Colors ----
G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
B='\033[1;34m'
N='\033[0m'

phase_header() {
    echo ""
    echo -e "${B}=============================================${N}"
    echo -e "${B} PHASE $1: $2${N}"
    echo -e "${B}=============================================${N}"
    echo ""
}

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
            echo -e "  ${R}[TIMEOUT]${N} Exceeded ${budget}s budget. Skipping."
        else
            echo -e "  ${R}[CRASH]${N} Exit code ${exit_code}. Continuing."
        fi
        return 0  # Don't abort the pipeline
    }
    echo -e "  ${G}[DONE]${N}"
}

# ---- Helper: check if checkpoint exists ----
has_ckpt() {
    [ -f "${CKPT_DIR}/$1/best.pt" ]
}

# ---- Helper: check if data file exists ----
has_data() {
    [ -f "${DATA_DIR}/$1" ]
}

# =============================================
echo -e "${G}=============================================${N}"
echo -e "${G} Spectral Conformal Prediction — Experiment Campaign${N}"
echo -e "${G} $(date)${N}"
echo -e "${G} Phases: ${PHASES[*]}${N}"
echo -e "${G} Alpha: ${ALPHA}${N}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo -e "${Y} *** DRY RUN — nothing will execute ***${N}"
fi
echo -e "${G}=============================================${N}"

# ==============================================================
# PHASE 1: TRAINING
# ==============================================================
if printf '%s\n' "${PHASES[@]}" | grep -qx "1"; then
    phase_header "1" "Model Training (one-time)"

    # Priority 1: FNO on Darcy (dev target)
    if ! has_ckpt "fno_darcy_64"; then
        echo -e "${G}[1/6]${N} FNO on Darcy 64×64"
        if ! has_data "2D_DarcyFlow_beta1.0_Train.hdf5"; then
            echo -e "  ${R}[SKIP]${N} Darcy data not found"
        else
            run_with_budget 5400 "${LOG_DIR}/train_fno_darcy_64.log" \
                "python scripts/train.py --model fno --pde darcy --resolution 64 --epochs 500 --lr 1e-3 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}"
        fi
    else
        echo -e "${G}[1/6]${N} FNO/Darcy — ${Y}checkpoint exists, skipping${N}"
    fi

    # Priority 2: TFNO on Darcy (cross-arch)
    if ! has_ckpt "tfno_darcy_64"; then
        echo -e "${G}[2/6]${N} TFNO on Darcy 64×64"
        run_with_budget 5400 "${LOG_DIR}/train_tfno_darcy_64.log" \
            "python scripts/train.py --model tfno --pde darcy --resolution 64 --epochs 500 --lr 1e-3 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}"
    else
        echo -e "${G}[2/6]${N} TFNO/Darcy — ${Y}skipping${N}"
    fi

    # Priority 3: DeepONet on Darcy (negative control)
    if ! has_ckpt "deeponet_darcy_64"; then
        echo -e "${G}[3/6]${N} DeepONet on Darcy 64×64"
        run_with_budget 7200 "${LOG_DIR}/train_deeponet_darcy_64.log" \
            "python scripts/train.py --model deeponet --pde darcy --resolution 64 --epochs 500 --lr 5e-4 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}"
    else
        echo -e "${G}[3/6]${N} DeepONet/Darcy — ${Y}skipping${N}"
    fi

    # Priority 4: FNO on Burgers 1D
    if ! has_ckpt "fno_burgers_128"; then
        echo -e "${G}[4/6]${N} FNO on Burgers 1D (res 128)"
        if has_data "1D_Burgers_Sols_Nu0.001.hdf5"; then
            run_with_budget 5400 "${LOG_DIR}/train_fno_burgers_128.log" \
                "python scripts/train.py --model fno --pde burgers --resolution 128 --epochs 500 --lr 1e-3 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}"
        else
            echo -e "  ${R}[SKIP]${N} Burgers data not found"
        fi
    else
        echo -e "${G}[4/6]${N} FNO/Burgers — ${Y}skipping${N}"
    fi

    # Priority 5-6: Navier-Stokes (for rollout)
    NS_FILE="2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    if has_data "${NS_FILE}"; then
        if ! has_ckpt "fno_navier_stokes_64"; then
            echo -e "${G}[5/6]${N} FNO on Navier-Stokes 64×64"
            run_with_budget 5400 "${LOG_DIR}/train_fno_ns_64.log" \
                "python scripts/train.py --model fno --pde navier_stokes --resolution 64 --epochs 500 --lr 1e-3 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}"
        else
            echo -e "${G}[5/6]${N} FNO/NS — ${Y}skipping${N}"
        fi

        if ! has_ckpt "tfno_navier_stokes_64"; then
            echo -e "${G}[6/6]${N} TFNO on Navier-Stokes 64×64"
            run_with_budget 5400 "${LOG_DIR}/train_tfno_ns_64.log" \
                "python scripts/train.py --model tfno --pde navier_stokes --resolution 64 --epochs 500 --lr 1e-3 --data_dir ${DATA_DIR} --save_dir ${CKPT_DIR}"
        else
            echo -e "${G}[6/6]${N} TFNO/NS — ${Y}skipping${N}"
        fi
    else
        echo -e "${G}[5-6/6]${N} ${R}NS data not found. Skipping.${N}"
    fi

    echo ""
    echo "Checkpoints:"
    ls -la "${CKPT_DIR}"/*/best.pt 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
fi

# ==============================================================
# PHASE 2: STATIC CONFORMAL EXPERIMENTS
# ==============================================================
if printf '%s\n' "${PHASES[@]}" | grep -qx "2"; then
    phase_header "2" "Static Conformal Experiments"

    ALL_SCORES="l2 spectral_uniform spectral_sobolev_1 spectral_sobolev_2 spectral_inverse spectral_learned cqr_simplified"

    # FNO on Darcy — full comparison (hero experiment)
    if has_ckpt "fno_darcy_64"; then
        echo -e "${G}[Hero]${N} FNO/Darcy — all scores"
        run_with_budget 180 "${LOG_DIR}/cal_fno_darcy_64.log" \
            "python scripts/calibrate.py --model fno --pde darcy --resolution 64 --scores ${ALL_SCORES} --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    fi

    # TFNO on Darcy — cross-architecture
    if has_ckpt "tfno_darcy_64"; then
        echo -e "${G}[Cross-arch]${N} TFNO/Darcy"
        run_with_budget 120 "${LOG_DIR}/cal_tfno_darcy_64.log" \
            "python scripts/calibrate.py --model tfno --pde darcy --resolution 64 --scores l2 spectral_sobolev_1 spectral_sobolev_2 cqr_simplified --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    fi

    # DeepONet — negative control
    if has_ckpt "deeponet_darcy_64"; then
        echo -e "${G}[Neg. control]${N} DeepONet/Darcy"
        run_with_budget 120 "${LOG_DIR}/cal_deeponet_darcy_64.log" \
            "python scripts/calibrate.py --model deeponet --pde darcy --resolution 64 --scores l2 spectral_sobolev_1 spectral_sobolev_2 --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    fi

    # FNO on Burgers 1D
    if has_ckpt "fno_burgers_128"; then
        echo -e "${G}[1D]${N} FNO/Burgers"
        run_with_budget 120 "${LOG_DIR}/cal_fno_burgers_128.log" \
            "python scripts/calibrate.py --model fno --pde burgers --resolution 128 --scores l2 spectral_sobolev_1 cqr_simplified --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    fi
fi

# ==============================================================
# PHASE 3: ROLLOUT EXPERIMENTS (Spectral ACI)
# ==============================================================
if printf '%s\n' "${PHASES[@]}" | grep -qx "3"; then
    phase_header "3" "Rollout Experiments (Spectral ACI)"

    # FNO on NS — the key C2 experiment
    if has_ckpt "fno_navier_stokes_64"; then
        echo -e "${G}[Key C2]${N} FNO/NS rollout (20 steps)"
        run_with_budget 300 "${LOG_DIR}/rollout_fno_ns_64.log" \
            "python scripts/rollout_eval.py --model fno --pde navier_stokes --resolution 64 --n_steps 20 --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    else
        echo -e "  ${R}[SKIP]${N} No FNO/NS checkpoint"
    fi

    # TFNO on NS
    if has_ckpt "tfno_navier_stokes_64"; then
        echo -e "${G}[Cross-arch]${N} TFNO/NS rollout (20 steps)"
        run_with_budget 300 "${LOG_DIR}/rollout_tfno_ns_64.log" \
            "python scripts/rollout_eval.py --model tfno --pde navier_stokes --resolution 64 --n_steps 20 --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    fi

    # FNO on Burgers rollout (1D validation)
    if has_ckpt "fno_burgers_128"; then
        echo -e "${G}[1D]${N} FNO/Burgers rollout (20 steps)"
        run_with_budget 300 "${LOG_DIR}/rollout_fno_burgers_128.log" \
            "python scripts/rollout_eval.py --model fno --pde burgers --resolution 128 --n_steps 20 --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}"
    fi
fi

# ==============================================================
# PHASE 4: ABLATIONS
# ==============================================================
if printf '%s\n' "${PHASES[@]}" | grep -qx "4"; then
    phase_header "4" "Ablations"

    # Ablation 1: Calibration set size
    if has_ckpt "fno_darcy_64"; then
        echo -e "${G}[Ablation]${N} Calibration set size"
        for n_cal in 50 100 200 500; do
            echo "  n_cal=${n_cal}..."
            run_with_budget 120 "${LOG_DIR}/ablation_ncal_${n_cal}.log" \
                "python scripts/calibrate.py --model fno --pde darcy --resolution 64 --scores l2 spectral_sobolev_1 --n_cal ${n_cal} --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}/ablation_ncal_${n_cal}"
        done
    fi

    # Ablation 2: ACI learning rate
    if has_ckpt "fno_navier_stokes_64"; then
        echo -e "${G}[Ablation]${N} ACI gamma sensitivity"
        for gamma in 0.005 0.01 0.02 0.05; do
            echo "  gamma=${gamma}..."
            run_with_budget 300 "${LOG_DIR}/ablation_gamma_${gamma}.log" \
                "python scripts/rollout_eval.py --model fno --pde navier_stokes --resolution 64 --n_steps 20 --aci_gamma ${gamma} --data_dir ${DATA_DIR} --checkpoint_dir ${CKPT_DIR} --output_dir ${RESULTS_DIR}/ablation_gamma_${gamma}"
        done
    fi
fi

# ==============================================================
# PHASE 5: GENERATE FIGURES
# ==============================================================
if printf '%s\n' "${PHASES[@]}" | grep -qx "5"; then
    phase_header "5" "Generating Figures"

    run_with_budget 60 "${LOG_DIR}/plot_results.log" \
        "python scripts/plot_results.py --results_dir ${RESULTS_DIR} --output_dir ./figures"

    echo ""
    echo "Figures:"
    ls -la ./figures/*.pdf 2>/dev/null | awk '{print "  " $NF}' || echo "  (none generated)"
fi

# ==============================================================
# SUMMARY
# ==============================================================
echo ""
echo -e "${G}=============================================${N}"
echo -e "${G} EXPERIMENT CAMPAIGN COMPLETE${N}"
echo -e "${G} $(date)${N}"
echo -e "${G}=============================================${N}"
echo ""

# Print results.tsv if it has data
RESULTS_TSV="${RESULTS_DIR}/results.tsv"
if [ -f "${RESULTS_TSV}" ]; then
    N_ROWS=$(( $(wc -l < "${RESULTS_TSV}") - 1 ))
    echo "Results: ${N_ROWS} experiments logged"
    echo ""
    if [ "${N_ROWS}" -gt 0 ] && [ "${N_ROWS}" -lt 50 ]; then
        column -t -s $'\t' "${RESULTS_TSV}" 2>/dev/null || cat "${RESULTS_TSV}"
    elif [ "${N_ROWS}" -ge 50 ]; then
        echo "(Showing last 20 rows)"
        tail -20 "${RESULTS_TSV}" | column -t -s $'\t' 2>/dev/null || tail -20 "${RESULTS_TSV}"
    fi
fi

echo ""
echo "Files:"
echo "  Results:     ${RESULTS_DIR}/"
echo "  Figures:     ./figures/"
echo "  Logs:        ${LOG_DIR}/"
echo "  Checkpoints: ${CKPT_DIR}/"
