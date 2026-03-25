#!/bin/bash
# =============================================================================
# RunPod Setup — One-Click Environment Bootstrap
# =============================================================================
# Idempotent: safe to re-run. Skips already-completed steps.
#
# What this does:
#   1. Detect GPU, verify CUDA
#   2. Install Python deps (with fallbacks)
#   3. Create directory structure
#   4. Verify all imports work
#   5. Download PDEBench data (priority datasets only)
#
# Usage:
#   # On RunPod (PyTorch template, 24GB+ VRAM):
#   git clone <your-repo> /workspace/S2-conformal-uq
#   cd /workspace/S2-conformal-uq
#   bash scripts/setup_runpod.sh
#
# After setup:
#   bash scripts/oneclick.sh        # Full pipeline
#   bash scripts/oneclick.sh --dry  # Show what would run
#
# Expected time: ~15 min (mostly data download)
# =============================================================================

set -euo pipefail

# ---- Constants ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/pdebench"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
step() { echo -e "\n${GREEN}[$1]${NC} $2"; }

echo "============================================================"
echo " S2: Spectral Conformal Prediction — RunPod Setup"
echo " $(date)"
echo "============================================================"

# ================================================================
# 1. GPU & System Check
# ================================================================
step "1/5" "System check"

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    ok "GPU: ${GPU_NAME} (${GPU_MEM})"
else
    warn "No nvidia-smi. Will run on CPU (slow but works)."
fi

# Python
PYTHON_VER=$(python3 --version 2>/dev/null || python --version 2>/dev/null || echo "NOT FOUND")
ok "Python: ${PYTHON_VER}"

# PyTorch + CUDA
TORCH_INFO=$(python3 -c "
import torch
cuda = torch.cuda.is_available()
ver = torch.__version__
dev = torch.cuda.get_device_name(0) if cuda else 'CPU'
print(f'PyTorch {ver}, Device: {dev}, CUDA: {cuda}')
" 2>/dev/null || echo "PyTorch NOT installed")
ok "${TORCH_INFO}"

# Disk space
DISK_FREE=$(df -h "${PROJECT_DIR}" 2>/dev/null | awk 'NR==2{print $4}' || echo "unknown")
ok "Free disk: ${DISK_FREE}"

# ================================================================
# 2. Install Dependencies
# ================================================================
step "2/5" "Installing Python dependencies"

cd "${PROJECT_DIR}"
pip install --quiet --upgrade pip 2>/dev/null

# Core deps (fast, should all succeed)
pip install --quiet \
    "numpy>=1.24.0" \
    "scipy>=1.11.0" \
    "matplotlib>=3.7.0" \
    "h5py>=3.9.0" \
    "pandas>=2.0.0" \
    "tqdm>=4.65.0" \
    "einops>=0.7.0" \
    "pyyaml>=6.0" \
    2>&1 | tail -1 || true
ok "Core packages"

# PyTorch (skip if already installed — RunPod templates have it)
python3 -c "import torch" 2>/dev/null || {
    warn "PyTorch not found. Installing..."
    pip install --quiet "torch>=2.1.0" 2>&1 | tail -1
}
ok "PyTorch"

# Neural operator library
python3 -c "from neuralop.models import FNO" 2>/dev/null || {
    echo "  Installing neuraloperator..."
    pip install --quiet "neuraloperator>=0.3.0" 2>/dev/null || {
        warn "Pip install failed, trying from source..."
        pip install --quiet git+https://github.com/neuraloperator/neuraloperator.git@main 2>/dev/null || {
            fail "neuraloperator install failed. FNO/TFNO will use fallbacks."
        }
    }
}
python3 -c "from neuralop.models import FNO; print('  [OK] neuraloperator')" 2>/dev/null || \
    warn "neuraloperator not available — will use fallback models"

# Optional: wandb (don't fail if missing)
pip install --quiet wandb 2>/dev/null && ok "wandb (optional)" || true

# ================================================================
# 3. Create Directory Structure
# ================================================================
step "3/5" "Creating directories"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

ok "data/pdebench/"
ok "checkpoints/"
ok "results/"
ok "logs/"

# Initialize results.tsv if missing
RESULTS_TSV="${RESULTS_DIR}/results.tsv"
if [ ! -f "${RESULTS_TSV}" ]; then
    printf "timestamp\tcommit\tpde\tmodel\tscore\talpha\tcoverage\tband_width\tcal_error\tstatus\tdescription\n" \
        > "${RESULTS_TSV}"
    ok "Created results.tsv"
else
    N_ROWS=$(wc -l < "${RESULTS_TSV}")
    ok "results.tsv exists (${N_ROWS} rows)"
fi

# ================================================================
# 4. Verify All Imports
# ================================================================
step "4/5" "Verifying project modules"

python3 -c "
import sys, os
sys.path.insert(0, '${PROJECT_DIR}')
os.chdir('${PROJECT_DIR}')

errors = []

# Core modules
try:
    from src.scores import L2Score, SpectralScore, LearnedSpectralScore
    print('  [OK] src.scores (L2Score, SpectralScore, LearnedSpectralScore)')
except Exception as e:
    errors.append(f'src.scores: {e}')

try:
    from src.conformal import SplitConformalPredictor, three_way_split
    print('  [OK] src.conformal')
except Exception as e:
    errors.append(f'src.conformal: {e}')

try:
    from src.spectral_aci import SpectralACI
    print('  [OK] src.spectral_aci')
except Exception as e:
    errors.append(f'src.spectral_aci: {e}')

try:
    from src.evaluation import evaluate_coverage, evaluate_rollout
    print('  [OK] src.evaluation')
except Exception as e:
    errors.append(f'src.evaluation: {e}')

try:
    from src.data import PDE_CONFIGS, PDEBenchDataset
    print(f'  [OK] src.data — PDEs: {list(PDE_CONFIGS.keys())}')
except Exception as e:
    errors.append(f'src.data: {e}')

try:
    from src.models import get_model
    print('  [OK] src.models')
except Exception as e:
    errors.append(f'src.models: {e}')

if errors:
    print()
    for e in errors:
        print(f'  [FAIL] {e}')
    sys.exit(1)
else:
    print('  All modules OK.')
" || {
    fail "Import verification failed. Fix errors above before continuing."
    exit 1
}

# ================================================================
# 5. Download PDEBench Data
# ================================================================
step "5/5" "Downloading PDEBench data"

# DaRUS file IDs (University of Stuttgart data repository)
# Source: doi:10.18419/darus-2986
declare -A DOWNLOADS=(
    ["2D_DarcyFlow_beta1.0_Train.hdf5"]="131447"
    ["1D_Burgers_Sols_Nu0.001.hdf5"]="131421"
    ["2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"]="131433"
)

DARUS_BASE="https://darus.uni-stuttgart.de/api/access/datafile"

download_one() {
    local fname="$1"
    local fid="$2"
    local fpath="${DATA_DIR}/${fname}"

    if [ -f "${fpath}" ]; then
        local size
        size=$(du -h "${fpath}" | cut -f1)
        ok "${fname} (${size}, exists)"
        return 0
    fi

    echo "  Downloading ${fname}..."
    local url="${DARUS_BASE}/${fid}"

    if command -v wget &>/dev/null; then
        wget -q --show-progress --timeout=120 -O "${fpath}.tmp" "${url}" 2>&1
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar --connect-timeout 30 -o "${fpath}.tmp" "${url}"
    else
        fail "Neither wget nor curl found."
        return 1
    fi

    # Verify HDF5
    if python3 -c "import h5py; h5py.File('${fpath}.tmp','r').close()" 2>/dev/null; then
        mv "${fpath}.tmp" "${fpath}"
        local size
        size=$(du -h "${fpath}" | cut -f1)
        ok "${fname} (${size})"
    else
        rm -f "${fpath}.tmp"
        fail "${fname} — corrupt or incomplete download. Re-run to retry."
        return 1
    fi
}

# Phase 1 budget mode: only download Darcy Flow (~1.2GB instead of ~5.2GB)
# Set DOWNLOAD_ALL_DATASETS=1 to download everything
DOWNLOAD_FAILED=0
if [ "${DOWNLOAD_ALL_DATASETS:-0}" = "1" ]; then
    for fname in "2D_DarcyFlow_beta1.0_Train.hdf5" \
                 "1D_Burgers_Sols_Nu0.001.hdf5" \
                 "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"; do
        download_one "${fname}" "${DOWNLOADS[${fname}]}" || DOWNLOAD_FAILED=1
    done
else
    echo "  Budget mode: downloading Darcy Flow only"
    download_one "2D_DarcyFlow_beta1.0_Train.hdf5" "${DOWNLOADS[2D_DarcyFlow_beta1.0_Train.hdf5]}" || DOWNLOAD_FAILED=1
fi

if [ "${DOWNLOAD_FAILED}" -eq 1 ]; then
    warn "Some downloads failed. Non-critical: run experiments on available data."
fi

# ================================================================
# Summary
# ================================================================
echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo " Project:     ${PROJECT_DIR}"
echo " Data:        ${DATA_DIR}"
echo " Checkpoints: ${CKPT_DIR}"
echo " Results:     ${RESULTS_DIR}"
echo ""
echo " Data files:"
ls -lh "${DATA_DIR}"/*.hdf5 "${DATA_DIR}"/*.h5 2>/dev/null | awk '{print "   "$NF" ("$5")"}'|| echo "   (none)"
echo ""
echo " Next steps:"
echo "   bash scripts/oneclick.sh            # Full pipeline: train → calibrate → eval"
echo "   bash scripts/oneclick.sh --phase 2  # Skip training, just run experiments"
echo ""
