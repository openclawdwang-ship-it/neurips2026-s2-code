#!/bin/bash
# =============================================================================
# PDEBench Data Download Script
# =============================================================================
# Downloads ONLY the datasets we need (not the full 50GB+ PDEBench).
#
# Priority order (matching PRINCIPLES.md experiment queue):
#   1. 2D Darcy Flow    — static PDE, primary development target (~1.2 GB)
#   2. 2D Navier-Stokes — time-dependent, for rollout experiments (~3.5 GB)
#   3. 1D Burgers       — 1D baseline, fast experiments (~0.5 GB)
#
# Optional (Week 3+):
#   4. 1D Diffusion-Reaction (~0.3 GB)
#   5. 2D Shallow Water (~2.0 GB)
#
# Total required: ~5.2 GB (priority 1-3)
# Total optional: ~7.5 GB (all 5)
#
# Source: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
# =============================================================================

set -euo pipefail

# Usage: bash download_data.sh [darcy|burgers|ns|all] [data_dir]
WHICH="${1:-all}"
# If first arg looks like a path, treat it as data_dir (backward compat)
if [[ "${WHICH}" == *"/"* ]] || [[ "${WHICH}" == "."* ]]; then
    DATA_DIR="${WHICH}"
    WHICH="all"
else
    DATA_DIR="${2:-./data/pdebench}"
fi
mkdir -p "${DATA_DIR}"

echo "============================================="
echo "PDEBench Data Download"
echo "Target: ${DATA_DIR}"
echo "Dataset: ${WHICH}"
echo "============================================="
echo ""

# PDEBench uses DaRUS (Data Repository of the University of Stuttgart)
# Direct download links from the dataset DOI
DARUS_BASE="https://darus.uni-stuttgart.de/api/access/datafile"

# File IDs from PDEBench DaRUS repository
# These are the specific file IDs for each dataset
# Source: doi:10.18419/darus-2986
declare -A FILE_IDS=(
    # Priority 1: 2D Darcy Flow
    ["2D_DarcyFlow_beta1.0_Train.hdf5"]="131447"
    # Priority 2: 2D Compressible Navier-Stokes (Rand, M=0.1)
    ["2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"]="131433"
    # Priority 3: 1D Burgers
    ["1D_Burgers_Sols_Nu0.001.hdf5"]="131421"
)

declare -A FILE_IDS_OPTIONAL=(
    # Priority 4: 1D Diffusion-Reaction
    ["1D_diff-react_NA_NA.h5"]="131425"
    # Priority 5: 2D Shallow Water (radial dam break)
    ["2D_rdb_NA_NA.h5"]="131451"
)

download_file() {
    local filename="$1"
    local file_id="$2"
    local filepath="${DATA_DIR}/${filename}"

    if [ -f "${filepath}" ]; then
        echo "  [SKIP] ${filename} already exists"
        return 0
    fi

    echo "  [DOWNLOADING] ${filename}..."
    local url="${DARUS_BASE}/${file_id}"

    # Try wget first, fall back to curl
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${filepath}.tmp" "${url}" && \
            mv "${filepath}.tmp" "${filepath}"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${filepath}.tmp" "${url}" && \
            mv "${filepath}.tmp" "${filepath}"
    else
        echo "  [ERROR] Neither wget nor curl found. Install one."
        return 1
    fi

    # Verify it's a valid HDF5 file
    python -c "import h5py; f=h5py.File('${filepath}','r'); print(f'  [OK] {filename}: {list(f.keys())}, {len(f[list(f.keys())[0]])} samples'); f.close()" 2>/dev/null || {
        echo "  [WARNING] ${filename} may be corrupt. Verify manually."
    }
}

# --- Selective or full download ---
case "${WHICH}" in
    darcy)
        echo "--- Downloading Darcy Flow only ---"
        download_file "2D_DarcyFlow_beta1.0_Train.hdf5" "${FILE_IDS[2D_DarcyFlow_beta1.0_Train.hdf5]}"
        ;;
    burgers)
        echo "--- Downloading Burgers only ---"
        download_file "1D_Burgers_Sols_Nu0.001.hdf5" "${FILE_IDS[1D_Burgers_Sols_Nu0.001.hdf5]}"
        ;;
    ns|navier_stokes)
        echo "--- Downloading Navier-Stokes only ---"
        download_file "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5" "${FILE_IDS[2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5]}"
        ;;
    all|*)
        echo "--- Priority 1-3: Required datasets ---"
        for filename in "2D_DarcyFlow_beta1.0_Train.hdf5" \
                        "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5" \
                        "1D_Burgers_Sols_Nu0.001.hdf5"; do
            file_id="${FILE_IDS[${filename}]}"
            download_file "${filename}" "${file_id}"
        done
        ;;
esac

echo ""

# --- Optional downloads ---
if [ "${DOWNLOAD_ALL:-0}" = "1" ]; then
    echo "--- Priority 4-5: Optional datasets ---"
    for filename in "1D_diff-react_NA_NA.h5" "2D_rdb_NA_NA.h5"; do
        file_id="${FILE_IDS_OPTIONAL[${filename}]}"
        download_file "${filename}" "${file_id}"
    done
fi

echo ""
echo "============================================="
echo "Download complete!"
echo "============================================="
echo ""

# Summary
echo "Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"/*.hdf5 "${DATA_DIR}"/*.h5 2>/dev/null || echo "  (none found)"
echo ""
echo "Total disk usage:"
du -sh "${DATA_DIR}"
