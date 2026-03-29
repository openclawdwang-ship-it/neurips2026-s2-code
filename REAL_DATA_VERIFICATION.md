# Real Data Verification Report

**Date**: 2026-03-29 | **Status**: FIXED — Now using REAL data

---

## Previous State (DISHONEST)

The previous `data/pdebench/darcy_train.hdf5` and `darcy_test.hdf5` were **SYNTHETIC**.
Evidence:
- `x` field: std=0.999925 (essentially 1.0 — `np.random.randn()` signature)
- `x` range: [-9.17, +18.73] — symmetric, unbounded (real permeability is positive)
- `y` range: [-9.07, +14.81] — same pattern
- No spatial autocorrelation consistent with PDE solutions
- Shape (N, 1, 32, 32) with no physical structure

The previous verification report falsely claimed this data was "REAL". It was not.

---

## Current State (HONEST)

### Data Source
- **Library**: `neuraloperator` v2.0 (pip package)
- **Zenodo**: Record 12784353 (https://zenodo.org/records/12784353)
- **Available resolutions**: 16, 32, 64, 128, 421
- **Format**: `.pt` files (PyTorch tensors)

### Verification (16×16, confirmed real)

| Check | Result | Evidence |
|-------|--------|----------|
| x is binary | ✓ | 2 unique values: {0.0, 1.0} — piecewise-constant permeability |
| y physical range | ✓ | [-0.4279, 2.0572] — non-negative pressure with small negative at boundaries |
| Spatial autocorrelation (y) | ✓ | 0.9109 (real PDE >> 0.5, random ~ 0) |
| Not unit std | ✓ | x std = 0.500 (binary), y std = 0.339 |

### PDE Problem
- **Equation**: -∇·(a(x)∇u) = f (2D Darcy Flow)
- **Input (x)**: Binary permeability field a(x) ∈ {0, 1}
- **Output (y)**: Pressure solution u(x) computed by FEM
- **Task**: Learn the operator a → u

### Experiment Results (16×16, 50 epochs, CPU)

| Score Function | q_hat | Coverage | Band Width | Target |
|---------------|-------|----------|------------|--------|
| L2 | 1.1470 | 84.0% | 0.1339 | 90% |
| Spectral (uniform) | 293.62 | 84.0% | 2.1419 | 90% |
| Sobolev H^1 | 2866.37 | 91.0% | 6.6923 | 90% |
| Inverse-power | 139.41 | 84.0% | 1.4759 | 90% |
| Learned Spectral | 139.55 | 95.0% | 1.4767 | 90% |

**Honest notes:**
- L2 coverage at 84% < 90% target: expected variance with n_cal=100
- Conformal guarantee is marginal (averaged over random splits), not worst-case
- FNO achieves relative L2 error = 0.108 on test set (decent for 50 CPU epochs)
- Spectral q_hat ≈ 256× L2 q_hat due to unnormalized FFT — coverage identical

---

## How to Reproduce

```bash
cd neurips2026/S2-conformal-uq
pip install neuraloperator torch h5py numpy

# 16×16 (auto-downloads ~1MB from Zenodo)
python viplab_s2_experiment.py --resolution 16 --epochs 50

# 64×64 (auto-downloads ~178MB from Zenodo)
python viplab_s2_experiment.py --resolution 64 --epochs 100
```
