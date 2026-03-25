# S2: Neural Operator Conformal Prediction

## Core Idea
Apply Conformal Prediction to Neural Operators (PDE solvers) — two hot communities with ZERO intersection.

## Why It Works
- Neural operators (FNO, DeepONet) have NO calibrated uncertainty quantification
- Conformal Prediction provides distribution-free coverage guarantees
- First principled UQ for PDE solvers

## Experiments
- **Benchmarks**: Navier-Stokes, Darcy Flow (from PINNacle benchmark)
- **Models**: FNO, DeepONet, U-NO
- **Baselines**: MC Dropout, Deep Ensemble, vanilla conformal on standard NNs
- **Metrics**: Coverage, interval width, calibration error

## Compute
- GPU: <4GB VRAM
- Cost: ~$11
- Timeline: 6 weeks

## Key Papers
- FNO (Li et al., 2021)
- DeepONet (Lu et al., 2021)
- PINNacle benchmark (NeurIPS 2024)
- Conformal Prediction tutorial (Angelopoulos & Bates, 2023)
- ECT/iCT for consistency (ICLR 2025)

## TODO
1. Literature deep-dive: existing UQ for neural operators
2. Implement FNO + DeepONet on Navier-Stokes & Darcy Flow
3. Implement Conformal Prediction wrapper (split conformal, CQR)
4. Design functional conformal scores (point-wise vs field-level)
5. Run experiments, ablations
6. Write paper
