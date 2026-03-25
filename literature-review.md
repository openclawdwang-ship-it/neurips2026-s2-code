# Literature Review: Conformal Prediction for Neural Operators

## ⚠️ Critical Finding: This is NOT a Zero-Intersection Space

Our initial assumption that "conformal prediction + neural operators = zero papers" is **FALSE**. At least 6 papers directly combine conformal prediction with neural operators (DeepONet, general operator learning). The paper must be repositioned.

---

## 1. Direct Competitors: Conformal Prediction + Neural Operators

### 1.1 Conformalized-DeepONet (Moya et al., Feb 2024)
- **Venue**: Physica D (journal), arXiv 2402.15406
- **Method**: Split conformal prediction wrapping Bayesian/Probabilistic/Quantile DeepONet
- **Coverage**: Yes — finite-sample marginal coverage
- **PDEs**: Viscous Burgers, diffusion-reaction, ODE systems
- **Limitations**: DeepONet only (not FNO); pointwise scores only; simple 1D PDEs
- **Citation**: ~15-20 (growing)

### 1.2 Calibrated UQ for Operator Learning (Ma, Azizzadenesheli, Anandkumar, Feb 2024)
- **Venue**: arXiv 2402.01960 — **from the Caltech/NVIDIA neuraloperator group**
- **Method**: Risk-Controlling Quantile Neural Operator
- **Coverage**: Yes — functional coverage rate guarantee (expected % of domain points within band)
- **PDEs**: 2D Darcy flow, 3D car surface pressure
- **Limitations**: Coverage averaged over domain (not uniform pointwise); limited PDE diversity
- **Significance**: From the SAME group as the neuraloperator library

### 1.3 Split CP in Function Space with Neural Operators (Millard et al., Sep 2025)
- **Venue**: arXiv 2509.04623
- **Method**: Function-space split CP with discretization-aware coverage lifting
- **Coverage**: Finite-sample in discretized space + asymptotic convergence to function-space coverage
- **PDEs**: Super-resolution tasks, resolution transfer
- **Limitations**: Asymptotic (not finite-sample) for infinite-dim; focuses on resolution transfer

### 1.4 CMCO — Conformalized Monte Carlo Operator (Kobayashi et al., Jul 2025)
- **Venue**: arXiv 2507.11574
- **Method**: MC Dropout + split conformal prediction in DeepONet
- **Coverage**: Near-nominal empirical coverage
- **PDEs**: Turbulent flow, elastoplastic deformation, cosmic radiation
- **Limitations**: Relies on MC Dropout as base; DeepONet only

### 1.5 LSCI — Locally Adaptive Conformal for Operators (Harris et al., Jul 2025)
- **Venue**: arXiv 2507.20975 (CMU)
- **Method**: Local Sliced Conformal Inference — depth-based conformity scores + localization
- **Coverage**: Finite-sample marginal + asymptotic conditional validity
- **PDEs**: Air quality, energy demand, weather (applied, not standard PDE benchmarks)
- **Limitations**: Not tested on Navier-Stokes/Darcy; computational overhead from localization

### 1.6 CoNBONet (Garg & Chakraborty, Mar 2026)
- **Venue**: arXiv 2603.21678 — **VERY RECENT (days ago)**
- **Method**: Bayesian DeepONet + spiking neural computation + split conformal
- **Coverage**: Yes
- **PDEs**: Time-dependent reliability analysis of nonlinear dynamical systems
- **Limitations**: Niche application; unusual architecture

---

## 2. Related: Conformal Prediction + PINNs

### 2.1 Conformalized PINNs (May 2024)
- **Venue**: arXiv 2405.08111
- **PDEs**: Logistic growth ODE, Buckley-Leverett PDE
- **Note**: PINNs solve ONE instance, not operator learning (family of solutions)

### 2.2 Conformal Prediction Framework for PINNs (Sep 2025)
- **Venue**: arXiv 2509.13717
- **Method**: Local conformal quantile estimation for spatially adaptive bands
- **PDEs**: Harmonic oscillator, Poisson, Allen-Cahn, Helmholtz

---

## 3. Non-Conformal UQ for Neural Operators

### 3.1 DINOZAUR — Bayesian FNO (NeurIPS 2025 Spotlight)
- **Venue**: NeurIPS 2025, arXiv 2508.00643
- **Method**: Diffusion multiplier with priors over time parameters for FNO
- **Coverage**: No distribution-free guarantees (Bayesian credible intervals)
- **Significance**: **NeurIPS 2025 Spotlight** — shows UQ for neural operators is valued

### 3.2 LUNO — Linearized Neural Operator (ICML 2025)
- **Venue**: ICML 2025, arXiv 2406.05072
- **Method**: Linearized Laplace → function-valued Gaussian Processes
- **Coverage**: No (Gaussian assumption)

### 3.3 Probabilistic Neural Operators (TMLR 2025)
- **Venue**: TMLR 2025, arXiv 2502.12902
- **Method**: Proper scoring rules for generative neural operators
- **Coverage**: No formal guarantee; reports CRPS and calibration

### 3.4 VB-DeepONet (2023), alpha-VI DeepONet (2025)
- Variational Bayesian approaches for DeepONet
- No coverage guarantees

### 3.5 DiffFNO (CVPR 2025)
- Diffusion model + FNO for implicit UQ via sampling
- Focus on super-resolution, not PDE solving

### 3.6 NeuralUQ Library (SIAM Review 2024)
- Comprehensive library for UQ in neural DEs and operators
- Supports MC Dropout, ensembles, Bayesian for PINNs/DeepONet

### 1.7 CP-PRE — Physics-Informed Conformal Prediction (Gopakumar et al., ICML 2025)
- **Venue**: ICML 2025 — **CRITICAL COMPETITOR**
- **Method**: Uses PDE residuals (L1 norm via convolutional finite-difference stencils) as conformal scores with coverage guarantees
- **Coverage**: Yes — distribution-free via split conformal with residual-based scores
- **PDEs**: Various including 20-step auto-regressive rollouts
- **Limitations**: Does NOT connect to PDE stability theory; does NOT exploit spectral structure; uses static per-step recalibration (not adaptive ACI)
- **Significance**: Scoops "physics-informed conformal scores" as a contribution. Also does temporal rollouts but with static recalibration, not ACI's adaptive learning rate.

### 1.8 CP for Dynamic Systems with Neural Operators (Dec 2024)
- **Venue**: arXiv 2412.10459
- **Method**: Evaluates FNO, UNO, TFNO with standard split CP for UQ comparison
- **Coverage**: Yes (standard split conformal)
- **PDEs**: Dynamic systems
- **Limitations**: Standard L2 nonconformity scores only; no spectral or physics-informed scores
- **Significance**: Closest to our C3 (benchmark), but limited to standard scores

---

## 4. Additional Related Work (from deep literature search)

### 4.1 Functional Conformal Prediction Theory
- **Diquigiovanni et al. (2021-2022)**: Foundational work on simultaneous conformal prediction bands for functional data. Defines proper coverage hierarchy (marginal → pointwise → simultaneous).
- **CONSIGN (Spatial Segmentation CP)**: Spatial segmentation approach for conformal prediction on spatially-varying outputs.
- **OT-based Multivariate CP (ICML 2025)**: Optimal-transport based conformal prediction for multivariate outputs — related to function-valued coverage.

### 4.2 Spectral Bias in Neural Operators
- **Toward a Better Understanding of FNO from a Spectral Perspective** (arXiv 2404.07200, 2024): Introduces NMSE spectrum decomposition and SpecB-FNO. Characterizes "Fourier parameterization bias" — FNO learns dominant frequencies first. Defines frequency-resolved error metrics e_{F,p} structurally similar to our spectral score but uses them only as diagnostics, NOT as conformal scores.
- **Spectral Bias in Physics-Informed and Operator Learning** (arXiv 2602.19265, ICLR 2026): Systematic analysis across architectures, activations, optimizers. Proposes frequency-resolved error metrics. No UQ or conformal prediction.
- **Fourier Features and Operator Learning**: Multiple papers show spectral structure matters for accuracy — but NONE exploit this for uncertainty quantification.

---

## 5. Identified Gaps — What Has NOT Been Done (Updated)

### Gap 1: Comprehensive Cross-Architecture Conformal Benchmark ⭐⭐⭐
No paper systematically compares conformal prediction across **FNO, DeepONet, U-NO, and Transolver** on the **same** set of diverse PDEs. Each paper tests on 1-2 architectures, 2-3 PDEs.

### Gap 2: Adaptive Temporal Conformal for Auto-Regressive PDE Rollouts ⭐⭐⭐⭐
CP-PRE (ICML 2025) does 20-step rollouts but with **static per-step recalibration** — no adaptive learning rate, no per-frequency tracking. **Adaptive Conformal Inference** (ACI, Gibbs & Candes 2021) with online learning rate adaptation for sequential PDE prediction is **unstudied**. Combining ACI with spectral scores for per-frequency coverage drift tracking is completely novel.

### Gap 3: Conformal vs Bayesian vs Ensemble Head-to-Head ⭐⭐⭐
No paper provides a systematic comparison of ALL UQ methods (conformal, Bayesian DINOZAUR-style, ensemble, MC Dropout, probabilistic scoring rules) on the SAME architectures and SAME benchmarks.

### Gap 4: Tightness Optimization ⭐⭐
CQR/conditional conformal methods for the tightest bands while maintaining coverage have not been systematically applied.

### ~~Gap 5: Physics-Informed Conformal Scores~~ ⭐ (PARTIALLY SCOOPED)
~~Using the PDE residual as a conformity score is unexplored.~~ **CP-PRE (ICML 2025) already does this.** Remaining sub-gap: connecting PDE residuals to stability theory (Lax-Richtmyer) for formal tightness bounds — but this alone is thin for a contribution.

### Gap 6: Spectral Conformal Prediction ⭐⭐⭐⭐⭐ (NEW — STRONGEST)
**No paper defines conformal scores in the Fourier/spectral domain.** FNO operates natively in Fourier space, and its errors have known spectral structure (low-freq learned first, high-freq errors dominate). A spectral nonconformity score s(u) = Σ_k w_k |û_true(k) - û_pred(k)|² with frequency-dependent weights w_k would:
- Naturally connect to Sobolev norms (w_k = (1+|k|²)^s gives H^s norm)
- Exploit FNO's spectral bias for tighter bands
- Unify L2/H1/adaptive scores under one framework
- Be the first conformal method aware of the operator's internal representation

**This is the most novel angle available.**

---

## 6. Benchmarks and Tools

### PDEBench (NeurIPS 2022) — PRIMARY BENCHMARK
- **GitHub**: https://github.com/pdebench/PDEBench
- **PDEs**: 1D Advection, 1D Burgers, 1D/2D Diffusion-Reaction, 1D Diffusion-Sorption, 1D/2D/3D Compressible Navier-Stokes, 2D Darcy Flow, 2D Shallow Water
- **Data**: HDF5 format, pre-generated multi-parameter families
- **Pre-trained models**: Yes (FNO, U-Net) — doi:10.18419/darus-2987
- **Baselines**: FNO, U-Net, PINN

### neuraloperator Library (NVIDIA/Caltech) — PRIMARY CODEBASE
- **GitHub**: https://github.com/neuraloperator/neuraloperator
- **Models**: FNO, TFNO, SFNO, GNO, GINO, UQNO, LocalFNO
- **Datasets**: Darcy Flow (bundled), Navier-Stokes (download), Car-CFD
- **Note**: Includes UQNO (Uncertainty-Quantifying Neural Operator) — directly relevant

### PINNacle (NeurIPS 2024) — NOT for operator learning
- Benchmarks PINNs, not neural operators. Skip for this paper.

### Conformal Prediction Libraries
- **TorchCP** (JMLR 2024): Best for our use — native PyTorch, GPU-accelerated
- **MAPIE**: scikit-learn API, good for prototyping
- **crepes**: Lightweight but CPU-only

---

## 7. GPU/Cost Estimates

| Model | Dataset | Training Time (RTX 3090) | VRAM |
|-------|---------|--------------------------|------|
| FNO (2D Darcy 64×64) | PDEBench | 1-2 hours | ~4 GB |
| FNO (2D Navier-Stokes 64×64) | PDEBench | 4-8 hours | ~4 GB |
| TFNO (2D Darcy) | PDEBench | 0.5-1 hour | ~2 GB |
| DeepONet (2D Darcy) | Custom | 2-4 hours | ~4 GB |
| Conformal calibration | Any | <10 min | ~1 GB |

**Total for 3 PDEs × 3 models + ablations**: ~50-80 GPU-hours on RTX 3090
**Cost**: ~$11-18 at $0.22/hr

---

## 8. Key References

### Conformal Prediction Theory
- Vovk, Gammerman, Shafer (2005) — Algorithmic Learning in a Random World
- Romano, Patterson, Candes (2019) — CQR (Conformalized Quantile Regression)
- Angelopoulos & Bates (2023) — Conformal Prediction Tutorial
- Gibbs & Candes (2021) — Adaptive Conformal Inference (ACI)
- Bates, Angelopoulos et al. (2021) — Distribution-Free Risk Control

### Neural Operators
- Li et al. (2021) — Fourier Neural Operator (FNO)
- Lu et al. (2021) — DeepONet
- Rahman et al. (2023) — U-NO
- Wu et al. (2024) — Transolver

### Functional/Multivariate Conformal Prediction
- Diquigiovanni et al. (2021-2022) — Simultaneous conformal prediction bands
- CP-PRE (Gopakumar et al., ICML 2025) — Physics-informed conformal scores using PDE residuals
- OT-based Multivariate CP (ICML 2025) — Optimal-transport conformal for multivariate outputs

### UQ Baselines
- DINOZAUR (NeurIPS 2025 Spotlight) — Bayesian FNO
- LUNO (ICML 2025) — Linearized neural operator → GP
- Probabilistic Neural Operators (TMLR 2025)
