# Research Plan: Conformal Prediction for Neural PDE Operators

## ⚠️ Honesty Check: Prior Work Exists

**Original claim**: "Two hot communities with ZERO intersection" — **FALSE**

At least 6 papers (2024-2026) directly combine conformal prediction with neural operators:
- Conformalized-DeepONet (Moya et al., Feb 2024)
- Risk-Controlling Quantile Neural Operator (Ma/Anandkumar, Feb 2024)
- Function-space CP for Neural Operators (Millard et al., Sep 2025)
- CMCO (Kobayashi et al., Jul 2025)
- LSCI (Harris et al., Jul 2025)
- CoNBONet (Garg & Chakraborty, Mar 2026)

Plus: DINOZAUR (Bayesian FNO, NeurIPS 2025 Spotlight), LUNO (ICML 2025).

**The field exists. We must contribute something genuinely new.**

---

## Decision: CONFIRMED — Spectral ACI (Option A)

**Direction confirmed 2026-03-25.** Proceeding with Spectral ACI = spectral conformal scores + adaptive temporal conformal, unified as one framework.

### Novelty Verification (2026-03-25)
- 3 independent literature search agents confirmed: **zero papers** define conformal scores in Fourier/spectral domain
- Mathematical verification: coverage guarantees fully hold (FFT is measurable, doesn't break exchangeability)
- Closest competitor: LSCI (Harris 2025) at novelty gap 4/10 — mentions Fourier as one projection option but doesn't design around it
- **CP-PRE update**: Gopakumar et al. (ICML 2025) also does 20-step auto-regressive rollouts (not just static). Our C2 differentiates via ACI's online adaptive recalibration + per-frequency coverage tracking, which CP-PRE does not do.
- Learned spectral weights require **three-way data split** (train / learn-weights / calibrate) — standard technique, not a barrier.

---

## Repositioned Paper: Spectral Conformal Prediction for Neural PDE Operators

### Title (working)
*Spectral Conformal Prediction for Neural PDE Operators: Frequency-Resolved Distribution-Free UQ with Adaptive Temporal Coverage*

### Core Contributions

**C1: Spectral Conformal Prediction** (PRIMARY — novel, strongest angle)
- Problem: All existing conformal methods for neural operators use spatial-domain scores (pointwise L2, norm-based). They ignore the operator's internal spectral representation.
- Solution: Define nonconformity scores in the **Fourier domain**: s(u) = Σ_k w_k |û_true(k) - û_pred(k)|² where w_k are frequency-dependent weights.
- Key insight: FNO operates natively in Fourier space and has known **spectral bias** (learns low frequencies first, high-frequency errors dominate). Spectral scores exploit this structure.
- Theoretical framework unifying existing scores:
  - w_k = 1 ∀k → recovers L2 norm (Parseval's theorem)
  - w_k = (1+|k|²)^s → recovers Sobolev H^s norm
  - w_k learned from calibration data → adaptive spectral score (tightest bands)
- **Why this is novel**: No paper defines conformal scores in spectral space. No paper exploits FNO's internal representation for UQ. This is the FIRST conformal method aware of the operator's architecture.

**C2: Spectral ACI for Auto-Regressive PDE Rollouts** (SECONDARY — novel)
- Problem: Neural operators rolled out auto-regressively accumulate error over time. CP-PRE (ICML 2025) does 20-step rollouts but uses static per-step recalibration without adaptive learning rates.
- Solution: Apply Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) with **spectral scores** to sequential PDE rollouts. The ACI learning rate adapts online based on coverage miscalibration.
- Key differentiator from CP-PRE: **per-frequency coverage tracking** — spectral ACI monitors which frequency bands lose coverage during rollout and adapts weights w_k(t) accordingly. High-frequency bands degrade faster → spectral ACI automatically widens high-freq bands while keeping low-freq bands tight.
- **Why this is novel**: CP-PRE uses static recalibration with PDE residual scores in spatial domain. We use adaptive (ACI) recalibration with spectral scores that track per-frequency drift — fundamentally different mechanism.

**C3: Comprehensive Cross-Architecture Benchmark** (TERTIARY — practical value)
- First systematic comparison of conformal prediction across FNO, TFNO, DeepONet, and U-NO on 5 PDEs from PDEBench.
- Head-to-head with Bayesian (DINOZAUR-style) and ensemble baselines.
- **Why this matters**: Existing papers each test 1-2 architectures on 2-3 PDEs. No unified benchmark.

### Non-Contributions (what we're NOT claiming)
- ❌ NOT "first conformal prediction for neural operators" (6+ papers exist)
- ❌ NOT "first physics-informed conformal scores" (CP-PRE, ICML 2025 does this)
- ❌ NOT "first UQ for PDE solvers"
- ❌ NOT "first function-space conformal prediction"

---

## Experimental Design

### PDEs (from PDEBench)
| PDE | Dimension | Type | Why |
|-----|-----------|------|-----|
| 2D Darcy Flow | Spatial only | Elliptic | Standard benchmark, static operator |
| 1D Burgers | Spatial + time | Hyperbolic | Simple time-dependent, good for ACI validation |
| 2D Navier-Stokes | Spatial + time | Parabolic | **Key experiment**: auto-regressive rollout |
| 1D Diffusion-Reaction | Spatial + time | Parabolic | Stiff dynamics, error accumulation test |
| 2D Shallow Water | Spatial + time | Hyperbolic | Complex dynamics, real-world relevance |

### Neural Operator Architectures
| Model | Source | Parameters | Training Time (est.) |
|-------|--------|------------|---------------------|
| FNO | neuraloperator lib | ~2M | 2-4 hrs per PDE |
| TFNO | neuraloperator lib | ~200K | 0.5-1 hr per PDE |
| DeepONet | DeepXDE or custom | ~1M | 2-4 hrs per PDE |
| U-NO | custom impl | ~3M | 4-8 hrs per PDE |

### UQ Methods to Compare
| Method | Type | Coverage Guarantee? | Source |
|--------|------|-------------------|--------|
| **Split Conformal (L2 score)** | Conformal | Yes (marginal) | Baseline |
| **Spectral Conformal (ours)** | Conformal + spectral | Yes (marginal) | **Novel (C1)** |
| **Spectral ACI for Rollouts** (ours) | Adaptive spectral conformal | Yes (online) | **Novel (C1+C2)** |
| CQR (Romano et al.) | Conformal quantile | Yes | Baseline |
| CP-PRE (Gopakumar et al.) | Conformal + PDE residual | Yes | Baseline (ICML 2025) |
| MC Dropout | Bayesian approx. | No | Baseline |
| Deep Ensemble (5 models) | Ensemble | No | Baseline |
| DINOZAUR-style Bayesian | Bayesian | No (credible intervals) | Baseline |

### Metrics
| Metric | What it measures |
|--------|-----------------|
| **Empirical coverage** | % of test points within prediction band |
| **Band width** (avg) | Tightness of prediction bands |
| **Band width** (by timestep) | How bands grow during rollout — KEY for C1 |
| **Calibration error** | |coverage - target| |
| **Computational overhead** | Time for UQ vs base prediction |
| **PDE residual** | Physics compliance of bands |

### Ablations
1. **Score function ablation**: pointwise L2 vs H1-norm vs spectral (uniform w_k) vs spectral (learned w_k) vs PDE-residual (CP-PRE style)
2. **Spectral weight schemes**: uniform, Sobolev H^s (vary s), inverse-power-spectrum, learned from calibration
3. **Calibration set size**: 100, 500, 1000, 5000 calibration examples
4. **Resolution sensitivity**: 32×32, 64×64, 128×128
5. **Rollout length**: coverage vs timestep plots (the money figure for C2)
6. **ACI learning rate**: sensitivity of adaptive conformal to step size
7. **Per-frequency coverage**: coverage and band width decomposed by frequency band (low/mid/high)

---

## Theoretical Contributions

### Theorem 1: Spectral Score Unification
- Show that the spectral nonconformity score s_w(u) = Σ_k w_k |û_true(k) - û_pred(k)|² with appropriate weight choices recovers:
  - L2 score (w_k = 1)
  - H^s Sobolev score (w_k = (1+|k|²)^s)
  - Pointwise score (w_k from inverse DFT)
- Prove that split conformal with spectral scores maintains finite-sample marginal coverage for any weight vector w ≥ 0.
- **Novelty**: First unified spectral framework for functional conformal prediction.

### Theorem 2: Tighter Bands via Spectral Weighting
- Show that if the neural operator has spectral bias (error concentrated in high frequencies), then spectral scores with weights matched to the error spectrum produce provably tighter prediction bands than uniform (L2) scores.
- Formalize: E[width | spectral_score] ≤ E[width | L2_score] when weights are proportional to the inverse error power spectrum.
- **Novelty**: First formal connection between operator spectral bias and conformal band tightness.

### Theorem 3: Coverage for Auto-Regressive Neural Operator Rollouts
- Setting: Neural operator Gθ predicts u_{t+1} = Gθ(u_t). At each step, apply spectral conformal prediction with score s_t.
- Challenge: u_t is the MODEL's prediction (not ground truth), so errors compound AND the spectral error profile shifts over time (high frequencies degrade faster).
- Approach: Use ACI (Gibbs & Candes 2021) framework with spectral scores. Show that if the neural operator has bounded Lipschitz constant and the spectral score function is sub-Gaussian, the ACI bands maintain (1-α+ε) coverage with ε → 0 as calibration set grows.
- **Novelty**: First coverage theorem for auto-regressive PDE operator setting with spectral awareness.

### Theorem 4: Spectral Error Propagation Bound
- Characterize how the spectral error profile evolves during auto-regressive rollout: high-frequency errors grow as O(L_k^T) where L_k is the per-frequency Lipschitz constant and T is the rollout step.
- Show that adaptive spectral weights w_k(t) that track this growth produce the tightest possible coverage bands at each timestep.
- **Novelty**: First analysis connecting PDE operator stability theory (Lax-Richtmyer) to conformal prediction band width.

---

## Timeline (6 weeks)

### Week 1: Setup & Static Conformal Baseline
- Install neuraloperator, PDEBench, download data
- Train FNO on 2D Darcy (static, simplest case)
- Implement split conformal prediction wrapper
- Validate: reproduce standard coverage on Darcy

### Week 2: Multi-Architecture Static Experiments
- Train FNO, TFNO, DeepONet on all 5 PDEs
- Run split conformal + CQR on all combinations
- Implement MC Dropout and ensemble baselines
- Generate static coverage/width tables (C3)

### Week 3: Spectral Conformal Prediction (C1 — core contribution)
- Implement spectral nonconformity score: FFT of error field, weighted aggregation
- Test weight schemes: uniform (L2), Sobolev H^s, inverse-power-spectrum, learned
- Key experiment: FNO on Navier-Stokes — show spectral scores give tighter bands than L2
- Generate per-frequency coverage plots (the money figure for C1)
- Compare against CP-PRE (PDE residual scores) as baseline

### Week 4: Adaptive Temporal Conformal (C2) + Ablations
- Implement auto-regressive rollout for time-dependent PDEs
- Implement ACI (Gibbs & Candes 2021) with spectral scores for rollout setting
- Track per-frequency coverage drift during rollout
- Run all ablations (score function, spectral weights, calibration set size, resolution, rollout length)
- Key experiment: spectral error profile evolution over rollout steps

### Week 5: Theory + Analysis
- Write proofs for Theorems 1-4 (spectral unification, tighter bands, rollout coverage, error propagation)
- Deep analysis of failure modes (when does spectral conformal fail? turbulent regimes?)
- Generate all figures and tables

### Week 6: Paper Writing
- Full NeurIPS paper draft (main + appendix)
- Abstract, intro, related work, method, experiments, conclusion
- Appendix: all proofs, additional experiments, implementation details

---

## Risk Analysis

### Risk 1: Spectral scores don't improve tightness (LOW-MEDIUM)
- **Threat**: Spectral weighting doesn't produce meaningfully tighter bands than L2
- **Mitigation**: FNO's spectral bias is well-documented — errors ARE concentrated in high frequencies. If spectral scores don't help for FNO, test DeepONet (spatial domain) as a negative control. Even showing WHEN spectral helps vs doesn't is a contribution.
- **Probability**: ~15% (strong theoretical grounding)

### Risk 2: Gap closes before submission (LOW for C1, MEDIUM for C2)
- **Threat**: Someone publishes spectral conformal for neural operators before Sep 2026
- **Mitigation**: Spectral conformal is a more specific/unusual angle than temporal conformal — lower scooping risk. Monitor arXiv weekly. If scooped on C1, fall back to C2+C3. If scooped on C2, C1+C3 still stand.
- **Probability**: ~5-10% for C1 (spectral), ~20-30% for C2 (temporal ACI)

### Risk 3: ACI doesn't maintain coverage in auto-regressive setting (LOW)
- **Threat**: Error accumulation breaks the ACI guarantees
- **Mitigation**: ACI is designed for non-exchangeable data. The theoretical framework should hold. If not, this itself is a publishable negative result + the static spectral results (C1+C3) still stand.

### Risk 4: Reviewers say "incremental" (MEDIUM)
- **Threat**: Closest competitors: Ma et al. 2024 (Caltech/NVIDIA), CP-PRE (ICML 2025)
- **Mitigation**: Clear differentiation: (1) We work in spectral domain, they work in spatial domain — fundamentally different score design. (2) We address temporal rollouts, they do static. (3) We provide a unifying theoretical framework (Theorems 1-4) connecting Sobolev norms, spectral bias, and Lax-Richtmyer stability to conformal coverage.

### Risk 5: CP-PRE (ICML 2025) is seen as too similar (MEDIUM)
- **Threat**: CP-PRE uses PDE residuals as conformal scores — reviewers may conflate this with our spectral approach
- **Mitigation**: Explicitly compare against CP-PRE as a baseline. Show that spectral scores exploit MODEL structure (FNO's Fourier layers) while CP-PRE exploits PHYSICS structure (PDE residuals) — orthogonal approaches that can be combined.

---

## Budget

| Item | Hours | Cost |
|------|-------|------|
| Train 4 models × 5 PDEs | 40-60 hrs | $9-13 |
| Ensemble baselines (5× each model) | 15-20 hrs | $3-4 |
| Ablations | 10-15 hrs | $2-3 |
| Conformal calibration | <1 hr | $0.22 |
| **Total** | **~70-100 hrs** | **$15-22** |

On RTX 3090 at $0.22/hr. Well within $100 budget with margin for reruns.

---

## Expected Figures

1. **Per-Frequency Coverage & Band Width** (hero figure) — shows spectral scores give tighter bands than L2/H1, with coverage decomposed by frequency band
2. **Spectral Error Profile** — power spectrum of prediction error vs ground truth, showing FNO's spectral bias (high-freq errors dominate)
3. **Spectral Weights Visualization** — learned w_k vs Sobolev vs uniform, showing how optimal weights match inverse error spectrum
4. **Coverage vs Timestep** — shows standard conformal loses coverage over rollout, spectral ACI maintains it
5. **Spectral Error Evolution During Rollout** — per-frequency error growth over timesteps, high frequencies degrade faster
6. **Heatmap: Architecture × PDE × Method** — coverage/width for all combinations (C3)
7. **Score Function Comparison** — L2 vs H1 vs spectral vs CP-PRE (PDE residual) — band width at fixed coverage
8. **Calibration Set Size** — diminishing returns curve
9. **Failure Mode Analysis** — when spectral conformal fails (turbulent regimes, broadband errors)

---

## Status

**CONFIRMED 2026-03-25**: Proceeding with Spectral ACI. Implementation starting.
