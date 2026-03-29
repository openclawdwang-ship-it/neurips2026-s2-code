# S2 Theory ↔ Experiment Alignment Audit
**Date**: 2026-03-26 | **Auditor**: H办公 (automated + manual review)

---

## Executive Summary

| Category | Status | Severity |
|----------|--------|----------|
| **Theory Soundness** | ⚠️ SOLID with caveats | MEDIUM |
| **Code-Theory Alignment** | ✅ EXCELLENT | NONE |
| **Experiment-Theory Gap** | ✅ FIXED (mostly) | LOW |
| **Reviewer Attack Surface** | ⚠️ MEDIUM | MEDIUM |
| **Ready for NeurIPS?** | 🟡 YES, with caveats | — |

**Bottom line**: Theory is mathematically sound. Code implements theory correctly. Previous consistency issues were fixed. BUT: 3 theoretical claims need qualification before submission.

---

## 1. THEORY SOUNDNESS

### Theorem 1: Spectral Score Unification ✅ SOLID

**Claim** (RESEARCH_PLAN.md L125-130):
> s_w(u) = Σ_k w_k |û_true(k) - û_pred(k)|² with weight choices recovers:
>   - L2 score (w_k = 1)
>   - H^s Sobolev score (w_k = (1+|k|²)^s)
>   - Pointwise score (w_k from inverse DFT)

**Status**: ✅ MATHEMATICALLY CORRECT
- Parseval's theorem: ||e||² = Σ_k |ê_k|² (exact for full FFT)
- Sobolev norm: ||e||_{H^s}² = Σ_k (1+|k|²)^s |ê_k|² (standard definition)
- Code implements all three via `weight_type` parameter in `SpectralScore`

**Code verification** (src/scores.py):
- `weight_type="uniform"` → `w_k = 1` ✅
- `weight_type="sobolev"` → `w_k = (1+|k|²)^s` ✅
- Parseval correction applied for rfft ✅

**⚠️ CAVEAT**: Theorem 1 uses full FFT, but code uses rfft (real FFT). The Parseval correction in `_apply_parseval_correction()` handles this correctly (interior bins × 2 for even N, all non-DC bins × 2 for odd N). This is correct but should be stated explicitly in the paper: "Theorem 1 holds for both FFT and rfft with appropriate normalization."

**Recommendation**: Add footnote: "For computational efficiency, we use rfft (real FFT) with Parseval correction factors to ensure equivalence to full FFT."

---

### Theorem 2: Tighter Bands via Spectral Weighting ⚠️ WEAK

**Claim** (RESEARCH_PLAN.md L133-135):
> If neural operator has spectral bias (error in high freq), spectral scores with weights matched to error spectrum produce provably tighter bands than uniform (L2).

**Status**: ⚠️ THEORETICALLY SOUND BUT IMPRECISELY STATED

**What's correct**:
- Conformal prediction theory: if score s₁ is stochastically smaller than s₂, then bands from s₁ are tighter (lower quantile)
- FNO spectral bias is well-documented (arXiv 2404.07200, 2602.19265)
- If w_k ∝ 1/σ_k² (inverse error variance), then spectral score has lower variance → tighter bands

**What's imprecise**:
- Theorem 2 claims "provably tighter" but doesn't state the sufficient condition precisely
- Current code uses `LearnedSpectralScore` which minimizes variance (heuristic), not inverse error power spectrum (theoretical optimum)
- The paper should state: "Theorem 2 provides the theoretical optimum (inverse error power spectrum). In practice, we learn weights via variance minimization, which is a heuristic approximation."

**Code verification** (src/scores.py LearnedSpectralScore):
- `learn_weights()` minimizes score variance via gradient descent ✅
- This is a reasonable heuristic but NOT the theoretical prescription

**Recommendation**: 
1. Restate Theorem 2 more precisely: "If weights are proportional to the inverse error power spectrum, then E[width | spectral] ≤ E[width | L2]."
2. Add: "In practice, we estimate weights by minimizing score variance on a held-out set, which is a finite-sample approximation to the theoretical optimum."
3. Optional ablation: compare learned weights vs. inverse-power weights on Darcy to show the heuristic is reasonable.

---

### Theorem 3: Coverage for Auto-Regressive Rollouts ✅ SOLID

**Claim** (RESEARCH_PLAN.md L138-141):
> ACI with spectral scores maintains (1-α+ε) coverage with ε→0 as calibration set grows.

**Status**: ✅ MATHEMATICALLY CORRECT

**Justification**:
- ACI (Gibbs & Candes 2021) guarantees coverage for ANY measurable score function
- FFT is a measurable linear transformation → spectral scores are measurable
- Exchangeability is preserved (no distribution shift assumed)
- Therefore, ACI + spectral scores = valid coverage guarantee

**Code verification** (src/spectral_aci.py):
- ACI update rule (line 148): `alpha_t = alpha_t + gamma * (alpha - err_t)` ✅ matches Gibbs & Candes exactly
- Quantile recomputation (lines 155-161): uses original calibration scores with adapted alpha_t ✅
- Clamping (line 150): `max(0.001, min(0.999, alpha_t))` ✅ standard practice

**No issues found.**

---

### Theorem 4: Spectral Error Propagation Bound ⚠️ PARTIAL

**Claim** (RESEARCH_PLAN.md L144-146):
> High-freq errors grow as O(L_k^T) where L_k is per-frequency Lipschitz constant. Adaptive w_k(t) tracking this growth produce tightest bands.

**Status**: ⚠️ THEORETICALLY SOUND BUT NOT FULLY IMPLEMENTED

**What's correct**:
- High-frequency error accumulation in auto-regressive rollouts is well-known
- Per-frequency Lipschitz constants are a valid model for error growth
- Theorem 4 is mathematically sound

**What's NOT implemented**:
- Code tracks per-frequency coverage via `compute_per_frequency_scores()` ✅
- Code does NOT adapt weights w_k(t) during rollout ❌
- Only the overall alpha_t is adapted via ACI

**Code verification** (src/spectral_aci.py):
- `per_freq_coverage` list is populated (line 161) ✅
- But weights are NOT updated based on per-frequency drift ❌
- This is documented as a "Week 4 enhancement" in REVIEW.md

**Recommendation**: 
1. Restate Theorem 4 as a theoretical result: "Optimal adaptive weights should track per-frequency error growth."
2. In the paper, present current implementation as "Spectral ACI with global adaptation" and per-frequency adaptation as a future direction.
3. This is NOT a blocker — the paper can present Theorem 4 as motivation and show empirically that even without per-frequency adaptation, spectral ACI outperforms static methods.

---

## 2. CODE-THEORY ALIGNMENT

### Consistency Check Results ✅ EXCELLENT

**Previous audit** (CONSISTENCY_CHECK.md) flagged 14 issues. Status of each:

| Issue | Status | Verification |
|-------|--------|--------------|
| Parseval correction for rfft | ✅ FIXED | `_apply_parseval_correction()` correctly handles even/odd N |
| Channel dim guard | ✅ FIXED | `ensure_channel_dim()` in utils.py checks `ndim == spatial_dims + 1` |
| Input/output normalization | ✅ FIXED | z-score computed from training data, saved in checkpoint |
| Relative L2 loss | ✅ FIXED | `relative_l2_loss()` normalizes by solution magnitude |
| Reproducibility seed | ✅ FIXED | `torch.manual_seed(42)` in train.py |
| Checkpoint format | ✅ FIXED | Saves dict with model_state_dict, x_mean, x_std, y_mean, y_std |
| Grid coordinates | ✅ FIXED | `append_grid()` adds spatial meshgrid as extra channels |
| Training epochs | ✅ FIXED | 100 → 500 epochs (matching neuraloperator standard) |
| CQR baseline | ⚠️ NOTED | Flagged as potentially infeasible; recommend dropping |
| DeepONet framing | ✅ FIXED | Clarified as "negative control" for spectral bias |
| Theorem 2 precision | ⚠️ NOTED | Needs qualification (see Section 1) |
| Shallow Water data | ⚠️ NOTED | Verify availability during Week 1 |
| LearnedSpectralScore objective | ✅ OK | Variance minimization is reasonable heuristic |
| Per-frequency adaptation | ✅ NOTED | Documented as Week 4 enhancement, not a bug |

**Conclusion**: All critical issues fixed. Remaining items are either documented limitations or low-priority enhancements.

---

### Score Function Correctness ✅ VERIFIED

**L2Score** (src/scores.py):
```python
error.reshape(batch_size, -1).pow(2).sum(dim=-1)
```
✅ Correct: ||e||² = Σ_i e_i²

**SpectralScore** (src/scores.py):
```python
e_hat = fft.rfft(error, dim=-1)  # or rfft2/rfftn
power = e_hat.real.pow(2) + e_hat.imag.pow(2)
power = _apply_parseval_correction(power, spatial_shape[-1])
weighted_power = power * w
scores = weighted_power.reshape(batch_size, -1).sum(dim=-1)
```
✅ Correct: s_w = Σ_k w_k |ê_k|² with Parseval correction

**Conformal Quantile** (src/conformal.py):
```python
quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
self.q_hat = float(torch.quantile(scores.float(), quantile_level).item())
```
✅ Correct: standard split conformal quantile formula

**ACI Update** (src/spectral_aci.py):
```python
self.state.alpha_t = self.state.alpha_t + self.gamma * (self.alpha - err_t)
self.state.alpha_t = max(0.001, min(0.999, self.state.alpha_t))
```
✅ Correct: Gibbs & Candes 2021 update rule with clamping

---

### Data Independence ✅ VERIFIED

**Three-way split** (for LearnedSpectralScore):
1. Training data → fit neural operator
2. Weight-learning data → optimize w_k
3. Calibration data → compute conformal quantile

**Code verification** (src/conformal.py):
- `SplitConformalPredictor.calibrate()` uses separate calibration set ✅
- `LearnedSpectralScore.learn_weights()` uses separate weight-learning set ✅
- No data leakage between splits ✅

---

## 3. EXPERIMENT-THEORY GAP

### Experimental Design ✅ SOUND

**Metrics** (PRINCIPLES.md):
1. Hard constraint: coverage ≥ (1-α) - 0.02
2. Primary metric: band width (lower = better)

✅ Correct: lexicographic ordering ensures coverage first, then tightness.

**Ablations** (PRINCIPLES.md, priority queue):
1. Baseline: L2 score on Darcy
2. Spectral uniform (verify Parseval)
3. Sobolev H^1, H^2
4. Inverse power spectrum
5. Learned spectral weights
6. Cross-architecture (TFNO, DeepONet, UNO)
7. Rollout: static L2 vs spectral vs spectral ACI
8. Calibration set size ablation
9. Resolution sensitivity

✅ Logical progression from simple to complex.

**Budget** (PRINCIPLES.md):
- Model training: 30 min per PDE (one-time)
- Conformal calibration: 2 min per experiment
- Rollout evaluation: 5 min per experiment
- Full ablation point: 10 min

✅ Realistic for RTX 3090 or RunPod A40.

---

### Baseline Comparisons ✅ COMPREHENSIVE

**Conformal baselines**:
- Split conformal with L2 score ✅
- CQR (quantile regression) — flagged as potentially infeasible
- CP-PRE (ICML 2025) — requires implementation

**Non-conformal baselines**:
- MC Dropout ✅
- Deep Ensemble ✅
- DINOZAUR-style Bayesian ✅

**Recommendation**: Drop CQR if it requires training a new model. Replace with simpler quantile baseline (e.g., conformalized quantile from ensemble).

---

## 4. REVIEWER ATTACK SURFACE

### Likely Criticisms & Mitigations

| Criticism | Likelihood | Mitigation |
|-----------|------------|-----------|
| "Spectral scores are just using arXiv 2602.19265's frequency-resolved metrics" | MEDIUM | Emphasize: (1) we define conformal scores in spectral domain (novel), (2) we prove coverage guarantees, (3) we integrate with ACI for temporal adaptation, (4) 4 theorems + benchmark go far beyond diagnostic use |
| "CP-PRE already does rollouts with recalibration" | MEDIUM | Clarify: CP-PRE uses static per-step recalibration + spatial PDE residual scores. We use adaptive (ACI) recalibration + spectral scores with per-frequency tracking. Fundamentally different mechanism. |
| "Theorem 2 is not rigorously proven" | LOW-MEDIUM | Restate precisely: "If weights ∝ inverse error power spectrum, then E[width \| spectral] ≤ E[width \| L2]." Add finite-sample approximation bound. |
| "Per-frequency adaptation (Theorem 4) is not implemented" | LOW | Acknowledge as future work. Present current implementation as "global ACI" and show empirically it outperforms static methods. |
| "Spectral scores don't actually improve tightness" | LOW | Strong theoretical grounding + FNO spectral bias is well-documented. Empirical results should show improvement. |
| "Three-way split reduces effective calibration set size" | LOW | Standard practice for learned scores. Justify: "We use 1000 calibration points, split 500/250/250 for weight learning and conformal calibration. This is standard in conformal prediction literature." |

---

## 5. FINAL CHECKLIST

### Before Submission

- [ ] **Theorem 1**: Add footnote about rfft normalization
- [ ] **Theorem 2**: Restate precisely with finite-sample bound
- [ ] **Theorem 4**: Clarify as theoretical optimum; current implementation is "global ACI"
- [ ] **CQR baseline**: Drop or replace with simpler quantile baseline
- [ ] **Shallow Water data**: Verify availability during Week 1
- [ ] **Learned weights ablation**: Compare learned vs. inverse-power weights on Darcy
- [ ] **Per-frequency tracking**: Include diagnostics in appendix (even if not adapted)
- [ ] **Reproducibility**: Commit all code + configs to GitHub with seed=42
- [ ] **Figures**: Hero figure should show Darcy 64×64 results (coverage + band width comparison)

### Experimental Milestones

**Week 1 (Setup)**:
- [ ] Verify all 5 PDEs load correctly
- [ ] Train baseline FNO on Darcy 64×64
- [ ] Run L2 conformal baseline
- [ ] Verify coverage ≥ 0.88 (for α=0.1)

**Week 2 (Spectral scores)**:
- [ ] Implement SpectralScore (uniform, Sobolev, inverse_power)
- [ ] Verify Parseval equivalence: uniform ≈ L2
- [ ] Test Sobolev H^1, H^2 on Darcy
- [ ] Log results to results.tsv

**Week 3 (Learned weights)**:
- [ ] Implement LearnedSpectralScore
- [ ] Three-way split: 500 train / 250 weight-learn / 250 calibrate
- [ ] Compare learned vs. inverse-power weights
- [ ] Cross-architecture: TFNO, DeepONet, UNO

**Week 4 (Rollouts)**:
- [ ] Implement Spectral ACI
- [ ] Static L2 vs. spectral vs. spectral ACI on Navier-Stokes
- [ ] Per-frequency coverage tracking
- [ ] Ablation: calibration set size, resolution

**Week 5 (Polish)**:
- [ ] Generate figures + tables
- [ ] Write paper draft
- [ ] Address reviewer attack surface
- [ ] Commit to GitHub

---

## 6. CONCLUSION

**Theory**: ✅ Mathematically sound. Theorems 1 & 3 are solid. Theorems 2 & 4 need qualification but are not incorrect.

**Code**: ✅ Correctly implements theory. All previous consistency issues fixed. No bugs found.

**Experiments**: ✅ Well-designed. Comprehensive baselines. Realistic budget.

**Readiness**: 🟡 **YES, proceed with experiments.** Address the 3 theory qualifications before submission.

**Risk level**: MEDIUM (novelty gap with arXiv 2602.19265 is real but mitigable; CP-PRE comparison is critical).

---

**Next step**: Start Week 1 experiments. Run baseline L2 conformal on Darcy 64×64 to verify coverage ≥ 0.88.
