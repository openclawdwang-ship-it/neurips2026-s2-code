# Theory ↔ Code Consistency Check

Date: 2026-03-25
Checker: Claude (automated, then human-verified)

---

## 1. RESEARCH_PLAN.md Theorems → Code Implementation

### Theorem 1: Spectral Score Unification
**Claim** (RESEARCH_PLAN.md L125-130):
> s_w(u) = Σ_k w_k |û_true(k) - û_pred(k)|² with weight choices recovers:
>   - L2 score (w_k = 1)
>   - H^s Sobolev score (w_k = (1+|k|²)^s)
>   - Pointwise score (w_k from inverse DFT)

**Code** (`src/scores.py` SpectralScore):
- `weight_type="uniform"`: `_get_weights` returns `torch.ones(1)` → broadcasts as w_k=1 ✅
- `weight_type="sobolev"`: returns `(1.0 + k_sq).pow(self.sobolev_s)` ✅
- `_score_single_field`: computes `rfft → |e_hat|² → weighted sum` ✅
- Formula: `scores = (power * w).reshape(batch, -1).sum(dim=-1)` = Σ_k w_k |ê_k|² ✅

**Parseval equivalence** (w_k=1 should match L2):
- `L2Score.__call__`: returns `error.reshape(batch, -1).pow(2).sum(dim=-1)` = ||e||²
- `SpectralScore(weight_type="uniform")`: returns `Σ_k |ê_k|²`
- By Parseval's theorem: ||e||² = Σ_k |ê_k|² ... **BUT** code uses `rfft` not `fft`.
  For `rfft`, Parseval's theorem is: ||e||² = |ê_0|² + 2·Σ_{k=1}^{N/2-1} |ê_k|² + |ê_{N/2}|²
  The `rfft` output already halves the spectrum, but the DC and Nyquist bins need
  factor-of-1 while interior bins need factor-of-2 for exact Parseval equivalence.

**⚠️ INCONSISTENCY FOUND**: `SpectralScore(uniform)` does NOT exactly equal `L2Score`
because `rfft` drops the redundant negative frequencies. The uniform-weighted sum of
`|rfft|²` is approximately `||e||²/2` (off by the missing conjugate terms). The coverage
guarantee is unaffected (any score function works), but the CLAIM in Theorem 1 that
"w_k=1 recovers L2" is only exact for full `fft`, not `rfft`.

**Fix options**:
1. Use `fft` instead of `rfft` (doubles memory, exact equivalence)
2. Apply Parseval correction factors for `rfft` (multiply interior bins by 2)
3. Weaken the claim: "recovers L2 up to a constant factor" (still valid for CP)

**Severity**: MEDIUM — coverage guarantee unaffected, but Theorem 1 Parseval claim
needs qualification if using `rfft`. The actual experimental comparison is fair because
ALL spectral scores use the same `rfft` convention.

**FIXED**: Applied option 2 — added Parseval correction in `_score_single_field`.

---

### Theorem 2: Tighter Bands via Spectral Weighting
**Claim** (RESEARCH_PLAN.md L133-135):
> If neural operator has spectral bias (error in high freq), spectral scores with
> weights matched to error spectrum produce provably tighter bands than uniform (L2).

**Code**: `LearnedSpectralScore.learn_weights` optimizes weights by minimizing score
variance on a held-out set. Lower variance → more concentrated score distribution →
tighter conformal bands (smaller q_hat at fixed coverage level).

**Consistency**: The optimization objective (minimize variance) is a HEURISTIC for
tighter bands, not a proof of Theorem 2. The theorem claims E[width|spectral] ≤
E[width|L2] when weights ∝ inverse error power spectrum. The code does NOT directly
implement the inverse error power spectrum — it learns weights via gradient descent.

**Status**: ⚠️ WEAK CONSISTENCY — code implements a reasonable algorithm to find good
weights, but does not directly implement the theorem's sufficient condition (inverse
error power spectrum). The `weight_type="inverse_power"` does compute 1/(1+|k|²) but
this is inverse of the FREQUENCY magnitude, not inverse of the ERROR power spectrum.

**Not a bug** — the theorem is a theoretical result about optimal weights, and the
learned weights are an empirical approximation. But the paper should be clear that
Theorem 2 provides the theoretical motivation while `LearnedSpectralScore` provides the
practical implementation. The `inverse_power` weight type is a DIFFERENT thing
(emphasizes low frequencies, opposite of what Theorem 2 suggests for FNO).

**Action**: Add a `weight_type="inverse_error_power"` that directly implements the
theorem's prescription for ablation comparison. (Week 3 task, not a blocker.)

---

### Theorem 3: Coverage for Auto-Regressive Rollouts
**Claim** (RESEARCH_PLAN.md L138-141):
> ACI with spectral scores maintains (1-α+ε) coverage with ε→0 as calibration set grows.

**Code** (`src/spectral_aci.py` SpectralACI):
- ACI update rule (line 148): `alpha_t = alpha_t + gamma * (alpha - err_t)` ✅
  Matches Gibbs & Candes 2021 exactly (see Check #4 below).
- Clamping (line 150): `max(0.001, min(0.999, self.state.alpha_t))` ✅
  Necessary for numerical stability, standard practice.
- Quantile recomputation (lines 155-161): recomputes q_hat from ORIGINAL calibration
  scores using the adapted alpha_t. ✅ This is the correct ACI mechanism.

**Status**: ✅ CONSISTENT — code implements ACI correctly for the rollout setting.

---

### Theorem 4: Spectral Error Propagation Bound
**Claim** (RESEARCH_PLAN.md L144-146):
> High-freq errors grow as O(L_k^T) where L_k is per-frequency Lipschitz constant.
> Adaptive w_k(t) tracking this growth produce tightest bands.

**Code**: `SpectralACI` tracks per-frequency coverage via `compute_per_frequency_scores`
but does NOT actually adapt the weights w_k(t) during rollout (as noted in REVIEW.md
item 2). It only adapts the overall alpha_t.

**Status**: ⚠️ PARTIAL — per-frequency TRACKING is implemented, per-frequency ADAPTATION
is not. This is a Week 4 enhancement documented in REVIEW.md. The current code gives
correct coverage via ACI (Theorem 3) but does not achieve the TIGHTEST possible bands
predicted by Theorem 4.

**Not a blocker**: The paper can present Theorem 4 as the theoretical optimal and show
empirically that even without per-frequency adaptation, spectral ACI outperforms static
methods. Per-frequency adaptation is an ablation for the appendix.

---

## 2. Literature-Review Gaps → Code Coverage

### Gap 1: Cross-Architecture Benchmark (⭐⭐⭐)
**Claim**: No systematic comparison across FNO, DeepONet, U-NO, TFNO on same PDEs.
**Code**: `src/models.py` implements FNO, TFNO, DeepONet, UNO. `scripts/run_all.sh`
Phase 2 runs calibration on fno/tfno/deeponet × darcy + fno × burgers. ✅
**Gap filled?**: PARTIALLY — UNO not tested by default (no training config in run_all.sh).
Could add, but UNO falls back to FNO in models.py when neuralop doesn't have it.

### Gap 2: Adaptive Temporal Conformal for Rollouts (⭐⭐⭐⭐)
**Claim**: ACI with spectral scores for sequential PDE prediction is unstudied.
**Code**: `src/spectral_aci.py` implements exactly this. `scripts/rollout_eval.py`
compares static L2 vs static spectral vs spectral ACI. ✅
**Gap filled?**: YES — this is the core C2 contribution.

### Gap 3: Conformal vs Bayesian vs Ensemble Head-to-Head (⭐⭐⭐)
**Claim**: No systematic comparison of ALL UQ methods.
**Code**: `calibrate.py` runs conformal (7 score types) + MC Dropout baseline.
Deep ensemble not implemented (requires training 5 models — budget concern). ✅ partial
**Gap filled?**: PARTIALLY — MC Dropout is there, ensemble is budget-gated.

### Gap 6: Spectral Conformal Prediction (⭐⭐⭐⭐⭐)
**Claim**: No paper defines conformal scores in Fourier/spectral domain.
**Code**: `src/scores.py` SpectralScore does exactly this. ✅
**Gap filled?**: YES — this is the core C1 contribution.

---

## 3. scores.py Spectral Score Formula ↔ RESEARCH_PLAN Math

### RESEARCH_PLAN formula (line 43):
> s(u) = Σ_k w_k |û_true(k) - û_pred(k)|²

### Code (`SpectralScore._score_single_field`):
```python
e_hat = fft.rfft2(error, dim=(-2, -1))      # û_err = û_true - û_pred
power = e_hat.real.pow(2) + e_hat.imag.pow(2) # |û_err(k)|²
w = self._get_weights(spatial_shape, device)   # w_k
weighted_power = power * w                     # w_k * |û_err(k)|²
scores = weighted_power.reshape(batch, -1).sum(dim=-1)  # Σ_k
```

**Line-by-line match**:
- `error = u_true - u_pred` → û_err = FFT(u_true - u_pred) = û_true - û_pred ✅
  (FFT is linear, so FFT(a-b) = FFT(a) - FFT(b))
- `|e_hat|² = real² + imag²` ✅ (correct complex modulus squared)
- `w * |e_hat|²` then `sum` → Σ_k w_k |û_err(k)|² ✅

**⚠️ BUT**: Uses `rfft2` not `fft2`. See Theorem 1 finding above. For the score to
exactly equal the RESEARCH_PLAN formula Σ_k (summing over ALL k), need Parseval
correction for rfft (interior bins counted once instead of twice).

**FIXED**: Added rfft Parseval correction factor.

---

## 4. spectral_aci.py ACI Update ↔ Gibbs & Candes 2021

### Gibbs & Candes 2021 (Algorithm 1):
> α_{t+1} = α_t + γ(α - err_t)
> where err_t = 1{Y_t ∉ C_t(X_t)} is the miscoverage indicator

### Code (SpectralACI.step, line 148):
```python
err_t = (scores_t > self.state.q_hat_t).float().mean().item()
self.state.alpha_t = self.state.alpha_t + self.gamma * (self.alpha - err_t)
```

**Match analysis**:
- `err_t`: Original paper uses binary indicator per sample. Code uses MEAN over batch.
  This is a BATCH extension of ACI — standard practice when processing batches, but
  differs from the per-sample guarantee in the original paper.
- Update rule: `α_{t+1} = α_t + γ(α - err_t)` ✅ Exact match.
- Clamping to [0.001, 0.999]: Not in original paper, but standard numerical safeguard. ✅

**Status**: ✅ CONSISTENT — faithful implementation of ACI with standard batch extension.

---

## 5. conformal.py Coverage Guarantee ↔ Theory

### Theory requirement:
> For split conformal with n calibration points, q_hat = Quantile_{⌈(1-α)(n+1)⌉/n}(scores)
> guarantees P(s(X_{n+1}) ≤ q_hat) ≥ 1-α.

### Code (SplitConformalPredictor.calibrate):
```python
quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
quantile_level = min(quantile_level, 1.0)
self.q_hat = float(torch.quantile(scores.float(), quantile_level).item())
```

**Match analysis**:
- `⌈(1-α)(n+1)⌉/n`: Code computes `ceil((1-α)(n+1))/n` ✅
- `min(quantile_level, 1.0)`: Necessary when n is small. ✅
- `torch.quantile(scores, level)`: torch.quantile uses linear interpolation by default.
  The theoretical formula requires taking the ⌈(1-α)(n+1)⌉-th ORDER STATISTIC (no
  interpolation). For large n, the difference is negligible. For small n (e.g., n=30
  in the three-way split), this could matter.

**⚠️ MINOR INCONSISTENCY**: `torch.quantile` interpolates between order statistics, while
the theoretical guarantee requires the exact order statistic (ceiling). In practice, this
makes coverage slightly CONSERVATIVE (overestimates q_hat), which is safe. The guarantee
still holds — it's just not the tightest possible q_hat.

**Severity**: LOW — coverage is valid (conservative direction). No fix needed.

---

## 6. calibrate.py Three-Way Split ↔ CP Theory

### Theory requirement:
For LearnedSpectralScore, calibration data used to learn weights MUST be independent
of calibration data used to compute q_hat. Otherwise coverage guarantee is invalidated.

### Code (conformal.py three_way_split):
```python
idx_weight = perm[:n_weight]        # for learning w_k
idx_cal = perm[n_weight:n_weight+n_cal]  # for computing q_hat
idx_test = perm[n_weight+n_cal:]    # for evaluation
```

**Status**: ✅ CONSISTENT — weight-learning and calibration use disjoint index sets.

### Code (calibrate.py, LearnedSpectralScore branch):
```python
weight_data, cal_data, _ = three_way_split(cal_pred, cal_y, frac_weight=0.3, frac_cal=0.7)
errors = weight_data["true"] - weight_data["pred"]
score_fn.learn_weights(errors)       # uses weight_data ONLY
cp.calibrate(cal_data["pred"], cal_data["true"])  # uses cal_data ONLY
```

**Status**: ✅ CONSISTENT — weight learning and calibration use disjoint splits.

**⚠️ CONCERN** (from code review): With n_cal=100 and frac_weight=0.3, only 30 samples
for weight learning. This is a PRACTICAL concern (underconstrained optimization), not a
THEORETICAL violation. Coverage guarantee holds regardless of weight quality — bad weights
just mean wider (less informative) bands, not invalid coverage.

---

## 7. PRINCIPLES.md Keep/Discard Rules ↔ run_all.sh

### PRINCIPLES.md decision rule (lines 43-51):
```
if coverage < (1 - alpha) - 0.02:
    status = "invalid"
elif band_width < best_band_width:
    status = "keep"
else:
    status = "discard"
```

### Code (run_all.sh does NOT implement keep/discard logic):
run_all.sh calls calibrate.py which logs results to results.tsv with status. But
run_all.sh itself does NOT implement the keep/discard/git-reset loop described in
PRINCIPLES.md.

**Status**: ⚠️ INCONSISTENCY — PRINCIPLES.md describes an autonomous experiment loop
with git commit + keep/discard + git reset. run_all.sh is a LINEAR pipeline that runs
all experiments sequentially without comparison logic. There is no `best_band_width`
tracking, no git commit per experiment, no git reset on failure.

**Analysis**: This is by design for the initial implementation. PRINCIPLES.md describes
the IDEAL autonomous loop for iterative research. run_all.sh is the FIRST-PASS pipeline
that runs the full experiment matrix. The keep/discard loop would be used AFTER the
first pass, when iterating on score functions. The two documents describe different
phases of the research workflow.

**Severity**: LOW — not a bug, but a documentation gap. PRINCIPLES.md should note that
run_all.sh is the initial sweep, and the keep/discard loop applies to subsequent
iteration.

### calibrate.py status logic:
```python
valid = cov >= (target_coverage - 0.02)
status = "keep" if valid and "error" not in metrics else "invalid"
```

**Match to PRINCIPLES.md**: Partial — checks coverage threshold ✅, but does NOT compare
band_width to best_band_width to distinguish "keep" from "discard". All valid results
are labeled "keep". This is correct for the initial sweep (no prior best to compare to).

---

## 8. Additional Checks

### 8.1 Data split independence
**Theory**: Train/cal/test must be independent for coverage guarantee.
**Code** (`src/data.py` get_data_splits): Uses `torch.randperm` with fixed seed to
create disjoint index sets. Training indices never overlap with cal or test. ✅

### 8.2 Score function measurability
**Theory**: Any measurable nonconformity score preserves CP guarantee.
**Code**: FFT, weighted sums, and norms are all measurable functions. ✅

### 8.3 Exchangeability assumption
**Theory**: Split conformal requires calibration and test data to be exchangeable.
**Code**: Data is i.i.d. from PDEBench (same distribution), split randomly. ✅
**For ACI (rollout)**: ACI explicitly handles non-exchangeable (sequential) data.
The original Gibbs & Candes 2021 guarantee applies to distribution shift. ✅

### 8.4 CQR-lite coverage guarantee
**Theory**: SimplifiedCQRScore is a valid nonconformity score (deterministic function
of pred and true), so split conformal with this score has the standard guarantee.
**Code**: `SimplifiedCQRScore.__call__` returns `max_x |error(x)| / sigma(x)` which
is a deterministic scalar function of (u_pred, u_true). ✅

**⚠️ SUBTLE ISSUE**: `sigma` is estimated from the CALIBRATION set. If the same
calibration set is used to both fit_scale() and calibrate(), the score function depends
on the calibration data, which COULD violate the coverage guarantee. In calibrate.py:
```python
cal_errors = cal_y - cal_pred
score_fn.fit_scale(cal_errors)   # fits sigma from cal data
cp.calibrate(cal_pred, cal_y)    # computes q_hat from SAME cal data
```

This is the SAME theoretical concern as learned spectral weights — using calibration
data to define the score function AND compute the quantile. However, for CQR-lite the
effect is milder because sigma is a simple average (not an optimization), and the
coverage violation is typically small in practice.

**FIXED**: Added option to use three-way split for CQR-lite too.

---

## Summary

| Check | Status | Severity | Action |
|-------|--------|----------|--------|
| Theorem 1 (Parseval) | ⚠️ rfft vs fft | MEDIUM | FIXED: Added Parseval correction |
| Theorem 2 (tighter bands) | ⚠️ Weak | LOW | Paper clarification (Week 5) |
| Theorem 3 (ACI coverage) | ✅ | — | None |
| Theorem 4 (per-freq adapt) | ⚠️ Partial | LOW | Week 4 enhancement |
| Gap 6 → SpectralScore | ✅ | — | None |
| Gap 2 → SpectralACI | ✅ | — | None |
| Score formula match | ✅ (after fix) | — | None |
| ACI update rule | ✅ | — | None |
| Coverage guarantee | ✅ | — | None |
| Three-way split | ✅ | — | None |
| Keep/discard rules | ⚠️ Design gap | LOW | Documentation |
| CQR-lite cal data reuse | ⚠️ Theory | MEDIUM | FIXED: Three-way split option |

**Critical fixes applied (Session 1 — theory/code consistency)**:
1. **Parseval correction for rfft** — added `_apply_parseval_correction()` helper and
   applied in ALL 4 locations: `SpectralScore._score_single_field`,
   `LearnedSpectralScore.learn_weights`, `LearnedSpectralScore.__call__`,
   `compute_per_frequency_scores`. Now `SpectralScore(uniform)` exactly equals `L2Score`
   (Theorem 1 Parseval claim is valid).
2. **CQR-lite three-way split** — `calibrate.py` now uses `three_way_split` for
   `SimplifiedCQRScore`, same as `LearnedSpectralScore`. Score fitting uses one split,
   conformal quantile uses another. Coverage guarantee preserved.
3. **MC Dropout state restoration** — `mc_dropout_predict` now saves/restores original
   `.p` values on all Dropout layers, and warns if model has no Dropout layers.
4. **run_all.sh data check** — FNO/Darcy training now properly guarded behind
   `if has_data` check (prevents fallthrough to training with missing data).
5. **plot_results.py hero figure** — explicitly prefers `fno_darcy_64` results instead
   of relying on filesystem ordering.
6. **fit_scale docstring** — clarified it accepts signed errors (abs applied internally).

**Critical fixes applied (Session 2 — neuraloperator/PDEBench best practices audit)**:
7. **Channel dim guard** — `ndim == 2` → `ndim == spatial_dims + 1` via shared
   `ensure_channel_dim()` in `src/utils.py`. Old guard missed 2D PDE data (ndim=3).
8. **Input/output normalization** — per-dataset z-score (mean/std) computed from training
   data, saved in checkpoint dict, applied consistently at inference in calibrate.py and
   rollout_eval.py. Denormalization applied before conformal scoring.
9. **Relative L2 loss** — replaced `nn.MSELoss()` with `relative_l2_loss()` that normalizes
   by solution magnitude. Standard for neural operators (neuraloperator library).
10. **Reproducibility seed** — `torch.manual_seed(args.seed)` + `cuda.manual_seed_all()`
    added to train.py. Default seed=42.
11. **Parseval correction for odd N** — `_apply_parseval_correction` now handles both
    even N (DC=1, interior=2, Nyquist=1) and odd N (DC=1, all others=2).
12. **Checkpoint format upgrade** — saves dict with `model_state_dict`, `x_mean`, `x_std`,
    `y_mean`, `y_std`, `in_channels`. `load_trained_model` handles both legacy and new format.
13. **Grid coordinates as input channels** — `append_grid()` in `src/utils.py` appends
    spatial meshgrid on [0,1]^d as extra channels. Standard FNO practice from neuraloperator.
    `in_channels = 1 + spatial_dims` for all models.
14. **Training epochs** — 100 → 500 (matching neuraloperator/PDEBench standard).
    Training budgets in run_all.sh updated to 5400s (90 min per model).
