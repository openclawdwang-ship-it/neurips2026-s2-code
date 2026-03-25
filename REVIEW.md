# Self-Review (2026-03-25)

## Audit Scope
Reviewed: RESEARCH_PLAN.md, literature-review.md, PRINCIPLES.md, src/*.py, scripts/*.py

---

## 1. Literature Completeness

### Verified OK
- **Core gap confirmed**: Fresh web search (2026-03-25) confirms zero papers combine spectral/Fourier nonconformity scores with conformal prediction. Searched 5 query variants across arXiv, Google Scholar, Semantic Scholar.
- **6 direct competitors correctly identified**: Conformalized-DeepONet, Ma et al., Millard et al., CMCO, LSCI, CoNBONet.
- **CP-PRE (ICML 2025) correctly flagged** as critical competitor.
- **COPA 2026 deadline is May 1** — no accepted papers yet. Risk window exists.

### Fixed
- **CP-PRE limitations were wrong**: literature-review.md said "static evaluations only" but CP-PRE does 20-step rollouts. Fixed to accurately describe their static per-step recalibration.
- **Gap 2 description was imprecise**: Said "completely unstudied" for temporal conformal on rollouts. Fixed to acknowledge CP-PRE does rollouts but without ACI's adaptive mechanism.
- **Missing paper**: Added arXiv 2412.10459 (Dec 2024) — CP comparison across FNO/UNO/TFNO with standard scores. Relevant to C3 benchmark.
- **Spectral bias papers were vague**: Added specific arXiv IDs (2404.07200, 2602.19265) with descriptions of what they actually contain vs. what they don't.

### Remaining Risk
- **COPA 2026 submissions close May 1**: Someone could submit spectral conformal to COPA. Low risk (niche venue) but worth monitoring.
- **arXiv 2602.19265 (ICLR 2026)**: This paper defines frequency-resolved error metrics e_{F,p} = Σ_k |k|^p |ê_k|² — structurally identical to our spectral score but used only as a diagnostic. A reviewer could argue our contribution is "just using their metric as a conformal score." Mitigation: emphasize the theoretical framework (4 theorems), the unification with Sobolev norms, and the ACI integration.

---

## 2. Experimental Design

### Verified OK
- **5 PDEs, 4 architectures, 7 ablations**: Comprehensive enough for NeurIPS.
- **Metrics table**: Complete (coverage, band width, calibration error, overhead, PDE residual).
- **Budget estimate** ($15-22): Realistic for RTX 3090.
- **autoresearch-style experiment loop**: Well-adapted with lexicographic metric ordering.

### Issues Found and Fixed
- **conformal.py band width computation**: Used `sqrt(q_hat / n_points)` as pointwise half-width. This is exact for L2 scores (Parseval) but only approximate for spectral scores where true bands vary by frequency. Added clear documentation that coverage is exact regardless, and this is a reporting approximation.

### Issues Found — NOT Fixed (need attention during implementation)
1. **CQR baseline may be impractical**: CQR (Romano et al.) requires training a quantile regression neural operator, which means training a NEW model with quantile loss — not just wrapping an existing model. This could blow the budget. **Recommendation**: Drop CQR or replace with a simpler quantile baseline (e.g., conformalized quantile from the existing model's ensemble).

2. **DeepONet negative control framing**: Plan says "use DeepONet as negative control for spectral scores since it works in spatial domain." But spectral scores apply to ANY function output via FFT — the question is whether spectral WEIGHTING helps when the model doesn't have spectral bias. The experiment is valid; the framing should be "spectral weighting helps FNO more than DeepONet because FNO has spectral bias and DeepONet doesn't."

3. **Theorem 2 precision**: Claims "E[width | spectral] ≤ E[width | L2] when weights proportional to inverse error power spectrum." The inverse error power spectrum is estimated from calibration data, not known exactly. The theorem should state the condition on the ESTIMATED weights, with a finite-sample approximation bound. This is a theory subtlety to address in Week 5.

4. **Shallow Water data availability**: PDEBench's 2D Shallow Water dataset may have different format or availability than assumed. Verify during Week 1 setup.

---

## 3. PRINCIPLES.md Experiment Loop

### Verified OK
- **Lexicographic metric**: Coverage constraint → band width minimization. Correct.
- **Three-way split for learned weights**: Documented and enforced.
- **10-step idea priority queue**: Logical progression from simple to complex.
- **File editability rules**: Clear boundaries.

### Issues Found and Fixed
None — PRINCIPLES.md is clean.

### Minor Concern
- **Coverage tolerance of 0.02**: Slightly arbitrary. In CP literature, any coverage ≥ 1-α is valid (by theory). Using a tolerance means we might discard experiments that are theoretically valid but have finite-sample variance. **Recommendation**: Keep the tolerance for the autonomous loop (prevents chasing noise), but report ALL coverage values in the paper regardless.

---

## 4. Code Quality

### Verified OK
- All 9 Python files parse correctly (AST check passed).
- `scores.py`: SpectralScore correctly uses rfft/rfft2/rfftn and computes weighted power spectrum.
- `spectral_aci.py`: ACI update rule matches Gibbs & Candes 2021.
- `conformal.py`: Three-way split implemented correctly.
- `data.py`: Handles static/one_step/trajectory modes for all 5 PDEs.

### Issues Found — NOT Fixed (need attention during implementation)
1. **scores.py LearnedSpectralScore optimization objective**: Currently minimizes score variance. This is a reasonable heuristic (concentrated distribution → tighter bands) but not provably optimal. Alternative: minimize the (1-α) quantile of scores directly. Low priority — can be refined during experiments.

2. **spectral_aci.py per-frequency weight adaptation**: Currently tracks per-frequency coverage but doesn't actually adapt the weights w_k(t) during rollout — it only adapts the overall alpha_t via ACI. Adding per-frequency weight adaptation is a Week 4 enhancement, not a bug.

3. **models.py DeepONet**: The custom implementation is simplified (3-layer MLP branch/trunk). May underperform compared to the DeepXDE library's implementation. Should compare during Week 1.

---

## 5. Risk Summary

| Risk | Level | Status |
|------|-------|--------|
| Spectral gap scooped before submission | LOW (5-10%) | Confirmed vacant as of 2026-03-25 |
| CP-PRE conflation by reviewers | MEDIUM | Mitigation: explicit baseline comparison in experiments |
| Spectral scores don't improve tightness | LOW (15%) | Strong theoretical grounding; FNO spectral bias is well-documented |
| arXiv 2602.19265 "just using their metric" critique | MEDIUM | Mitigation: 4 theorems + ACI integration + benchmark go far beyond diagnostic use |
| CQR baseline infeasible within budget | HIGH | Recommendation: drop or simplify |
| Theorem 2 finite-sample gap | LOW | Addressable in proof with standard concentration inequalities |

---

## 6. Changes Made During This Review

1. **literature-review.md**: Fixed CP-PRE description (was "static only", now correctly notes 20-step rollouts with static recalibration). Added paper 1.8 (arXiv 2412.10459). Added specific arXiv IDs for spectral bias papers in Section 4.2. Fixed Gap 2 description.
2. **conformal.py**: Added documentation clarifying that pointwise band width is exact for L2 and approximate for spectral scores, with note that coverage guarantee is unaffected.
3. **No changes to RESEARCH_PLAN.md or PRINCIPLES.md** — both are internally consistent after the literature fixes.

---

## 7. Verdict

**Ready to proceed to RunPod implementation.** All critical issues are documented. The spectral gap is confirmed as of today. The experiment design is sound. The main risks (CQR baseline, Theorem 2 precision, per-frequency weight adaptation) are Week 4-5 concerns, not blockers for starting.
