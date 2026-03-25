# Experimental Principles

Adapted from [autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026) for NeurIPS 2026 conformal prediction research.

---

## Core Philosophy

> The agent modifies code, runs experiments, checks if results improved, keeps or discards, and repeats. You wake up to a log of experiments and a better model.

We adopt the same loop but adapt the metric structure and time budgets for conformal prediction on neural operators.

---

## 1. Fixed Time Budget per Experiment

**autoresearch rule**: Every training run is exactly 5 minutes wall clock. This makes all experiments directly comparable regardless of what changed.

**Our adaptation**:

| Phase | Time Budget | What Runs |
|-------|-------------|-----------|
| Model training (FNO/TFNO/DeepONet) | 30 min per PDE | Fixed — train once, don't re-train per CP experiment |
| Conformal calibration + evaluation | 2 min | Score computation, quantile calibration, coverage eval |
| Rollout evaluation (Spectral ACI) | 5 min | 20-step auto-regressive rollout + per-step coverage |
| Full ablation point | 10 min | Train + calibrate + evaluate one configuration |

**Rule**: Never let a single experiment exceed its budget. If it takes longer, kill and discard. This prevents the agent from wasting hours on a broken idea.

---

## 2. Single Primary Metric with Hard Constraints

**autoresearch rule**: One metric — `val_bpb`. Lower is better. Period.

**Our adaptation**: We have two metrics in a **lexicographic order**:

1. **Hard constraint**: Empirical coverage ≥ (1 - α - 0.02). If coverage drops below target minus tolerance, the experiment is **invalid** regardless of band width. Discard.
2. **Primary metric**: Average band width. **Lower is better** (tighter bands = more informative UQ).

**Decision rule**:
```
if coverage < (1 - alpha) - 0.02:
    status = "invalid"  # coverage violation, discard
elif band_width < best_band_width:
    status = "keep"     # tighter bands, advance
elif band_width == best_band_width and code_is_simpler:
    status = "keep"     # simplification win
else:
    status = "discard"  # no improvement, revert
```

For rollout experiments, the metric is **minimum coverage across all timesteps** (not average) — the worst timestep matters most.

---

## 3. Keep / Discard with Git Commits

**autoresearch rule**: Every experiment gets a git commit BEFORE running. If it improves, keep the commit (advance branch). If it doesn't, `git reset` back. The branch is monotonically improving.

**Our adaptation**: Same mechanism, applied to the S2 branch.

```
LOOP:
  1. Read current state: best coverage, best band_width
  2. Edit code (scores.py, spectral_aci.py, configs, etc.)
  3. git commit -m "experiment: <description>"
  4. Run: python scripts/calibrate.py > run.log 2>&1
  5. Extract metrics: grep "coverage\|band_width" run.log
  6. Log to results.tsv
  7. If improved → keep commit, update best
  8. If not → git reset --hard HEAD~1
```

**Branch naming**: `experiment/s2-spectral-<date>` (e.g. `experiment/s2-spectral-mar26`)

---

## 4. Simplicity Criterion

**autoresearch rule**: "All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome."

**Our adaptation**: This is critical for a NeurIPS paper. Reviewers reward clean methods.

| Scenario | Decision |
|----------|----------|
| 0.5% tighter bands + 20 lines of hacky code | Discard |
| 0.5% tighter bands from deleting code | Keep |
| Equal bands but much simpler score function | Keep |
| 2%+ tighter bands even with added complexity | Keep |
| New score function with identical performance | Discard (prefer existing) |

**Concrete application**: If `SpectralScore(weight_type="sobolev", s=1)` matches `LearnedSpectralScore` in band width, **keep the Sobolev version** — it's closed-form, has a clean theorem, and doesn't need the three-way split.

---

## 5. Results Logging: results.tsv

**autoresearch format**: Tab-separated, one row per experiment.

**Our format** (adapted for conformal prediction):

```
commit	pde	model	score	coverage	band_width	cal_error	status	description
a1b2c3d	darcy	fno	l2	0.912	0.0423	0.012	keep	baseline L2 score
b2c3d4e	darcy	fno	sobolev_1	0.905	0.0387	0.005	keep	Sobolev H^1 tighter by 8.5%
c3d4e5f	darcy	fno	learned	0.891	0.0352	0.009	invalid	coverage below threshold
```

**Rules**:
- Tab-separated (commas break in descriptions)
- `results.tsv` is NOT committed to git (untracked)
- One file per PDE, or one combined file with PDE column
- Log crashes as `coverage=0.000, band_width=0.000, status=crash`
- Always record the commit hash — enables reproducing any experiment

For rollout experiments, add columns: `min_coverage`, `final_band_width`, `n_steps`.

---

## 6. NEVER STOP

**autoresearch rule**: "Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder."

**Our adaptation**: Same principle, with a prioritized idea queue.

**Idea priority for S2 (Spectral Conformal)**:
1. Baseline: L2 score on Darcy (establish ground truth)
2. Spectral uniform weights (verify Parseval equivalence to L2)
3. Sobolev H^1 (s=1) — expect tighter bands if FNO has spectral bias
4. Sobolev H^2 (s=2) — test stronger frequency emphasis
5. Inverse power spectrum weights — test opposite direction
6. Learned spectral weights (three-way split) — expect tightest
7. Cross-architecture: repeat best score on TFNO, DeepONet
8. Rollout: Static L2 vs Static Spectral vs Spectral ACI on Navier-Stokes
9. Ablation: calibration set size (100, 500, 1000, 5000)
10. Ablation: resolution sensitivity (32, 64, 128)

**If stuck**: Re-read the spectral bias literature (arXiv 2404.07200, 2602.19265) for new weight function ideas. Try combining spectral + PDE residual scores. Try per-channel spectral weights for multi-field PDEs.

---

## 7. What the Agent Can and Cannot Modify

**autoresearch rule**: Agent edits ONE file (train.py). Everything else is read-only.

**Our adaptation**: More files are editable, but with clear boundaries.

| File | Editable? | What |
|------|-----------|------|
| `src/scores.py` | YES | Core contribution — score functions |
| `src/spectral_aci.py` | YES | ACI logic |
| `src/conformal.py` | YES | CP wrapper |
| `configs/*.yaml` | YES | Hyperparameters |
| `scripts/*.py` | YES (carefully) | Experiment scripts |
| `src/data.py` | NO | Data loading is fixed |
| `src/models.py` | NO (after initial training) | Model architecture is fixed per experiment set |
| `src/evaluation.py` | NO | Metrics are ground truth |
| `results.tsv` | APPEND ONLY | Never delete rows |
| `RESEARCH_PLAN.md` | NO (during experiments) | Plan is fixed; update only between experiment campaigns |

---

## 8. Crash Recovery

**autoresearch rule**: "If it's something dumb and easy to fix (typo, missing import), fix and re-run. If the idea itself is fundamentally broken, just skip it."

**Our adaptation**: Same. Plus:
- OOM on RTX 3090: reduce batch size or resolution, not the experiment
- NaN in FFT: check for zero-padding issues or denormalized inputs
- Coverage collapse (0% coverage): likely a score function bug, not a conceptual failure — debug before discarding the idea
- Training divergence: not our problem (we use pre-trained models)

---

## 9. Adaptation for A1 (if we run both S2 and A1)

If we also pursue A1 (Optimizers Across Architectures), the same principles apply with these changes:

| Principle | S2 (Spectral Conformal) | A1 (Optimizers) |
|-----------|------------------------|-----------------|
| Time budget | 2 min per calibration experiment | 30 min per training run |
| Primary metric | Band width (lower = better) | Val loss (lower = better) |
| Hard constraint | Coverage ≥ 1-α-0.02 | Training must converge (no NaN) |
| Editable file | `src/scores.py` | `train.py` (optimizer config) |
| Fixed file | `src/evaluation.py` | `evaluate.py` (metrics) |
| Branch | `experiment/s2-spectral-<date>` | `experiment/a1-optimizers-<date>` |

---

## 10. Summary: The Loop

```
SETUP:
  git checkout -b experiment/s2-spectral-<date>
  Train models (one-time, fixed)
  Run baseline (L2 score), record in results.tsv

LOOP FOREVER:
  1. Pick next idea from priority queue
  2. Edit score function / conformal config
  3. git commit -m "experiment: <description>"
  4. Run: python scripts/calibrate.py > run.log 2>&1
  5. Extract: grep "coverage\|band_width" run.log
  6. Check: coverage >= threshold?
     - NO → status=invalid, git reset, next idea
     - YES → band_width < best?
       - YES → status=keep, update best, advance branch
       - NO → status=discard, git reset
  7. Append to results.tsv
  8. NEVER STOP. If out of ideas, think harder.
```
