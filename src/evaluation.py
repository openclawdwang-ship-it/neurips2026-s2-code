"""
Evaluation metrics for conformal prediction on neural operators.

Metrics:
- Empirical coverage (marginal, pointwise, per-frequency)
- Band width (average, by timestep, by frequency band)
- Calibration error
- Computational overhead
"""

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .scores import NonconformityScore, compute_per_frequency_scores


@dataclass
class CoverageResult:
    """Results from a coverage evaluation."""
    empirical_coverage: float
    target_coverage: float
    calibration_error: float
    avg_band_width: float
    n_test: int
    per_sample_covered: Optional[np.ndarray] = None


def evaluate_coverage(
    score_fn: NonconformityScore,
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    q_hat: float,
    alpha: float = 0.1,
) -> CoverageResult:
    """Evaluate marginal coverage of a conformal predictor.

    Coverage = fraction of test samples whose score <= q_hat.
    """
    scores = score_fn(u_pred, u_true).detach().cpu()
    covered = (scores <= q_hat).numpy()

    return CoverageResult(
        empirical_coverage=float(covered.mean()),
        target_coverage=1.0 - alpha,
        calibration_error=abs(float(covered.mean()) - (1.0 - alpha)),
        avg_band_width=2.0 * np.sqrt(q_hat / np.prod(u_pred.shape[1:])),
        n_test=len(scores),
        per_sample_covered=covered,
    )


def evaluate_per_frequency_coverage(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    q_hat: float,
    n_bins: int = 8,
    spatial_dims: int = 2,
) -> dict:
    """Evaluate coverage decomposed by frequency band.

    This is a KEY diagnostic for spectral conformal prediction:
    shows which frequency bands maintain/lose coverage.
    """
    freq_info = compute_per_frequency_scores(
        u_pred, u_true, n_bins=n_bins, spatial_dims=spatial_dims
    )

    # For each frequency band, check if the band-specific score
    # is below the proportional share of q_hat
    band_scores = freq_info["band_scores"]  # (n_bins, batch)
    n_bins_actual = band_scores.shape[0]

    # Proportional quantile per band (assuming uniform budget allocation)
    q_per_band = q_hat / n_bins_actual

    per_band_coverage = []
    per_band_avg_score = []
    for b in range(n_bins_actual):
        covered = (band_scores[b] <= q_per_band).float().mean().item()
        avg_score = band_scores[b].mean().item()
        per_band_coverage.append(covered)
        per_band_avg_score.append(avg_score)

    return {
        "bin_edges": freq_info["bin_edges"].numpy(),
        "per_band_coverage": np.array(per_band_coverage),
        "per_band_avg_score": np.array(per_band_avg_score),
        "per_band_avg_power": freq_info["band_power"].mean(dim=1).numpy(),
    }


def evaluate_rollout(
    rollout_results: List[dict],
    alpha: float = 0.1,
) -> dict:
    """Aggregate metrics from a Spectral ACI rollout.

    Returns per-timestep coverage and band width evolution.
    """
    timesteps = []
    coverages = []
    band_widths = []
    alpha_ts = []
    q_hat_ts = []

    for r in rollout_results:
        timesteps.append(r["timestep"])
        alpha_ts.append(r["alpha_t"])
        q_hat_ts.append(r["q_hat_t"])
        band_widths.append(r["band_half_width"] * 2)

        if "coverage_t" in r:
            coverages.append(r["coverage_t"])

    result = {
        "timesteps": np.array(timesteps),
        "alpha_t": np.array(alpha_ts),
        "q_hat_t": np.array(q_hat_ts),
        "band_width": np.array(band_widths),
    }

    if coverages:
        result["coverage"] = np.array(coverages)
        result["avg_coverage"] = np.mean(coverages)
        result["min_coverage"] = np.min(coverages)
        result["calibration_error_per_step"] = np.abs(
            np.array(coverages) - (1.0 - alpha)
        )

    return result


def compute_all_metrics(
    score_fn: NonconformityScore,
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    q_hat: float,
    alpha: float = 0.1,
    spatial_dims: int = 2,
    n_freq_bins: int = 8,
) -> dict:
    """Compute all metrics for a single experiment configuration."""
    coverage = evaluate_coverage(score_fn, u_pred, u_true, q_hat, alpha)
    freq_coverage = evaluate_per_frequency_coverage(
        u_pred, u_true, q_hat,
        n_bins=n_freq_bins, spatial_dims=spatial_dims,
    )

    return {
        "coverage": coverage,
        "freq_coverage": freq_coverage,
    }
