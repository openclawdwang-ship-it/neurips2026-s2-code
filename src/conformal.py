"""
Split Conformal Prediction wrapper for neural operators.

Provides the standard CP machinery:
- Split conformal with arbitrary nonconformity scores
- Three-way split for learned spectral weights
- Prediction band construction
"""

import torch
import numpy as np
from typing import Optional, Tuple
from .scores import NonconformityScore, LearnedSpectralScore


class SplitConformalPredictor:
    """Split conformal prediction for function-valued outputs.

    Coverage guarantee: P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha
    for any distribution, with n calibration points.

    This holds for ANY measurable nonconformity score, including
    spectral scores (FFT is a measurable linear transformation).
    """

    def __init__(self, score_fn: NonconformityScore, alpha: float = 0.1):
        """
        Args:
            score_fn: Nonconformity score function
            alpha: Miscoverage level (default 0.1 for 90% coverage)
        """
        self.score_fn = score_fn
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self._cal_scores: Optional[torch.Tensor] = None

    def calibrate(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
        """Calibrate conformal predictor on calibration set.

        Args:
            u_pred: Predictions on calibration set, shape (n_cal, *spatial)
            u_true: Ground truth on calibration set, same shape

        Returns:
            Conformal quantile q_hat
        """
        scores = self.score_fn(u_pred, u_true)  # (n_cal,)
        self._cal_scores = scores.detach().cpu()
        n = len(scores)

        # Quantile level: ceil((1-alpha)(n+1)) / n
        quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        quantile_level = min(quantile_level, 1.0)

        self.q_hat = float(torch.quantile(scores.float(), quantile_level).item())
        return self.q_hat

    def predict(self, u_pred: torch.Tensor) -> dict:
        """Construct prediction bands.

        For scalar nonconformity scores, the prediction band is:
        C(x) = { u : s(u_pred, u) <= q_hat }

        For norm-based scores (L2, spectral), this defines a ball in function
        space. The pointwise half-width is an approximation assuming uniform
        spatial contribution — valid for L2 scores (Parseval), approximate
        for weighted spectral scores. For spectral scores, the true band
        shape varies by frequency; this uniform approximation is used for
        visualization and reporting.

        Returns:
            dict with 'q_hat', 'u_pred', and 'band_half_width'
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() first")

        # Approximate pointwise half-width assuming uniform spatial contribution.
        # For L2 scores: exact (by Parseval's theorem).
        # For spectral scores: approximate — true band is tighter at low
        # frequencies and wider at high frequencies. This is a reporting
        # convenience; the COVERAGE guarantee is exact regardless.
        n_points = np.prod(u_pred.shape[1:])
        pointwise_half_width = np.sqrt(self.q_hat / n_points)

        return {
            "q_hat": self.q_hat,
            "u_pred": u_pred,
            "band_half_width": pointwise_half_width,
            "u_lower": u_pred - pointwise_half_width,
            "u_upper": u_pred + pointwise_half_width,
        }

    def evaluate_coverage(
        self, u_pred: torch.Tensor, u_true: torch.Tensor
    ) -> dict:
        """Evaluate empirical coverage and band width on test set."""
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() first")

        scores = self.score_fn(u_pred, u_true)
        covered = (scores <= self.q_hat).float()

        n_points = np.prod(u_pred.shape[1:])
        half_width = np.sqrt(self.q_hat / n_points)

        return {
            "empirical_coverage": covered.mean().item(),
            "avg_band_width": 2 * half_width,
            "calibration_error": abs(covered.mean().item() - (1 - self.alpha)),
            "n_test": len(scores),
            "q_hat": self.q_hat,
        }


def three_way_split(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    frac_weight: float = 0.2,
    frac_cal: float = 0.5,
    seed: int = 42,
) -> Tuple[dict, dict, dict]:
    """Split data into weight-learning, calibration, and test sets.

    Required when using LearnedSpectralScore to preserve coverage guarantees.
    Using calibration data for both weight optimization AND conformal
    calibration INVALIDATES the finite-sample coverage guarantee.

    Args:
        u_pred: All predictions, shape (N, *spatial)
        u_true: All ground truth, same shape
        frac_weight: Fraction for weight learning (default 0.2)
        frac_cal: Fraction for calibration (default 0.5)
        seed: Random seed

    Returns:
        (weight_data, cal_data, test_data) — each a dict with 'pred' and 'true'
    """
    n = len(u_pred)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)

    n_weight = int(n * frac_weight)
    n_cal = int(n * frac_cal)

    idx_weight = perm[:n_weight]
    idx_cal = perm[n_weight : n_weight + n_cal]
    idx_test = perm[n_weight + n_cal :]

    def _subset(idx):
        return {"pred": u_pred[idx], "true": u_true[idx]}

    return _subset(idx_weight), _subset(idx_cal), _subset(idx_test)
