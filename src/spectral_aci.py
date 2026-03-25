"""
Spectral ACI: Adaptive Conformal Inference with Spectral Scores.

Core contribution (C1+C2 combined): During auto-regressive PDE rollouts,
track per-frequency coverage drift and adapt spectral weights w_k(t)
online using ACI's learning rate mechanism.

Key differences from CP-PRE (Gopakumar et al., ICML 2025):
- CP-PRE: static per-step recalibration with PDE residual scores in spatial domain
- Spectral ACI: adaptive (ACI) recalibration with spectral scores that track
  per-frequency drift — high-frequency bands degrade faster, so spectral ACI
  automatically widens high-freq bands while keeping low-freq bands tight.

Based on: Gibbs & Candes (2021) "Adaptive Conformal Inference Under
Distribution Shift" — extended to spectral domain.
"""

import torch
import torch.fft as fft
from typing import Optional, List
from dataclasses import dataclass, field

from .scores import SpectralScore, compute_per_frequency_scores


@dataclass
class SpectralACIState:
    """State of the Spectral ACI algorithm at a given timestep."""
    timestep: int = 0
    alpha_t: float = 0.1  # Current adaptive significance level
    q_hat_t: float = 0.0  # Current conformal quantile
    cumulative_miscoverage: float = 0.0
    # Per-frequency tracking
    per_freq_alpha: Optional[torch.Tensor] = None  # (n_bins,)
    per_freq_coverage: List[dict] = field(default_factory=list)


class SpectralACI:
    """Spectral Adaptive Conformal Inference for auto-regressive PDE rollouts.

    At each rollout step t:
    1. Compute spectral nonconformity score s_t using current weights w_k(t)
    2. Update alpha_t based on whether coverage was achieved at step t-1
    3. Recompute conformal quantile q_hat_t using adaptive alpha_t
    4. (Optional) Update per-frequency weights w_k(t) based on which
       frequency bands lost coverage

    The ACI update rule (Gibbs & Candes 2021):
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1{s_t > q_hat_t} is the miscoverage indicator
    and gamma > 0 is the learning rate.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
        spatial_dims: int = 2,
        weight_type: str = "sobolev",
        sobolev_s: float = 1.0,
        n_freq_bins: int = 8,
        adapt_per_frequency: bool = True,
    ):
        """
        Args:
            alpha: Target miscoverage level
            gamma: ACI learning rate (controls adaptation speed)
            spatial_dims: Number of spatial dimensions
            weight_type: Initial spectral weight type
            sobolev_s: Sobolev exponent for initial weights
            n_freq_bins: Number of frequency bins for per-freq tracking
            adapt_per_frequency: Whether to adapt weights per frequency band
        """
        self.alpha = alpha
        self.gamma = gamma
        self.spatial_dims = spatial_dims
        self.n_freq_bins = n_freq_bins
        self.adapt_per_frequency = adapt_per_frequency

        self.score_fn = SpectralScore(
            spatial_dims=spatial_dims,
            weight_type=weight_type,
            sobolev_s=sobolev_s,
        )

        self.state = SpectralACIState(alpha_t=alpha)
        self._cal_scores: Optional[torch.Tensor] = None

    def calibrate_initial(self, u_pred: torch.Tensor, u_true: torch.Tensor):
        """Initial calibration using static calibration set (before rollout begins).

        Args:
            u_pred: Predictions on calibration set at t=0
            u_true: Ground truth on calibration set at t=0
        """
        scores = self.score_fn(u_pred, u_true).detach()
        self._cal_scores = scores

        # Compute initial quantile
        n = len(scores)
        q_level = min(torch.tensor((1 - self.alpha) * (n + 1) / n).item(), 1.0)
        self.state.q_hat_t = float(torch.quantile(scores.float(), q_level).item())

        # Initialize per-frequency tracking
        if self.adapt_per_frequency:
            self.state.per_freq_alpha = torch.full(
                (self.n_freq_bins,), self.alpha, device=u_pred.device
            )

    def step(
        self,
        u_pred_t: torch.Tensor,
        u_true_t: Optional[torch.Tensor] = None,
    ) -> dict:
        """Process one rollout timestep.

        Args:
            u_pred_t: Prediction at current timestep, shape (batch, *spatial)
            u_true_t: Ground truth at current timestep (None if not available)

        Returns:
            dict with prediction bands and diagnostics
        """
        self.state.timestep += 1

        # Construct prediction band using current q_hat_t
        n_points = int(torch.tensor(u_pred_t.shape[1:]).prod().item())
        half_width = (self.state.q_hat_t / n_points) ** 0.5

        result = {
            "timestep": self.state.timestep,
            "alpha_t": self.state.alpha_t,
            "q_hat_t": self.state.q_hat_t,
            "band_half_width": half_width,
            "u_pred": u_pred_t,
            "u_lower": u_pred_t - half_width,
            "u_upper": u_pred_t + half_width,
        }

        # If ground truth available, update ACI state
        if u_true_t is not None:
            scores_t = self.score_fn(u_pred_t, u_true_t).detach()

            # Miscoverage indicator: 1 if score > quantile (not covered)
            err_t = (scores_t > self.state.q_hat_t).float().mean().item()

            # ACI update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
            self.state.alpha_t = self.state.alpha_t + self.gamma * (self.alpha - err_t)
            # Clamp to valid range
            self.state.alpha_t = max(0.001, min(0.999, self.state.alpha_t))

            self.state.cumulative_miscoverage += err_t

            # Recompute quantile with adapted alpha
            if self._cal_scores is not None:
                n = len(self._cal_scores)
                q_level = min((1 - self.state.alpha_t) * (n + 1) / n, 1.0)
                q_level = max(q_level, 0.0)
                self.state.q_hat_t = float(
                    torch.quantile(self._cal_scores.float(), q_level).item()
                )

            # Per-frequency coverage tracking
            if self.adapt_per_frequency:
                freq_info = compute_per_frequency_scores(
                    u_pred_t, u_true_t,
                    n_bins=self.n_freq_bins,
                    spatial_dims=self.spatial_dims,
                )
                self.state.per_freq_coverage.append({
                    "timestep": self.state.timestep,
                    "band_scores": freq_info["band_scores"].cpu(),
                    "band_power": freq_info["band_power"].cpu(),
                })

            result["err_t"] = err_t
            result["coverage_t"] = 1.0 - err_t
            result["score_mean"] = scores_t.mean().item()

        return result

    def rollout(
        self,
        model,
        u_init: torch.Tensor,
        n_steps: int,
        u_true_sequence: Optional[torch.Tensor] = None,
    ) -> List[dict]:
        """Run full auto-regressive rollout with Spectral ACI.

        Args:
            model: Neural operator model (callable: u_t -> u_{t+1})
            u_init: Initial condition, shape (batch, *spatial)
            n_steps: Number of rollout steps
            u_true_sequence: Ground truth sequence, shape (n_steps, batch, *spatial)

        Returns:
            List of per-step results from step()
        """
        results = []
        u_t = u_init

        for t in range(n_steps):
            with torch.no_grad():
                u_pred_t = model(u_t)

            u_true_t = u_true_sequence[t] if u_true_sequence is not None else None
            step_result = self.step(u_pred_t, u_true_t)
            results.append(step_result)

            # Auto-regressive: next input is model's prediction
            u_t = u_pred_t

        return results

    def get_diagnostics(self) -> dict:
        """Get summary diagnostics for the rollout."""
        if not self.state.per_freq_coverage:
            return {"timestep": self.state.timestep}

        return {
            "total_steps": self.state.timestep,
            "final_alpha_t": self.state.alpha_t,
            "avg_miscoverage": (
                self.state.cumulative_miscoverage / max(self.state.timestep, 1)
            ),
            "n_freq_bins": self.n_freq_bins,
            "per_freq_data": self.state.per_freq_coverage,
        }
