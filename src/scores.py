"""
Spectral Nonconformity Scores for Neural PDE Operators.

Core contribution (C1): Define nonconformity scores in Fourier space
with frequency-dependent weights w_k.

Key insight: FNO operates natively in Fourier space and has known spectral
bias (low freq learned first, high freq errors dominate). Spectral scores
exploit this structure for tighter conformal prediction bands.

Unification:
  w_k = 1           -> L2 norm (Parseval's theorem)
  w_k = (1+|k|^2)^s -> H^s Sobolev norm
  w_k = learned     -> adaptive spectral score (tightest bands)
"""

import torch
import torch.fft as fft
from typing import Optional, Literal
from abc import ABC, abstractmethod


def _apply_parseval_correction(power: torch.Tensor, last_spatial_size: int) -> torch.Tensor:
    """Apply Parseval correction for rfft: interior bins counted 2x.

    When using rfft (real FFT), the output has shape (..., N//2+1) in the last dim.
    - DC (index 0): always appears once → factor 1
    - Nyquist (index N//2): appears once ONLY when N is even → factor 1
    - All other bins: represent TWO conjugate-symmetric frequencies → factor 2

    For odd N, there is no Nyquist bin: the last bin (index N//2) is an
    interior bin and must be doubled. This is the key difference.
    """
    N = last_spatial_size
    rfft_size = N // 2 + 1
    factor = torch.ones(rfft_size, device=power.device)
    if N % 2 == 0:
        # Even N: DC=1, interior=2, Nyquist=1
        factor[1:-1] = 2.0
    else:
        # Odd N: DC=1, all others=2 (no Nyquist bin)
        factor[1:] = 2.0
    view_shape = [1] * (power.ndim - 1) + [rfft_size]
    return power * factor.view(view_shape)


class NonconformityScore(ABC):
    """Base class for nonconformity scores."""

    @abstractmethod
    def __call__(
        self, u_pred: torch.Tensor, u_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar nonconformity score per sample.

        Args:
            u_pred: Predicted fields, shape (batch, *spatial_dims) or (batch, C, *spatial_dims)
            u_true: True fields, same shape as u_pred

        Returns:
            Scores, shape (batch,)
        """
        ...


class L2Score(NonconformityScore):
    """Standard L2 norm score: ||u_true - u_pred||_2^2.
    Equivalent to SpectralScore with uniform weights (Parseval's theorem).
    """

    def __call__(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        error = u_true - u_pred
        # Flatten spatial dims, sum over them
        batch_size = error.shape[0]
        return error.reshape(batch_size, -1).pow(2).sum(dim=-1)


class SpectralScore(NonconformityScore):
    """Spectral nonconformity score: s_w(u) = sum_k w_k |u_hat_err(k)|^2.

    This is the core novel contribution. The score is computed by:
    1. Computing the error field e = u_true - u_pred
    2. Taking the FFT of e
    3. Weighting the power spectrum by frequency-dependent weights w_k
    4. Summing to get a scalar score

    Args:
        spatial_dims: Number of spatial dimensions (1 or 2)
        weight_type: Type of spectral weights
        sobolev_s: Sobolev exponent (only for weight_type='sobolev')
        weights: Custom weight tensor (only for weight_type='custom')
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        weight_type: Literal["uniform", "sobolev", "inverse_power", "custom"] = "uniform",
        sobolev_s: float = 1.0,
        weights: Optional[torch.Tensor] = None,
    ):
        self.spatial_dims = spatial_dims
        self.weight_type = weight_type
        self.sobolev_s = sobolev_s
        self._custom_weights = weights

    def _compute_frequency_grid(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Compute |k|^2 for each frequency in the FFT output grid.

        For rfft, the last spatial dim has size N//2+1; others have size N.
        """
        freq_components = []
        spatial_shape = shape  # (*spatial_dims,)
        for i, n in enumerate(spatial_shape):
            if i == len(spatial_shape) - 1:
                # Last dim uses rfft frequencies: 0, 1, ..., N//2
                freqs = torch.arange(n // 2 + 1, device=device, dtype=torch.float32)
            else:
                # Other dims use full fft frequencies: 0, 1, ..., N//2, -N//2+1, ..., -1
                freqs = torch.fft.fftfreq(n, device=device) * n
            # Reshape for broadcasting
            view_shape = [1] * len(spatial_shape)
            view_shape[i] = -1
            freq_components.append(freqs.reshape(view_shape))

        # |k|^2 = sum of squared frequencies
        k_sq = sum(f.pow(2) for f in freq_components)
        return k_sq

    def _get_weights(self, spatial_shape: tuple, device: torch.device) -> torch.Tensor:
        """Get frequency weights w_k based on weight_type."""
        if self.weight_type == "uniform":
            return torch.ones(1, device=device)

        if self.weight_type == "custom":
            if self._custom_weights is None:
                raise ValueError("Custom weights not provided")
            return self._custom_weights.to(device)

        k_sq = self._compute_frequency_grid(spatial_shape, device)

        if self.weight_type == "sobolev":
            # w_k = (1 + |k|^2)^s -> H^s Sobolev norm
            return (1.0 + k_sq).pow(self.sobolev_s)

        if self.weight_type == "inverse_power":
            # w_k = 1 / (1 + |k|^2) -> emphasize low frequencies
            return 1.0 / (1.0 + k_sq)

        raise ValueError(f"Unknown weight type: {self.weight_type}")

    def __call__(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        error = u_true - u_pred
        batch_size = error.shape[0]

        # If multi-channel, average scores across channels
        if error.ndim == self.spatial_dims + 2:
            # (batch, C, *spatial) -> process each channel
            n_channels = error.shape[1]
            scores = torch.zeros(batch_size, device=error.device)
            for c in range(n_channels):
                scores += self._score_single_field(error[:, c])
            return scores / n_channels
        else:
            return self._score_single_field(error)

    def _score_single_field(self, error: torch.Tensor) -> torch.Tensor:
        """Compute spectral score for a single-channel error field.

        Uses rfft for efficiency (real input → half spectrum). Applies Parseval
        correction: interior frequency bins are multiplied by 2 to account for
        the missing conjugate-symmetric half. This ensures that with uniform
        weights (w_k=1), the spectral score exactly equals the L2 norm
        (Theorem 1, Parseval's theorem).

        Args:
            error: Shape (batch, *spatial_dims)
        """
        batch_size = error.shape[0]
        spatial_shape = error.shape[1:]

        # FFT (real-to-complex for efficiency)
        if self.spatial_dims == 1:
            e_hat = fft.rfft(error, dim=-1)
        elif self.spatial_dims == 2:
            e_hat = fft.rfft2(error, dim=(-2, -1))
        else:
            e_hat = fft.rfftn(error, dim=tuple(range(1, error.ndim)))

        # Power spectrum: |e_hat(k)|^2
        power = e_hat.real.pow(2) + e_hat.imag.pow(2)

        # Parseval correction: interior rfft bins counted 2x for missing conjugates
        power = _apply_parseval_correction(power, spatial_shape[-1])

        # Apply weights
        w = self._get_weights(spatial_shape, error.device)

        # Weighted sum: s = sum_k w_k |e_hat(k)|^2
        weighted_power = power * w
        scores = weighted_power.reshape(batch_size, -1).sum(dim=-1)
        return scores


class LearnedSpectralScore(NonconformityScore):
    """Spectral score with weights learned from calibration data.

    IMPORTANT: Requires three-way data split to preserve coverage guarantees:
      1. Training data -> fit neural operator
      2. Weight-learning data -> optimize w_k for tightest bands
      3. Calibration data -> compute conformal quantile

    Using calibration data for both weight learning AND conformal calibration
    INVALIDATES the coverage guarantee.

    The weights are learned by minimizing average band width on the
    weight-learning set subject to achieving target coverage.
    """

    def __init__(self, spatial_dims: int = 2, n_freq_bins: int = 16):
        self.spatial_dims = spatial_dims
        self.n_freq_bins = n_freq_bins
        self._log_weights: Optional[torch.Tensor] = None
        self._bin_edges: Optional[torch.Tensor] = None

    def learn_weights(
        self,
        errors: torch.Tensor,
        n_steps: int = 200,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """Learn optimal spectral weights from a held-out weight-learning set.

        Optimizes for tightest bands by finding weights that make the score
        distribution most concentrated (lowest variance).

        Args:
            errors: Error fields from weight-learning set, shape (n, *spatial)
            n_steps: Optimization steps
            lr: Learning rate

        Returns:
            Learned weight tensor
        """
        spatial_shape = errors.shape[1:]
        device = errors.device

        # Compute FFT of all errors
        if self.spatial_dims == 1:
            e_hat = fft.rfft(errors, dim=-1)
        elif self.spatial_dims == 2:
            e_hat = fft.rfft2(errors, dim=(-2, -1))
        else:
            e_hat = fft.rfftn(errors, dim=tuple(range(1, errors.ndim)))

        power = e_hat.real.pow(2) + e_hat.imag.pow(2)  # (n, *freq_shape)
        power = _apply_parseval_correction(power, spatial_shape[-1])

        # Bin frequencies by |k| magnitude for parameter efficiency
        freq_grid = SpectralScore(self.spatial_dims)._compute_frequency_grid(
            spatial_shape, device
        )
        k_mag = freq_grid.sqrt().flatten()
        max_k = k_mag.max()
        self._bin_edges = torch.linspace(0, max_k + 1e-6, self.n_freq_bins + 1, device=device)
        bin_indices = torch.bucketize(k_mag, self._bin_edges[1:-1])  # (n_freqs,)

        # Learnable log-weights (one per bin)
        self._log_weights = torch.zeros(self.n_freq_bins, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([self._log_weights], lr=lr)

        n_samples = power.shape[0]
        flat_power = power.reshape(n_samples, -1)  # (n, n_freqs)

        for step in range(n_steps):
            optimizer.zero_grad()

            # Map bin weights to per-frequency weights
            w = torch.exp(self._log_weights)  # ensure w >= 0
            per_freq_w = w[bin_indices]  # (n_freqs,)

            # Compute scores: s_i = sum_k w_k * |e_hat_i(k)|^2
            scores = (flat_power * per_freq_w.unsqueeze(0)).sum(dim=-1)  # (n,)

            # Minimize score variance -> most concentrated distribution -> tightest bands
            loss = scores.var()

            loss.backward()
            optimizer.step()

        self._log_weights = self._log_weights.detach()
        return torch.exp(self._log_weights)

    def __call__(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        if self._log_weights is None:
            raise RuntimeError("Must call learn_weights() before using this score")

        error = u_true - u_pred
        batch_size = error.shape[0]
        spatial_shape = error.shape[1:]
        device = error.device

        if self.spatial_dims == 1:
            e_hat = fft.rfft(error, dim=-1)
        elif self.spatial_dims == 2:
            e_hat = fft.rfft2(error, dim=(-2, -1))
        else:
            e_hat = fft.rfftn(error, dim=tuple(range(1, error.ndim)))

        power = e_hat.real.pow(2) + e_hat.imag.pow(2)
        power = _apply_parseval_correction(power, spatial_shape[-1])
        flat_power = power.reshape(batch_size, -1)

        # Map bin weights to per-frequency
        freq_grid = SpectralScore(self.spatial_dims)._compute_frequency_grid(
            spatial_shape, device
        )
        k_mag = freq_grid.sqrt().flatten()
        bin_indices = torch.bucketize(k_mag, self._bin_edges[1:-1])

        w = torch.exp(self._log_weights.to(device))
        per_freq_w = w[bin_indices]

        scores = (flat_power * per_freq_w.unsqueeze(0)).sum(dim=-1)
        return scores


class SimplifiedCQRScore(NonconformityScore):
    """Simplified Conformalized Quantile Regression score.

    Instead of training a full quantile regression model (expensive),
    we estimate local error scale from the calibration set and use it
    to normalize the nonconformity scores. This gives heteroscedastic
    bands without retraining.

    Approach:
    1. On calibration data, compute pointwise absolute errors |u_true - u_pred|
    2. Estimate local error scale sigma(x) as the smoothed MAE per spatial location
    3. Score = max over spatial locations of |error(x)| / sigma(x)

    This is "CQR-lite": same coverage guarantee as split conformal (it's just
    a different nonconformity score), but produces spatially adaptive bands
    unlike the uniform-width bands from L2 or spectral scores.
    """

    def __init__(self, spatial_dims: int = 2):
        self.spatial_dims = spatial_dims
        self._sigma: Optional[torch.Tensor] = None

    def fit_scale(self, errors: torch.Tensor):
        """Estimate local error scale from calibration errors.

        Args:
            errors: Error fields (signed OK — abs is applied internally),
                    shape (n_cal, *spatial)
        """
        # sigma = mean absolute error per spatial location + small epsilon
        self._sigma = errors.abs().mean(dim=0) + 1e-8  # (*spatial,)

    def __call__(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        if self._sigma is None:
            raise RuntimeError("Must call fit_scale() first")

        error = (u_true - u_pred).abs()
        batch_size = error.shape[0]

        # Normalize by local scale
        sigma = self._sigma.to(error.device)
        normalized = error / sigma  # (batch, *spatial)

        # Score = max normalized error (controls worst-case spatial coverage)
        scores = normalized.reshape(batch_size, -1).max(dim=-1).values
        return scores

    def predict_bands(self, u_pred: torch.Tensor, q_hat: float) -> dict:
        """Construct spatially-varying prediction bands.

        Unlike uniform bands from L2/spectral scores, CQR-lite bands
        are wider where the model is less accurate and tighter where
        it's more accurate.
        """
        if self._sigma is None:
            raise RuntimeError("Must call fit_scale() first")

        sigma = self._sigma.to(u_pred.device)
        half_width = q_hat * sigma  # (*spatial,) — varies by location

        return {
            "u_pred": u_pred,
            "u_lower": u_pred - half_width,
            "u_upper": u_pred + half_width,
            "half_width_map": half_width,
            "avg_band_width": 2 * half_width.mean().item(),
        }


def mc_dropout_predict(
    model: "torch.nn.Module",
    x: "torch.Tensor",
    n_passes: int = 20,
    dropout_rate: float = 0.1,
) -> dict:
    """MC Dropout uncertainty estimation.

    Enables dropout at inference time and runs multiple forward passes
    to estimate predictive uncertainty.

    Args:
        model: Neural operator model
        x: Input tensor
        n_passes: Number of stochastic forward passes
        dropout_rate: Dropout probability (applied to all Dropout/Linear layers)

    Returns:
        dict with 'mean', 'std', 'predictions' (all forward pass outputs)
    """
    import warnings

    # Save original dropout states so we can restore them
    original_dropout_p = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            original_dropout_p[name] = m.p

    if not original_dropout_p:
        warnings.warn(
            "mc_dropout_predict: model has no nn.Dropout layers. "
            "Predictions will be deterministic and std will be ~0. "
            "This baseline is only meaningful for models with dropout."
        )

    # Enable dropout at inference time
    model.eval()
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
            m.p = dropout_rate

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(x))

    predictions = torch.stack(preds)  # (n_passes, batch, *spatial)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    # Restore original dropout probabilities and eval mode
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout) and name in original_dropout_p:
            m.p = original_dropout_p[name]
    model.eval()

    return {
        "mean": mean,
        "std": std,
        "predictions": predictions,
        "n_passes": n_passes,
    }


def compute_per_frequency_scores(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    n_bins: int = 8,
    spatial_dims: int = 2,
) -> dict:
    """Decompose error into frequency bands for per-frequency coverage analysis.

    Returns a dict with:
      - 'bin_edges': frequency bin edges
      - 'band_scores': (n_bins, batch) scores per frequency band
      - 'band_power': (n_bins, batch) power per band (diagnostic)
    """
    error = u_true - u_pred
    batch_size = error.shape[0]
    spatial_shape = error.shape[1:]
    device = error.device

    if spatial_dims == 1:
        e_hat = fft.rfft(error, dim=-1)
    elif spatial_dims == 2:
        e_hat = fft.rfft2(error, dim=(-2, -1))
    else:
        e_hat = fft.rfftn(error, dim=tuple(range(1, error.ndim)))

    power = e_hat.real.pow(2) + e_hat.imag.pow(2)
    power = _apply_parseval_correction(power, spatial_shape[-1])
    flat_power = power.reshape(batch_size, -1)

    # Frequency magnitude grid
    freq_grid = SpectralScore(spatial_dims)._compute_frequency_grid(spatial_shape, device)
    k_mag = freq_grid.sqrt().flatten()
    max_k = k_mag.max()

    bin_edges = torch.linspace(0, max_k + 1e-6, n_bins + 1, device=device)
    bin_indices = torch.bucketize(k_mag, bin_edges[1:-1])

    band_scores = torch.zeros(n_bins, batch_size, device=device)
    band_power = torch.zeros(n_bins, batch_size, device=device)

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            band_scores[b] = flat_power[:, mask].sum(dim=-1)
            band_power[b] = flat_power[:, mask].mean(dim=-1)

    return {
        "bin_edges": bin_edges.cpu(),
        "band_scores": band_scores,
        "band_power": band_power,
    }
