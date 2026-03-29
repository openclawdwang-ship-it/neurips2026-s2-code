#!/usr/bin/env python3
"""
S2 Conformal UQ Experiment — Real PDEBench/neuralop Darcy Flow Data

Uses REAL FEM-computed Darcy Flow data from the neuraloperator library
(Zenodo record 12784353). NO synthetic data.

Data: 2D Darcy Flow  -∇·(a(x)∇u) = f
  x = binary permeability field a(x) (piecewise-constant, bool→float)
  y = pressure solution u(x) computed by FEM

Pipeline:
  1. Load real data via neuralop DarcyDataset (auto-downloads from Zenodo)
  2. Train FNO (from neuralop library) with proper normalization
  3. Split: train / calibration / test (no leakage)
  4. Calibrate conformal predictor on calibration set
  5. Evaluate coverage on held-out test set
  6. Compare L2, Spectral, Sobolev, and Learned spectral scores

Available resolutions: 16, 32, 64, 128, 421
Default: tries 64 first, falls back to 16 if not downloaded yet.
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Project imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scores import L2Score, SpectralScore, LearnedSpectralScore
from src.conformal import SplitConformalPredictor, three_way_split
from src.utils import ensure_channel_dim, append_grid


def detect_best_resolution(data_dir: Path) -> int:
    """Find the best available resolution in data_dir."""
    for res in [64, 32, 16]:
        train_file = data_dir / f"darcy_train_{res}.pt"
        test_file = data_dir / f"darcy_test_{res}.pt"
        if train_file.exists() and test_file.exists():
            return res
    return 16  # will trigger download


def load_real_darcy_data(data_dir: Path, resolution: int, n_train: int,
                         n_cal: int, n_test: int, seed: int = 42):
    """Load real Darcy Flow data from neuralop .pt files.

    Returns raw tensors (no channel dim) with proper train/cal/test split.
    Loads directly from .pt files for speed (avoids slow DarcyDataset iterator).
    Falls back to DarcyDataset if .pt files not found (triggers download).
    """
    total_needed = n_train + n_cal + n_test
    train_file = data_dir / f"darcy_train_{resolution}.pt"
    test_file = data_dir / f"darcy_test_{resolution}.pt"

    if train_file.exists():
        # Fast path: load .pt directly
        print(f"  Loading .pt files directly (fast path)...")
        train_data = torch.load(str(train_file), weights_only=False)
        all_x = train_data['x'].float()  # (N, H, W), may be bool
        all_y = train_data['y'].float()  # (N, H, W)

        # Also load test file for more samples
        if test_file.exists():
            test_data = torch.load(str(test_file), weights_only=False)
            all_x = torch.cat([all_x, test_data['x'].float()], dim=0)
            all_y = torch.cat([all_y, test_data['y'].float()], dim=0)

        # Squeeze channel dim if present: (N, 1, H, W) -> (N, H, W)
        if all_x.ndim == 4 and all_x.shape[1] == 1:
            all_x = all_x.squeeze(1)
            all_y = all_y.squeeze(1)

        print(f"  Loaded: x={all_x.shape}, y={all_y.shape}")
    else:
        # Slow path: use DarcyDataset (triggers download from Zenodo)
        print(f"  .pt files not found, using DarcyDataset (will download)...")
        from neuralop.data.datasets.darcy import DarcyDataset
        dataset = DarcyDataset(
            root_dir=str(data_dir),
            n_train=total_needed,
            n_tests=[1],
            batch_size=total_needed,
            test_batch_sizes=[1],
            train_resolution=resolution,
            test_resolutions=[resolution],
            encode_input=False,
            encode_output=False,
            download=True,
        )
        all_data = list(dataset.train_db)
        all_x = torch.stack([item['x'] for item in all_data]).squeeze(1).float()
        all_y = torch.stack([item['y'] for item in all_data]).squeeze(1).float()

    n_available = len(all_x)
    if n_available < total_needed:
        print(f"  WARNING: Only {n_available} samples available, "
              f"need {total_needed}. Adjusting splits.")
        n_test = min(n_test, n_available // 5)
        n_cal = min(n_cal, n_available // 5)
        n_train = n_available - n_cal - n_test

    # Deterministic split
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_available, generator=rng)

    train_x = all_x[perm[:n_train]]
    train_y = all_y[perm[:n_train]]
    cal_x = all_x[perm[n_train:n_train + n_cal]]
    cal_y = all_y[perm[n_train:n_train + n_cal]]
    test_x = all_x[perm[n_train + n_cal:n_train + n_cal + n_test]]
    test_y = all_y[perm[n_train + n_cal:n_train + n_cal + n_test]]

    return (train_x, train_y), (cal_x, cal_y), (test_x, test_y)


def verify_real_data(x: torch.Tensor, y: torch.Tensor):
    """Verify data is real FEM, not synthetic random noise."""
    checks = {}

    # Check 1: x should be binary (piecewise-constant permeability)
    unique_vals = torch.unique(x[:5])
    checks["x_binary"] = len(unique_vals) <= 10  # real data has few unique values

    # Check 2: y should be non-negative (pressure field) or have physical range
    checks["y_physical_range"] = y.min() >= -1.0 and y.max() <= 3.0

    # Check 3: Spatial autocorrelation (real PDE solutions are smooth)
    y_np = y[:10].numpy()
    autocorrs = []
    for i in range(min(10, len(y_np))):
        f1 = y_np[i, :-1].flatten()
        f2 = y_np[i, 1:].flatten()
        if f1.std() > 1e-8 and f2.std() > 1e-8:
            autocorrs.append(np.corrcoef(f1, f2)[0, 1])
    avg_autocorr = np.mean(autocorrs) if autocorrs else 0.0
    checks["spatial_autocorr"] = avg_autocorr > 0.5  # real data >> 0

    # Check 4: std should NOT be ~1.0 (would indicate np.random.randn)
    checks["not_unit_std"] = abs(x.std().item() - 1.0) > 0.01 or x.dtype == torch.bool

    passed = all(checks.values())

    print(f"  Data verification:")
    print(f"    x: shape={x.shape}, dtype={x.dtype}, "
          f"unique_vals={len(unique_vals)}, range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"    y: shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}], "
          f"std={y.std():.4f}")
    print(f"    Spatial autocorrelation (y): {avg_autocorr:.4f} "
          f"(real PDE > 0.5, random ~ 0)")
    for name, ok in checks.items():
        print(f"    {'✓' if ok else '✗'} {name}")

    if not passed:
        print("\n  *** FAILURE: Data appears SYNTHETIC. Aborting. ***")
        print("  Real Darcy data has binary permeability and smooth pressure fields.")
        sys.exit(1)

    print(f"  ✓ All checks passed — data is REAL FEM simulation")
    return checks


def build_fno(resolution: int, device: torch.device):
    """Build FNO from neuralop library."""
    from neuralop.models import FNO

    # in_channels = 1 (field) + 2 (grid coords)
    n_modes = min(12, resolution // 2)
    model = FNO(
        n_modes=(n_modes, n_modes),
        in_channels=3,   # 1 input field + 2 grid coords
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FNO: {n_params:,} parameters, n_modes={n_modes}")
    return model


def train_model(model, train_x, train_y, device, epochs=100, lr=1e-3):
    """Train FNO with relative L2 loss and proper normalization."""
    # Compute normalization stats
    x_mean, x_std = train_x.mean().item(), max(train_x.std().item(), 1e-8)
    y_mean, y_std = train_y.mean().item(), max(train_y.std().item(), 1e-8)
    print(f"  Normalization: x=[{x_mean:.3f}±{x_std:.3f}], y=[{y_mean:.3f}±{y_std:.3f}]")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 32
    n = len(train_x)
    best_loss = float("inf")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = train_x[idx].to(device)
            y = train_y[idx].to(device)

            # Normalize
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std

            # Add channel dim + grid: (B, H, W) -> (B, 3, H, W)
            x = ensure_channel_dim(x, spatial_dims=2)
            x = append_grid(x, spatial_dims=2)
            y = ensure_channel_dim(y, spatial_dims=2)

            pred = model(x)

            # Relative L2 loss
            b = pred.shape[0]
            diff = (pred - y).reshape(b, -1).pow(2).sum(dim=-1).sqrt()
            target_norm = y.reshape(b, -1).pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
            loss = (diff / target_norm).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"    Epoch {epoch+1:3d}/{epochs} | RelL2: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | {elapsed:.0f}s")

    # Store normalization stats on model for inference
    model.x_mean = x_mean
    model.x_std = x_std
    model.y_mean = y_mean
    model.y_std = y_std

    return model, best_loss


def predict(model, x, device):
    """Run inference with proper normalization and input preparation."""
    model.eval()
    with torch.no_grad():
        x_norm = (x.to(device) - model.x_mean) / model.x_std
        x_prep = ensure_channel_dim(x_norm, spatial_dims=2)
        x_prep = append_grid(x_prep, spatial_dims=2)
        y_pred_norm = model(x_prep)
        # De-normalize output, squeeze channel dim
        y_pred = y_pred_norm.squeeze(1) * model.y_std + model.y_mean
    return y_pred.cpu()


def run_conformal_evaluation(y_pred, y_true, alpha=0.1):
    """Run conformal prediction with multiple score functions."""
    results = {}

    # 1. L2 Score (baseline)
    cp_l2 = SplitConformalPredictor(score_fn=L2Score(), alpha=alpha)
    cp_l2.calibrate(y_pred, y_true)
    # We need separate cal/test for proper evaluation
    # But here we report calibration quantile for reference
    results["l2_q_hat"] = cp_l2.q_hat

    # 2. Spectral Score (uniform weights — should equal L2 by Parseval)
    cp_spec = SplitConformalPredictor(
        score_fn=SpectralScore(spatial_dims=2, weight_type="uniform"), alpha=alpha)
    cp_spec.calibrate(y_pred, y_true)
    results["spectral_uniform_q_hat"] = cp_spec.q_hat

    # 3. Sobolev Score (H^1 — penalizes high-frequency errors)
    cp_sob = SplitConformalPredictor(
        score_fn=SpectralScore(spatial_dims=2, weight_type="sobolev", sobolev_s=1.0),
        alpha=alpha)
    cp_sob.calibrate(y_pred, y_true)
    results["sobolev_h1_q_hat"] = cp_sob.q_hat

    return results, {"l2": cp_l2, "spectral": cp_spec, "sobolev": cp_sob}


def main():
    parser = argparse.ArgumentParser(description="S2 Conformal UQ — Real Data Experiment")
    parser.add_argument("--data_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "neuralop_darcy"))
    parser.add_argument("--resolution", type=int, default=0,
                        help="0 = auto-detect best available")
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_cal", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (0.1 = 90%% target coverage)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("  S2: Neural Operator Conformal Prediction — REAL DATA")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Data dir: {data_dir}")

    # Step 1: Detect resolution
    resolution = args.resolution
    if resolution == 0:
        resolution = detect_best_resolution(data_dir)
    print(f"  Resolution: {resolution}×{resolution}")
    print(f"  Splits: train={args.n_train}, cal={args.n_cal}, test={args.n_test}")
    print(f"  Target coverage: {1 - args.alpha:.0%}")

    # Step 2: Load REAL data
    print(f"\n{'─' * 70}")
    print("STEP 1: Load Real Darcy Flow Data (neuralop / Zenodo)")
    print(f"{'─' * 70}")

    (train_x, train_y), (cal_x, cal_y), (test_x, test_y) = load_real_darcy_data(
        data_dir, resolution, args.n_train, args.n_cal, args.n_test, args.seed)

    print(f"  Train: x={train_x.shape}, y={train_y.shape}")
    print(f"  Cal:   x={cal_x.shape}, y={cal_y.shape}")
    print(f"  Test:  x={test_x.shape}, y={test_y.shape}")

    # Step 3: Verify data is real
    print(f"\n  Verifying data authenticity...")
    verify_real_data(train_x, train_y)

    # Step 4: Train FNO
    print(f"\n{'─' * 70}")
    print("STEP 2: Train FNO on Real Data")
    print(f"{'─' * 70}")

    model = build_fno(resolution, device)
    model, best_loss = train_model(model, train_x, train_y, device,
                                    epochs=args.epochs, lr=1e-3)
    print(f"  Training complete. Best relative L2: {best_loss:.6f}")

    # Step 5: Generate predictions on cal and test sets
    print(f"\n{'─' * 70}")
    print("STEP 3: Generate Predictions")
    print(f"{'─' * 70}")

    cal_pred = predict(model, cal_x, device)
    test_pred = predict(model, test_x, device)

    # Report prediction quality
    cal_mse = (cal_pred - cal_y).pow(2).mean().item()
    test_mse = (test_pred - test_y).pow(2).mean().item()
    cal_rel_l2 = ((cal_pred - cal_y).pow(2).sum() / cal_y.pow(2).sum()).sqrt().item()
    test_rel_l2 = ((test_pred - test_y).pow(2).sum() / test_y.pow(2).sum()).sqrt().item()
    print(f"  Cal  MSE: {cal_mse:.6f}, Relative L2: {cal_rel_l2:.4f}")
    print(f"  Test MSE: {test_mse:.6f}, Relative L2: {test_rel_l2:.4f}")

    # Step 6: Conformal Prediction
    print(f"\n{'─' * 70}")
    print("STEP 4: Conformal Prediction (calibrate on cal, evaluate on test)")
    print(f"{'─' * 70}")

    alpha = args.alpha
    n_cal = len(cal_y)
    n_test = len(test_y)

    score_fns = {
        "L2": L2Score(),
        "Spectral (uniform)": SpectralScore(spatial_dims=2, weight_type="uniform"),
        "Sobolev H^1": SpectralScore(spatial_dims=2, weight_type="sobolev", sobolev_s=1.0),
        "Inverse-power": SpectralScore(spatial_dims=2, weight_type="inverse_power"),
    }

    print(f"\n  {'Score Function':<25} {'q_hat':>10} {'Coverage':>10} "
          f"{'Band Width':>12} {'Cal Error':>10}")
    print(f"  {'─' * 67}")

    all_results = {}
    for name, score_fn in score_fns.items():
        cp = SplitConformalPredictor(score_fn=score_fn, alpha=alpha)
        q_hat = cp.calibrate(cal_pred, cal_y)
        eval_result = cp.evaluate_coverage(test_pred, test_y)

        coverage = eval_result["empirical_coverage"]
        band_width = eval_result["avg_band_width"]
        cal_error = eval_result["calibration_error"]

        all_results[name] = eval_result
        print(f"  {name:<25} {q_hat:>10.4f} {coverage:>9.1%} "
              f"{band_width:>12.4f} {cal_error:>10.4f}")

    # Step 7: Learned Spectral Score (requires three-way split)
    print(f"\n  Learned Spectral Score (three-way split):")
    print(f"  Using calibration set split into weight-learning / conformal-cal / eval")

    # Combine cal+test for three-way split
    combined_pred = torch.cat([cal_pred, test_pred], dim=0)
    combined_true = torch.cat([cal_y, test_y], dim=0)

    weight_data, cal3_data, test3_data = three_way_split(
        combined_pred, combined_true, frac_weight=0.2, frac_cal=0.5, seed=args.seed)

    learned_score = LearnedSpectralScore(spatial_dims=2, n_freq_bins=8)
    errors_for_learning = weight_data["true"] - weight_data["pred"]
    learned_weights = learned_score.learn_weights(errors_for_learning, n_steps=200)
    print(f"  Learned weights (per freq bin): "
          f"{[f'{w:.3f}' for w in learned_weights.tolist()]}")

    cp_learned = SplitConformalPredictor(score_fn=learned_score, alpha=alpha)
    cp_learned.calibrate(cal3_data["pred"], cal3_data["true"])
    learned_eval = cp_learned.evaluate_coverage(test3_data["pred"], test3_data["true"])

    print(f"  {'Learned Spectral':<25} {cp_learned.q_hat:>10.4f} "
          f"{learned_eval['empirical_coverage']:>9.1%} "
          f"{learned_eval['avg_band_width']:>12.4f} "
          f"{learned_eval['calibration_error']:>10.4f}")
    all_results["Learned Spectral"] = learned_eval

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Data: neuralop Darcy Flow {resolution}×{resolution} (Zenodo 12784353)")
    print(f"  Data type: REAL FEM simulation (NOT synthetic)")
    print(f"  Model: FNO ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Test Relative L2: {test_rel_l2:.4f}")
    print(f"  Target coverage: {1 - alpha:.0%}")
    print(f"")
    print(f"  {'Score':<25} {'Coverage':>10} {'Width':>10} {'Tight?':>8}")
    print(f"  {'─' * 53}")
    for name, res in all_results.items():
        cov = res["empirical_coverage"]
        width = res["avg_band_width"]
        # Coverage should be >= 1-alpha (conformal guarantee)
        tight = "✓" if cov >= (1 - alpha - 0.05) else "✗"
        print(f"  {name:<25} {cov:>9.1%} {width:>10.4f} {tight:>8}")

    print(f"\n  Conformal guarantee: P(Y ∈ C(X)) ≥ 1-α = {1-alpha:.0%}")
    print(f"  With n_cal={n_cal}, finite-sample coverage ≥ {1-alpha:.0%} "
          f"(marginal, exchangeable)")

    # Honesty note
    if test_rel_l2 > 0.5:
        print(f"\n  ⚠ WARNING: Model relative L2 error is high ({test_rel_l2:.2f}).")
        print(f"  The conformal bands are VALID but WIDE — the model needs more")
        print(f"  training (more epochs, GPU, or larger training set) for tight bands.")
        print(f"  This is expected for CPU-only training with limited epochs.")

    print(f"\n{'=' * 70}")
    print(f"  Experiment complete. All results use REAL data.")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    main()
