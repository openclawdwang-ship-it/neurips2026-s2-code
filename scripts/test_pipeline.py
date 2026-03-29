#!/usr/bin/env python3
"""
End-to-end pipeline test for S2: Spectral Conformal Prediction.

Runs the full train → calibrate → evaluate flow on CPU with tiny data.
No GPU required. Verifies all components work together.

Usage:
    python scripts/test_pipeline.py          # uses neuralop data (auto-downloads ~10MB)
    python scripts/test_pipeline.py --quick  # uses tiny random tensors for ARCH TEST ONLY (not experiments)

NOTE: Random data is for ARCHITECTURE TESTING ONLY - not for experiments or results.
"""

import argparse
import sys
import time
import tempfile
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model, load_trained_model
from src.scores import (
    L2Score, SpectralScore, LearnedSpectralScore, SimplifiedCQRScore,
    mc_dropout_predict,
)
from src.conformal import SplitConformalPredictor, three_way_split
from src.evaluation import evaluate_coverage, evaluate_per_frequency_coverage
from src.utils import ensure_channel_dim, append_grid, prepare_input


# ---- Colors ----
G = "\033[0;32m"
R = "\033[0;31m"
Y = "\033[1;33m"
N = "\033[0m"


def ok(msg):
    print(f"  {G}[OK]{N} {msg}")

def fail(msg):
    print(f"  {R}[FAIL]{N} {msg}")
    return False

def warn(msg):
    print(f"  {Y}[WARN]{N} {msg}")


def make_random_data(n_samples=100, resolution=16):
    """Create random tensors mimicking Darcy data for quick test."""
    x = torch.randn(n_samples, resolution, resolution)
    y = torch.randn(n_samples, resolution, resolution)
    return x, y


def test_model_build():
    """Test model construction and forward pass."""
    print("\n=== Test 1: Model Build + Forward ===")
    spatial_dims = 2
    resolution = 16
    in_channels = 1 + spatial_dims  # field + grid coords
    model = get_model("fno", spatial_dims=spatial_dims,
                      in_channels=in_channels, out_channels=1,
                      resolution=resolution)
    n_params = sum(p.numel() for p in model.parameters())
    ok(f"FNO built: {n_params:,} params, in_channels={in_channels}")

    # Forward pass
    x = torch.randn(4, in_channels, resolution, resolution)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 1, resolution, resolution), f"Bad output shape: {y.shape}"
    ok(f"Forward pass: input={x.shape} -> output={y.shape}")
    return True


def test_checkpoint_roundtrip():
    """Test save/load checkpoint with in_channels consistency."""
    print("\n=== Test 2: Checkpoint Save/Load ===")
    spatial_dims = 2
    resolution = 16
    in_channels = 3  # 1 + 2 grid

    model = get_model("fno", spatial_dims=spatial_dims,
                      in_channels=in_channels, out_channels=1,
                      resolution=resolution)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "x_mean": 0.5, "x_std": 1.0,
            "y_mean": 0.0, "y_std": 0.3,
            "in_channels": in_channels,
            "epoch": 10,
            "loss": 0.01,
        }, ckpt_path)
        ok("Checkpoint saved")

        # Load with load_trained_model (the fixed version)
        loaded = load_trained_model(
            str(ckpt_path),
            model_name="fno",
            spatial_dims=spatial_dims,
            resolution=resolution,
            # NOTE: NOT passing in_channels — it should come from checkpoint
        )
        assert loaded.in_channels == in_channels, \
            f"in_channels mismatch: expected {in_channels}, got {loaded.in_channels}"
        ok(f"Loaded with in_channels={loaded.in_channels} from checkpoint")

        # Verify forward pass works
        x = torch.randn(2, in_channels, resolution, resolution)
        with torch.no_grad():
            y = loaded(x)
        assert y.shape == (2, 1, resolution, resolution)
        ok(f"Forward pass after load: {y.shape}")

    return True


def test_data_utils():
    """Test ensure_channel_dim and append_grid."""
    print("\n=== Test 3: Data Utilities ===")
    spatial_dims = 2
    res = 16

    # (B, H, W) -> (B, 1, H, W)
    x = torch.randn(4, res, res)
    x_ch = ensure_channel_dim(x, spatial_dims)
    assert x_ch.shape == (4, 1, res, res), f"Bad shape: {x_ch.shape}"
    ok(f"ensure_channel_dim: {x.shape} -> {x_ch.shape}")

    # (B, 1, H, W) -> (B, 3, H, W) with grid
    x_grid = append_grid(x_ch, spatial_dims)
    assert x_grid.shape == (4, 3, res, res), f"Bad shape: {x_grid.shape}"
    ok(f"append_grid: {x_ch.shape} -> {x_grid.shape}")

    # Already has channel dim: should not double-add
    x_already = torch.randn(4, 1, res, res)
    x_already_ch = ensure_channel_dim(x_already, spatial_dims)
    assert x_already_ch.shape == (4, 1, res, res), f"Double-added channel: {x_already_ch.shape}"
    ok(f"ensure_channel_dim idempotent: {x_already.shape} unchanged")

    # prepare_input end-to-end
    x_raw = torch.randn(4, res, res)
    x_prep = prepare_input(x_raw, spatial_dims, use_grid=True)
    assert x_prep.shape == (4, 3, res, res)
    ok(f"prepare_input: {x_raw.shape} -> {x_prep.shape}")

    return True


def test_scores():
    """Test all nonconformity score functions."""
    print("\n=== Test 4: Score Functions ===")
    B, H, W = 20, 16, 16
    pred = torch.randn(B, H, W)
    true = pred + 0.1 * torch.randn(B, H, W)  # small noise

    scores_to_test = {
        "l2": L2Score(),
        "spectral_uniform": SpectralScore(spatial_dims=2, weight_type="uniform"),
        "spectral_sobolev_1": SpectralScore(spatial_dims=2, weight_type="sobolev", sobolev_s=1.0),
        "spectral_sobolev_2": SpectralScore(spatial_dims=2, weight_type="sobolev", sobolev_s=2.0),
        "spectral_inverse": SpectralScore(spatial_dims=2, weight_type="inverse_power"),
    }

    for name, score_fn in scores_to_test.items():
        s = score_fn(pred, true)
        assert s.shape == (B,), f"{name}: bad shape {s.shape}"
        assert (s >= 0).all(), f"{name}: negative scores"
        ok(f"{name}: shape={s.shape}, range=[{s.min():.4f}, {s.max():.4f}]")

    # Learned spectral
    errors = true - pred
    learned = LearnedSpectralScore(spatial_dims=2, n_freq_bins=8)
    weights = learned.learn_weights(errors, n_steps=50, lr=0.01)
    s = learned(pred, true)
    assert s.shape == (B,)
    ok(f"spectral_learned: weights={weights.shape}, scores={s.shape}")

    # CQR simplified
    cqr = SimplifiedCQRScore(spatial_dims=2)
    cqr.fit_scale(errors)
    s = cqr(pred, true)
    assert s.shape == (B,)
    ok(f"cqr_simplified: scores={s.shape}")

    return True


def test_conformal_prediction():
    """Test split conformal prediction end-to-end."""
    print("\n=== Test 5: Conformal Prediction ===")
    B_cal, B_test, H, W = 50, 50, 16, 16

    # Simulate model predictions
    cal_pred = torch.randn(B_cal, H, W)
    cal_true = cal_pred + 0.1 * torch.randn(B_cal, H, W)
    test_pred = torch.randn(B_test, H, W)
    test_true = test_pred + 0.1 * torch.randn(B_test, H, W)

    alpha = 0.1
    target_cov = 1.0 - alpha

    for name, score_fn in [
        ("l2", L2Score()),
        ("spectral_sobolev_1", SpectralScore(spatial_dims=2, weight_type="sobolev", sobolev_s=1.0)),
    ]:
        cp = SplitConformalPredictor(score_fn, alpha=alpha)
        q_hat = cp.calibrate(cal_pred, cal_true)
        assert q_hat > 0, f"q_hat should be positive: {q_hat}"

        result = cp.evaluate_coverage(test_pred, test_true)
        cov = result["empirical_coverage"]
        bw = result["avg_band_width"]
        ok(f"{name}: coverage={cov:.3f} (target={target_cov}), band_width={bw:.6f}, q_hat={q_hat:.6f}")

    # Three-way split
    w_data, c_data, t_data = three_way_split(cal_pred, cal_true, frac_weight=0.3, frac_cal=0.7)
    assert len(w_data["pred"]) > 0
    assert len(c_data["pred"]) > 0
    ok(f"three_way_split: weight={len(w_data['pred'])}, cal={len(c_data['pred'])}")

    # Per-frequency coverage
    freq_result = evaluate_per_frequency_coverage(
        test_pred, test_true, q_hat, n_bins=4, spatial_dims=2
    )
    assert "per_band_coverage" in freq_result
    ok(f"per_freq_coverage: {len(freq_result['per_band_coverage'])} bands")

    return True


def test_full_pipeline(use_neuralop: bool = False):
    """Full train → calibrate → evaluate pipeline on tiny data."""
    print("\n=== Test 6: Full Pipeline (train → calibrate → evaluate) ===")

    spatial_dims = 2
    resolution = 16
    in_channels = 1 + spatial_dims
    epochs = 3
    n_train, n_cal, n_test = 40, 20, 20

    device = torch.device("cpu")

    # Data
    if use_neuralop:
        from src.data import get_neuralop_darcy_splits
        try:
            train_loader, cal_loader, test_loader = get_neuralop_darcy_splits(
                n_train=n_train, n_cal=n_cal, n_test=n_test,
                resolution=resolution,
            )
            ok("neuralop data loaded")
        except Exception as e:
            warn(f"neuralop data failed: {e}. Using random data.")
            use_neuralop = False

    if not use_neuralop:
        from torch.utils.data import TensorDataset, DataLoader
        x_all, y_all = make_random_data(n_train + n_cal + n_test, resolution)
        train_ds = TensorDataset(x_all[:n_train], y_all[:n_train])
        cal_ds = TensorDataset(x_all[n_train:n_train+n_cal], y_all[n_train:n_train+n_cal])
        test_ds = TensorDataset(x_all[n_train+n_cal:], y_all[n_train+n_cal:])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        cal_loader = DataLoader(cal_ds, batch_size=n_cal, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=n_test, shuffle=False)
        ok("Random data generated")

    # --- Train ---
    model = get_model("fno", spatial_dims=spatial_dims,
                      in_channels=in_channels, out_channels=1,
                      resolution=resolution).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Compute normalization stats
    all_x_list, all_y_list = [], []
    for xb, yb in train_loader:
        all_x_list.append(xb)
        all_y_list.append(yb)
    all_x_cat = torch.cat(all_x_list)
    all_y_cat = torch.cat(all_y_list)
    x_mean, x_std = all_x_cat.mean().item(), max(all_x_cat.std().item(), 1e-8)
    y_mean, y_std = all_y_cat.mean().item(), max(all_y_cat.std().item(), 1e-8)
    ok(f"Norm stats: x_mean={x_mean:.4f}, y_mean={y_mean:.4f}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batch = 0
        for xb, yb in train_loader:
            xb = (xb - x_mean) / x_std
            yb = (yb - y_mean) / y_std
            xb = prepare_input(xb, spatial_dims, use_grid=True)
            yb = ensure_channel_dim(yb, spatial_dims)
            pred = model(xb)
            loss = (pred - yb).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        avg_loss = total_loss / max(n_batch, 1)
    ok(f"Training: {epochs} epochs, final_loss={avg_loss:.6f}")

    # --- Save/Load checkpoint ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "fno_darcy_16" / "best.pt"
        ckpt_path.parent.mkdir(parents=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
            "in_channels": in_channels,
            "epoch": epochs - 1, "loss": avg_loss,
        }, ckpt_path)
        ok("Checkpoint saved")

        loaded_model = load_trained_model(
            str(ckpt_path), model_name="fno",
            spatial_dims=spatial_dims, resolution=resolution,
        )
        assert loaded_model.in_channels == in_channels
        ok(f"Checkpoint loaded: in_channels={loaded_model.in_channels}")

    # --- Calibrate ---
    loaded_model.eval()
    cal_x, cal_y = next(iter(cal_loader))
    test_x, test_y = next(iter(test_loader))

    has_norm = loaded_model.x_mean is not None
    use_grid = loaded_model.in_channels is not None and loaded_model.in_channels > 1

    with torch.no_grad():
        cal_x_in = (cal_x - x_mean) / x_std if has_norm else cal_x
        test_x_in = (test_x - x_mean) / x_std if has_norm else test_x
        cal_x_in = prepare_input(cal_x_in, spatial_dims, use_grid=use_grid)
        test_x_in = prepare_input(test_x_in, spatial_dims, use_grid=use_grid)
        cal_pred = loaded_model(cal_x_in).squeeze(1)
        test_pred = loaded_model(test_x_in).squeeze(1)
        if has_norm:
            cal_pred = cal_pred * y_std + y_mean
            test_pred = test_pred * y_std + y_mean

    # Squeeze y if needed (neuralop data may have channel dim)
    if cal_y.ndim == 4 and cal_y.shape[1] == 1:
        cal_y = cal_y.squeeze(1)
        test_y = test_y.squeeze(1)

    ALL_SCORES = ["l2", "spectral_uniform", "spectral_sobolev_1",
                  "spectral_sobolev_2", "spectral_inverse",
                  "spectral_learned", "cqr_simplified"]
    alpha = 0.1

    from scripts.calibrate import build_score

    results = {}
    for score_name in ALL_SCORES:
        score_fn = build_score(score_name, spatial_dims)

        if isinstance(score_fn, LearnedSpectralScore):
            w_data, c_data, _ = three_way_split(cal_pred, cal_y, frac_weight=0.3, frac_cal=0.7)
            errors = w_data["true"] - w_data["pred"]
            score_fn.learn_weights(errors, n_steps=50)
            cp = SplitConformalPredictor(score_fn, alpha=alpha)
            cp.calibrate(c_data["pred"], c_data["true"])
        elif isinstance(score_fn, SimplifiedCQRScore):
            w_data, c_data, _ = three_way_split(cal_pred, cal_y, frac_weight=0.3, frac_cal=0.7)
            errors = w_data["true"] - w_data["pred"]
            score_fn.fit_scale(errors)
            cp = SplitConformalPredictor(score_fn, alpha=alpha)
            cp.calibrate(c_data["pred"], c_data["true"])
        else:
            cp = SplitConformalPredictor(score_fn, alpha=alpha)
            cp.calibrate(cal_pred, cal_y)

        cov = cp.evaluate_coverage(test_pred, test_y)
        results[score_name] = cov
        ok(f"{score_name:<22} cov={cov['empirical_coverage']:.3f}  bw={cov['avg_band_width']:.6f}")

    # MC Dropout
    mc_input = prepare_input((test_x - x_mean) / x_std, spatial_dims, use_grid=use_grid)
    mc = mc_dropout_predict(loaded_model, mc_input, n_passes=5, dropout_rate=0.1)
    ok(f"MC Dropout: mean={mc['mean'].shape}, std={mc['std'].shape}")

    print(f"\n  {G}ALL {len(ALL_SCORES)} scores + MC Dropout ran successfully{N}")
    return True


def main():
    parser = argparse.ArgumentParser(description="S2 pipeline test")
    parser.add_argument("--quick", action="store_true",
                        help="Use random data (no download)")
    parser.add_argument("--neuralop", action="store_true",
                        help="Use neuralop data (auto-downloads ~10MB)")
    args = parser.parse_args()

    use_neuralop = args.neuralop and not args.quick

    print("=" * 60)
    print("  S2: Spectral Conformal Prediction — Pipeline Test")
    print(f"  Data: {'neuralop' if use_neuralop else 'random (quick)'}")
    print(f"  Device: CPU")
    print("=" * 60)

    t0 = time.time()
    tests = [
        ("Model Build", test_model_build),
        ("Checkpoint Roundtrip", test_checkpoint_roundtrip),
        ("Data Utilities", test_data_utils),
        ("Score Functions", test_scores),
        ("Conformal Prediction", test_conformal_prediction),
        ("Full Pipeline", lambda: test_full_pipeline(use_neuralop=use_neuralop)),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            fail(f"{name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  {G}ALL {passed} TESTS PASSED{N} ({elapsed:.1f}s)")
    else:
        print(f"  {R}{failed} FAILED{N}, {passed} passed ({elapsed:.1f}s)")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
