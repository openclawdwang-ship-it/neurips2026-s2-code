"""
Calibration script: Run conformal prediction with ALL score functions + baselines.

Scores:
  - L2 (standard baseline)
  - Spectral uniform, Sobolev H^1, H^2, inverse power, learned
  - Simplified CQR (heteroscedastic bands, no retraining)

Baselines:
  - MC Dropout (Bayesian approx, no coverage guarantee)
  - Deep Ensemble (if checkpoints available)

Usage:
    python scripts/calibrate.py --model fno --pde darcy --resolution 64
    python scripts/calibrate.py --model fno --pde darcy --scores l2 spectral_sobolev_1
    python scripts/calibrate.py --model fno --pde darcy --scores cqr_simplified
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scores import (
    L2Score, SpectralScore, LearnedSpectralScore, SimplifiedCQRScore,
    mc_dropout_predict,
)
from src.conformal import SplitConformalPredictor, three_way_split
from src.evaluation import evaluate_coverage, evaluate_per_frequency_coverage
from src.models import load_trained_model
from src.data import get_data_splits, get_neuralop_darcy_splits, PDE_CONFIGS
from src.utils import prepare_input


def build_score(name: str, spatial_dims: int):
    """Factory for nonconformity score functions."""
    registry = {
        "l2": lambda: L2Score(),
        "spectral_uniform": lambda: SpectralScore(
            spatial_dims=spatial_dims, weight_type="uniform"
        ),
        "spectral_sobolev_1": lambda: SpectralScore(
            spatial_dims=spatial_dims, weight_type="sobolev", sobolev_s=1.0
        ),
        "spectral_sobolev_2": lambda: SpectralScore(
            spatial_dims=spatial_dims, weight_type="sobolev", sobolev_s=2.0
        ),
        "spectral_inverse": lambda: SpectralScore(
            spatial_dims=spatial_dims, weight_type="inverse_power"
        ),
        "spectral_learned": lambda: LearnedSpectralScore(spatial_dims=spatial_dims),
        "cqr_simplified": lambda: SimplifiedCQRScore(spatial_dims=spatial_dims),
    }
    if name not in registry:
        raise ValueError(f"Unknown score '{name}'. Available: {list(registry.keys())}")
    return registry[name]()


ALL_SCORES = [
    "l2", "spectral_uniform", "spectral_sobolev_1", "spectral_sobolev_2",
    "spectral_inverse", "spectral_learned", "cqr_simplified",
]


def run_calibration(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PDE_CONFIGS[args.pde]
    spatial_dims = config["spatial_dims"]

    # Load model
    ckpt_path = (
        Path(args.checkpoint_dir)
        / f"{args.model}_{args.pde}_{args.resolution}"
        / "best.pt"
    )
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run training first: bash scripts/run_training.sh")
        sys.exit(1)

    model = load_trained_model(
        str(ckpt_path),
        model_name=args.model,
        spatial_dims=spatial_dims,
        resolution=args.resolution,
    ).to(device)

    # Load data
    if args.data_source == "neuralop" and args.pde == "darcy":
        _, cal_loader, test_loader = get_neuralop_darcy_splits(
            n_train=args.n_train,
            n_cal=args.n_cal,
            n_test=args.n_test,
            resolution=args.resolution,
        )
    else:
        mode = "one_step" if config["time_dependent"] else "static"
        _, cal_loader, test_loader = get_data_splits(
            data_dir=args.data_dir,
            pde_name=args.pde,
            mode=mode,
            resolution=args.resolution,
            n_train=args.n_train,
            n_cal=args.n_cal,
            n_test=args.n_test,
        )

    # Get calibration and test data
    cal_x, cal_y = next(iter(cal_loader))
    test_x, test_y = next(iter(test_loader))

    # Retrieve normalization stats from checkpoint (None for legacy models)
    x_mean = getattr(model, "x_mean", None)
    x_std = getattr(model, "x_std", None)
    y_mean = getattr(model, "y_mean", None)
    y_std = getattr(model, "y_std", None)
    has_norm = x_mean is not None
    use_grid = getattr(model, "in_channels", None) is not None and getattr(model, "in_channels", 1) > 1

    with torch.no_grad():
        cal_x = cal_x.to(device)
        test_x = test_x.to(device)

        # Normalize inputs (must match training)
        cal_x_in = (cal_x - x_mean) / x_std if has_norm else cal_x
        test_x_in = (test_x - x_mean) / x_std if has_norm else test_x

        # Channel dim + grid coordinates
        cal_x_in = prepare_input(cal_x_in, spatial_dims, use_grid=use_grid)
        test_x_in = prepare_input(test_x_in, spatial_dims, use_grid=use_grid)

        cal_pred_norm = model(cal_x_in).squeeze(1)
        test_pred_norm = model(test_x_in).squeeze(1)

        # Denormalize predictions back to original scale
        if has_norm:
            cal_pred = cal_pred_norm * y_std + y_mean
            test_pred = test_pred_norm * y_std + y_mean
        else:
            cal_pred = cal_pred_norm
            test_pred = test_pred_norm

    # Move everything to CPU for conformal/numpy operations
    cal_pred = cal_pred.detach().cpu()
    test_pred = test_pred.detach().cpu()
    cal_y = cal_y.cpu()
    test_y = test_y.cpu()

    results = {}
    target_coverage = 1.0 - args.alpha

    print(f"\nModel: {args.model} | PDE: {args.pde} | Res: {args.resolution}")
    print(f"Cal: {len(cal_y)} | Test: {len(test_y)} | Target: {target_coverage:.0%}")
    print("=" * 65)

    for score_name in args.scores:
        t0 = time.time()
        print(f"\n--- {score_name} ---")

        try:
            score_fn = build_score(score_name, spatial_dims)

            if isinstance(score_fn, LearnedSpectralScore):
                # Three-way split: weight-learning / calibration / (extra test)
                weight_data, cal_data, _ = three_way_split(
                    cal_pred, cal_y, frac_weight=0.3, frac_cal=0.7
                )
                errors = weight_data["true"] - weight_data["pred"]
                learned_w = score_fn.learn_weights(errors)
                print(f"  Learned weights (per bin): {learned_w.cpu().numpy().round(3)}")
                cp = SplitConformalPredictor(score_fn, alpha=args.alpha)
                cp.calibrate(cal_data["pred"], cal_data["true"])

            elif isinstance(score_fn, SimplifiedCQRScore):
                # CQR-lite: fit local error scale on SEPARATE split, then calibrate.
                # Using the same data for fit_scale AND calibrate would make the score
                # function data-dependent, potentially violating coverage guarantee.
                # Three-way split (like LearnedSpectralScore) ensures validity.
                weight_data, cal_data, _ = three_way_split(
                    cal_pred, cal_y, frac_weight=0.3, frac_cal=0.7
                )
                cqr_errors = weight_data["true"] - weight_data["pred"]
                score_fn.fit_scale(cqr_errors)
                cp = SplitConformalPredictor(score_fn, alpha=args.alpha)
                cp.calibrate(cal_data["pred"], cal_data["true"])

            else:
                cp = SplitConformalPredictor(score_fn, alpha=args.alpha)
                cp.calibrate(cal_pred, cal_y)

            # Evaluate on test set
            cov = cp.evaluate_coverage(test_pred, test_y)
            freq_cov = evaluate_per_frequency_coverage(
                test_pred, test_y, cp.q_hat,
                n_bins=8, spatial_dims=spatial_dims,
            )

            elapsed = time.time() - t0

            # CQR-lite also reports spatially-varying band width
            extra = {}
            if isinstance(score_fn, SimplifiedCQRScore):
                bands = score_fn.predict_bands(test_pred, cp.q_hat)
                extra["avg_band_width_cqr"] = bands["avg_band_width"]
                extra["max_half_width"] = bands["half_width_map"].max().item()
                extra["min_half_width"] = bands["half_width_map"].min().item()

            results[score_name] = {
                "coverage": cov["empirical_coverage"],
                "band_width": cov["avg_band_width"],
                "calibration_error": cov["calibration_error"],
                "q_hat": cov["q_hat"],
                "per_freq_coverage": freq_cov["per_band_coverage"].tolist(),
                "per_freq_avg_power": freq_cov["per_band_avg_power"].tolist(),
                "elapsed_seconds": round(elapsed, 2),
                **extra,
            }

            # Status per PRINCIPLES.md
            valid = cov["empirical_coverage"] >= (target_coverage - 0.02)
            status = "VALID" if valid else "INVALID (coverage too low)"

            print(f"  Coverage:  {cov['empirical_coverage']:.3f}  (target: {target_coverage:.2f})  [{status}]")
            print(f"  Band width: {cov['avg_band_width']:.6f}")
            print(f"  Cal error:  {cov['calibration_error']:.4f}")
            print(f"  q_hat:      {cov['q_hat']:.6f}")
            print(f"  Time:       {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results[score_name] = {
                "coverage": 0.0,
                "band_width": 0.0,
                "calibration_error": 1.0,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            }
            print(f"  CRASH: {e}")

    # ---- MC Dropout baseline (no coverage guarantee) ----
    if args.mc_dropout:
        print(f"\n--- mc_dropout (n={args.mc_n_passes}) ---")
        t0 = time.time()
        try:
            # MC Dropout needs normalized input with channel dim + grid
            mc_input = (test_x - x_mean) / x_std if has_norm else test_x
            mc_input = prepare_input(mc_input, spatial_dims, use_grid=use_grid)
            mc = mc_dropout_predict(
                model, mc_input, n_passes=args.mc_n_passes, dropout_rate=0.1,
            )
            mc_mean = mc["mean"].squeeze(1)
            mc_std = mc["std"].squeeze(1)
            # Denormalize MC predictions
            if has_norm:
                mc_mean = mc_mean * y_std + y_mean
                mc_std = mc_std * abs(y_std)  # std scales linearly

            # Empirical coverage at ±2*std (should be ~95% for Gaussian)
            z = 1.645  # 90% for fair comparison with alpha=0.1
            mc_lower = mc_mean - z * mc_std
            mc_upper = mc_mean + z * mc_std
            mc_covered = ((test_y >= mc_lower) & (test_y <= mc_upper)).float()
            mc_coverage = mc_covered.mean().item()
            mc_band_width = (2 * z * mc_std).mean().item()
            elapsed = time.time() - t0

            results["mc_dropout"] = {
                "coverage": mc_coverage,
                "band_width": mc_band_width,
                "calibration_error": abs(mc_coverage - target_coverage),
                "note": "No coverage guarantee (Bayesian approx)",
                "n_passes": args.mc_n_passes,
                "z_score": z,
                "elapsed_seconds": round(elapsed, 2),
            }
            print(f"  Coverage:   {mc_coverage:.3f}  (NO guarantee)")
            print(f"  Band width: {mc_band_width:.6f}")
            print(f"  Time:       {elapsed:.1f}s")
        except Exception as e:
            results["mc_dropout"] = {"error": str(e)}
            print(f"  CRASH: {e}")

    # ---- Save results ----
    out_dir = Path(args.output_dir) / f"{args.model}_{args.pde}_{args.resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "calibration_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Append to results.tsv ----
    tsv_path = Path(args.output_dir) / "results.tsv"
    commit = "no-git"
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    with open(tsv_path, "a") as f:
        for score_name, metrics in results.items():
            cov = metrics.get("coverage", 0.0)
            bw = metrics.get("band_width", 0.0)
            ce = metrics.get("calibration_error", 1.0)
            valid = cov >= (target_coverage - 0.02)
            status = "keep" if valid and "error" not in metrics else "invalid"
            if "error" in metrics:
                status = "crash"
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{ts}\t{commit}\t{args.pde}\t{args.model}\t{score_name}\t"
                f"{args.alpha}\t{cov:.4f}\t{bw:.6f}\t{ce:.4f}\t{status}\t"
                f"{args.model}/{args.pde}/{args.resolution}\n"
            )

    # ---- Summary table ----
    print("\n" + "=" * 65)
    print(f"{'Score':<22} {'Coverage':>10} {'Band Width':>12} {'Cal Err':>10} {'Status':>8}")
    print("-" * 65)
    for score_name, metrics in results.items():
        cov = metrics.get("coverage", 0.0)
        bw = metrics.get("band_width", 0.0)
        ce = metrics.get("calibration_error", 1.0)
        valid = cov >= (target_coverage - 0.02)
        status = "OK" if valid and "error" not in metrics else "FAIL"
        if "error" in metrics:
            status = "CRASH"
        print(f"{score_name:<22} {cov:>10.4f} {bw:>12.6f} {ce:>10.4f} {status:>8}")
    print("=" * 65)

    print(f"\nResults: {out_file}")
    print(f"TSV log: {tsv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conformal calibration with spectral scores + baselines"
    )
    parser.add_argument("--model", type=str, default="fno",
                        choices=["fno", "tfno", "deeponet", "uno"])
    parser.add_argument("--pde", type=str, default="darcy",
                        choices=list(PDE_CONFIGS.keys()))
    parser.add_argument("--data_source", type=str, default="auto",
                        choices=["auto", "neuralop", "hdf5"])
    parser.add_argument("--data_dir", type=str, default="./data/pdebench")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scores", nargs="+", default=ALL_SCORES,
                        help=f"Score functions to run. Choices: {ALL_SCORES}")
    parser.add_argument("--mc_dropout", action="store_true", default=True,
                        help="Run MC Dropout baseline")
    parser.add_argument("--no_mc_dropout", dest="mc_dropout", action="store_false")
    parser.add_argument("--mc_n_passes", type=int, default=20)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_cal", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    args = parser.parse_args()
    run_calibration(args)
