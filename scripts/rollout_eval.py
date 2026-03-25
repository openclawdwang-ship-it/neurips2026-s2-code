"""
Rollout evaluation: Spectral ACI vs baselines on auto-regressive PDE rollouts.

This is the key experiment for C2 (Spectral ACI).
Compares:
- Standard split conformal (static q_hat, L2 scores)
- Standard split conformal (static q_hat, spectral scores)
- Spectral ACI (adaptive q_hat, spectral scores) -- OURS
- CP-PRE style (static per-step recalibration, PDE residual scores)

Generates coverage-vs-timestep and band-width-vs-timestep plots.

Usage:
    python scripts/rollout_eval.py --model fno --pde navier_stokes --n_steps 20
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scores import L2Score, SpectralScore
from src.conformal import SplitConformalPredictor
from src.spectral_aci import SpectralACI
from src.evaluation import evaluate_rollout
from src.models import load_trained_model
from src.data import PDEBenchDataset, PDE_CONFIGS
from src.utils import prepare_input


def run_rollout_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PDE_CONFIGS[args.pde]

    if not config["time_dependent"]:
        print(f"PDE '{args.pde}' is not time-dependent. Skipping rollout eval.")
        return

    spatial_dims = config["spatial_dims"]

    # Load model
    ckpt_path = (
        Path(args.checkpoint_dir)
        / f"{args.model}_{args.pde}_{args.resolution}"
        / "best.pt"
    )
    model = load_trained_model(
        str(ckpt_path),
        model_name=args.model,
        spatial_dims=spatial_dims,
        resolution=args.resolution,
    ).to(device).eval()

    # Load trajectory data
    dataset = PDEBenchDataset(
        args.data_dir, args.pde, mode="trajectory",
        resolution=args.resolution,
        max_samples=args.n_cal + args.n_test,
    )

    # Split into cal and test trajectories
    n_total = len(dataset)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    cal_indices = indices[: args.n_cal]
    test_indices = indices[args.n_cal : args.n_cal + args.n_test]

    # Retrieve normalization stats from checkpoint
    x_mean = getattr(model, "x_mean", None)
    x_std = getattr(model, "x_std", None)
    y_mean = getattr(model, "y_mean", None)
    y_std = getattr(model, "y_std", None)
    has_norm = x_mean is not None
    use_grid = getattr(model, "in_channels", None) is not None and getattr(model, "in_channels", 1) > 1

    def _normalize_input(x):
        return (x - x_mean) / x_std if has_norm else x

    def _denormalize_output(y):
        return y * y_std + y_mean if has_norm else y

    def _prepare(x):
        return prepare_input(_normalize_input(x), spatial_dims, use_grid=use_grid)

    # Collect calibration data (first step only, for initial calibration)
    cal_u0s, cal_u1s = [], []
    for idx in cal_indices:
        u0, traj = dataset[idx.item()]
        cal_u0s.append(u0)
        cal_u1s.append(traj[0])  # First timestep target

    cal_u0 = torch.stack(cal_u0s).to(device)
    cal_u1 = torch.stack(cal_u1s).to(device)

    with torch.no_grad():
        cal_input = _prepare(cal_u0)
        cal_pred = _denormalize_output(model(cal_input).squeeze(1))
    cal_u1_flat = cal_u1 if cal_u1.ndim == spatial_dims + 1 else cal_u1.squeeze(1)

    # Collect test trajectories
    test_trajectories = []
    test_u0s = []
    for idx in test_indices:
        u0, traj = dataset[idx.item()]
        test_u0s.append(u0)
        test_trajectories.append(traj)

    test_u0 = torch.stack(test_u0s).to(device)
    n_steps = min(args.n_steps, test_trajectories[0].shape[0])
    test_true_seq = torch.stack(
        [traj[:n_steps] for traj in test_trajectories]
    ).to(device)  # (n_test, n_steps, *spatial)

    all_results = {}

    # --- Method 1: Static L2 Conformal ---
    print("\n=== Static L2 Conformal ===")
    l2_score = L2Score()
    l2_cp = SplitConformalPredictor(l2_score, alpha=args.alpha)
    l2_cp.calibrate(cal_pred, cal_u1_flat)

    l2_results = _run_static_rollout(model, test_u0, test_true_seq, l2_cp, n_steps, spatial_dims, device, _prepare, _denormalize_output)
    all_results["static_l2"] = evaluate_rollout(l2_results, alpha=args.alpha)

    # --- Method 2: Static Spectral Conformal ---
    print("\n=== Static Spectral (Sobolev H^1) Conformal ===")
    spec_score = SpectralScore(spatial_dims=spatial_dims, weight_type="sobolev", sobolev_s=1.0)
    spec_cp = SplitConformalPredictor(spec_score, alpha=args.alpha)
    spec_cp.calibrate(cal_pred, cal_u1_flat)

    spec_results = _run_static_rollout(model, test_u0, test_true_seq, spec_cp, n_steps, spatial_dims, device, _prepare, _denormalize_output)
    all_results["static_spectral"] = evaluate_rollout(spec_results, alpha=args.alpha)

    # --- Method 3: Spectral ACI (OURS) ---
    print("\n=== Spectral ACI (ours) ===")
    saci = SpectralACI(
        alpha=args.alpha,
        gamma=args.aci_gamma,
        spatial_dims=spatial_dims,
        weight_type="sobolev",
        sobolev_s=1.0,
        n_freq_bins=8,
        adapt_per_frequency=True,
    )
    saci.calibrate_initial(cal_pred, cal_u1_flat)

    # Run rollout
    u_current = test_u0  # in original scale

    saci_results = []
    for t in range(n_steps):
        with torch.no_grad():
            u_input = _prepare(u_current)
            u_pred_t = _denormalize_output(model(u_input).squeeze(1))
        u_true_t = test_true_seq[:, t]
        if u_true_t.ndim > spatial_dims + 1:
            u_true_t = u_true_t.squeeze(1)

        step_result = saci.step(u_pred_t, u_true_t)
        saci_results.append(step_result)

        # Auto-regressive: feed prediction as next input (original scale)
        u_current = u_pred_t

    all_results["spectral_aci"] = evaluate_rollout(saci_results, alpha=args.alpha)

    # Print summary
    print("\n" + "=" * 60)
    print("ROLLOUT EVALUATION SUMMARY")
    print("=" * 60)
    for method_name, metrics in all_results.items():
        print(f"\n{method_name}:")
        if "coverage" in metrics:
            print(f"  Avg coverage: {metrics['avg_coverage']:.3f} (target: {1-args.alpha:.2f})")
            print(f"  Min coverage: {metrics['min_coverage']:.3f}")
        print(f"  Final band width: {metrics['band_width'][-1]:.4f}")
        print(f"  Band width growth: {metrics['band_width'][-1] / max(metrics['band_width'][0], 1e-10):.2f}x")

    # Save
    out_dir = Path(args.output_dir) / f"rollout_{args.model}_{args.pde}_{args.resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv for kk, vv in v.items()}

    with open(out_dir / "rollout_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_dir}")


def _run_static_rollout(model, test_u0, test_true_seq, cp, n_steps, spatial_dims, device,
                        prepare_fn, denormalize_output):
    """Run rollout with a STATIC conformal predictor (no adaptation)."""
    results = []
    u_current = test_u0  # original scale

    for t in range(n_steps):
        with torch.no_grad():
            u_input = prepare_fn(u_current)
            u_pred_t = denormalize_output(model(u_input).squeeze(1))
        u_true_t = test_true_seq[:, t]
        if u_true_t.ndim > spatial_dims + 1:
            u_true_t = u_true_t.squeeze(1)

        scores = cp.score_fn(u_pred_t, u_true_t)
        covered = (scores <= cp.q_hat).float().mean().item()

        n_points = int(torch.tensor(u_pred_t.shape[1:]).prod().item())
        half_width = (cp.q_hat / n_points) ** 0.5

        results.append({
            "timestep": t + 1,
            "alpha_t": cp.alpha,
            "q_hat_t": cp.q_hat,
            "band_half_width": half_width,
            "coverage_t": covered,
        })

        # Auto-regressive: feed prediction as next input (original scale)
        u_current = u_pred_t

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fno")
    parser.add_argument("--pde", type=str, default="navier_stokes")
    parser.add_argument("--data_dir", type=str, default="./data/pdebench")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--n_cal", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--aci_gamma", type=float, default=0.01)
    args = parser.parse_args()
    run_rollout_eval(args)
