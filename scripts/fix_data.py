#!/usr/bin/env python3
"""
Emergency Darcy Flow data fix for RunPod.
Tries 3 methods in order; guarantees data or exits with clear error.

Method 1: HuggingFace Hub (pdebench/PDEBench mirror)
Method 2: neuralop 2.0 built-in small Darcy dataset
Method 3: Synthetic generation via finite differences (guaranteed, ~2 min)

Usage:
    python fix_data.py                          # default: ./data/pdebench
    python fix_data.py --data_dir /workspace/neurips2026-s2/data/pdebench
    python fix_data.py --method synthetic       # force synthetic
    python fix_data.py --n_samples 1000 --resolution 128

Output: {data_dir}/2D_DarcyFlow_beta1.0_Train.hdf5
        with keys 'nu' (permeability, N×128×128) and 'tensor' (solution, N×128×128)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import h5py


def method_huggingface(data_dir: Path, target_file: str) -> bool:
    """Download from HuggingFace Hub (pdebench/PDEBench dataset)."""
    print("\n[Method 1] HuggingFace Hub download...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    try:
        # PDEBench data on HF: https://huggingface.co/datasets/pdebench/PDEBench
        path = hf_hub_download(
            repo_id="pdebench/PDEBench",
            filename="2D/CFD/2D_DarcyFlow_beta1.0_Train.hdf5",
            repo_type="dataset",
            local_dir=str(data_dir),
            local_dir_use_symlinks=False,
        )
        # HF might nest it; move to expected location
        dest = data_dir / target_file
        src = Path(path)
        if src != dest:
            if src.exists():
                import shutil
                shutil.move(str(src), str(dest))
                print(f"  Moved to {dest}")

        # Verify
        with h5py.File(str(dest), "r") as f:
            keys = list(f.keys())
            n = f[keys[0]].shape[0]
            print(f"  OK: {dest} — keys={keys}, n={n}")
        return True

    except Exception as e:
        print(f"  Failed: {e}")
        # Try alternative HF path
        try:
            path = hf_hub_download(
                repo_id="pdebench/PDEBench",
                filename="2D_DarcyFlow_beta1.0_Train.hdf5",
                repo_type="dataset",
                local_dir=str(data_dir),
                local_dir_use_symlinks=False,
            )
            dest = data_dir / target_file
            src = Path(path)
            if src != dest and src.exists():
                import shutil
                shutil.move(str(src), str(dest))
            with h5py.File(str(dest), "r") as f:
                print(f"  OK (alt path): keys={list(f.keys())}")
            return True
        except Exception as e2:
            print(f"  Alt path also failed: {e2}")
            return False


def method_neuralop(data_dir: Path, target_file: str) -> bool:
    """Try neuralop 2.0 built-in Darcy dataset."""
    print("\n[Method 2] neuralop built-in Darcy loader...")

    # Probe multiple import paths (API changed across versions)
    loaders = [
        ("neuralop.data.datasets.darcy", "load_darcy_flow_small"),
        ("neuralop.data.datasets", "load_darcy_flow_small"),
        ("neuralop.datasets", "load_darcy_pt"),
        ("neuralop.data", "load_darcy_flow_small"),
    ]

    for module_path, func_name in loaders:
        try:
            mod = __import__(module_path, fromlist=[func_name])
            loader = getattr(mod, func_name)
            print(f"  Found: {module_path}.{func_name}")

            result = loader(
                n_train=900, n_tests=[100],
                batch_size=32, test_batch_sizes=[100],
                resolution=128,
            )

            # Convert to HDF5
            print("  Converting to HDF5...")
            train_loader = result.train_loader if hasattr(result, 'train_loader') else result[0]
            all_x, all_y = [], []
            for batch in train_loader:
                x, y = batch['x'], batch['y']
                all_x.append(x.numpy())
                all_y.append(y.numpy())

            all_x = np.concatenate(all_x, axis=0)
            all_y = np.concatenate(all_y, axis=0)

            dest = data_dir / target_file
            with h5py.File(str(dest), "w") as f:
                f.create_dataset("nu", data=all_x.squeeze())
                f.create_dataset("tensor", data=all_y.squeeze())

            print(f"  OK: {dest} — nu={all_x.shape}, tensor={all_y.shape}")
            return True

        except (ImportError, AttributeError, TypeError) as e:
            print(f"  {module_path}.{func_name}: {e}")
            continue
        except Exception as e:
            print(f"  {module_path}.{func_name} runtime error: {e}")
            continue

    print("  No working neuralop data loader found.")
    return False


def method_synthetic(data_dir: Path, target_file: str,
                     n_samples: int = 1000, resolution: int = 128) -> bool:
    """Generate synthetic Darcy Flow data via finite differences.

    Solves: -∇·(a(x)∇u(x)) = f(x) on [0,1]^2, u=0 on boundary.
    - a(x) = permeability field (log-normal random field)
    - f(x) = 1 (constant forcing)
    - Finite difference discretization → sparse linear solve

    This is the EXACT same PDE as PDEBench Darcy Flow.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    print(f"\n[Method 3] Synthetic Darcy Flow generation...")
    print(f"  n_samples={n_samples}, resolution={resolution}")
    print(f"  PDE: -div(a(x) grad(u)) = 1, u=0 on boundary")

    N = resolution
    h = 1.0 / (N + 1)  # grid spacing (interior points only)

    def generate_permeability(rng, N, length_scale=0.1, beta=1.0):
        """Generate log-normal random permeability field.

        Uses random Fourier features for a Gaussian random field,
        then exponentiates. beta controls variance.
        """
        # Generate on full grid (including boundary for the field visualization)
        coords = np.linspace(0, 1, N)
        xx, yy = np.meshgrid(coords, coords, indexing='ij')

        # Random Fourier features for approximate Gaussian RF
        n_features = 200
        freqs = rng.normal(0, 1.0 / length_scale, size=(n_features, 2))
        phases = rng.uniform(0, 2 * np.pi, size=n_features)

        pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N^2, 2)
        proj = pts @ freqs.T + phases[None, :]  # (N^2, n_features)
        rf = np.cos(proj).mean(axis=1) * np.sqrt(2)  # (N^2,)
        rf = rf.reshape(N, N)

        # Log-normal: a(x) = exp(beta * z(x))
        a = np.exp(beta * rf)
        # Clip to avoid extreme values
        a = np.clip(a, 0.01, 100.0)
        return a

    def solve_darcy(a_full, N, h):
        """Solve -div(a * grad(u)) = 1 with u=0 on boundary.

        a_full: permeability on full N×N grid
        Returns u on full N×N grid (boundary = 0).
        """
        # Interior grid: (N-2) × (N-2) points
        M = N - 2
        n_interior = M * M

        if n_interior == 0:
            return np.zeros((N, N))

        # Build sparse matrix for 5-point stencil
        # -d/dx(a du/dx) - d/dy(a du/dy) = f
        # Discretized with harmonic averaging of a at cell faces

        def idx(i, j):
            return i * M + j

        rows, cols, vals = [], [], []
        rhs = np.ones(n_interior)  # f = 1

        for i in range(M):
            for j in range(M):
                # Interior indices map to full grid as (i+1, j+1)
                gi, gj = i + 1, j + 1
                k = idx(i, j)

                # Harmonic mean of permeability at cell faces
                a_c = a_full[gi, gj]

                # East face: between (gi, gj) and (gi, gj+1)
                a_e = 2.0 * a_c * a_full[gi, gj + 1] / (a_c + a_full[gi, gj + 1] + 1e-12)
                # West face
                a_w = 2.0 * a_c * a_full[gi, gj - 1] / (a_c + a_full[gi, gj - 1] + 1e-12)
                # North face
                a_n = 2.0 * a_c * a_full[gi - 1, gj] / (a_c + a_full[gi - 1, gj] + 1e-12)
                # South face
                a_s = 2.0 * a_c * a_full[gi + 1, gj] / (a_c + a_full[gi + 1, gj] + 1e-12)

                diag = (a_e + a_w + a_n + a_s) / (h * h)
                rows.append(k); cols.append(k); vals.append(diag)

                # East neighbor
                if j + 1 < M:
                    rows.append(k); cols.append(idx(i, j + 1)); vals.append(-a_e / (h * h))
                # West neighbor
                if j - 1 >= 0:
                    rows.append(k); cols.append(idx(i, j - 1)); vals.append(-a_w / (h * h))
                # North neighbor
                if i - 1 >= 0:
                    rows.append(k); cols.append(idx(i - 1, j)); vals.append(-a_n / (h * h))
                # South neighbor
                if i + 1 < M:
                    rows.append(k); cols.append(idx(i + 1, j)); vals.append(-a_s / (h * h))

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_interior, n_interior))
        u_interior = spsolve(A, rhs)

        # Place solution on full grid
        u_full = np.zeros((N, N))
        u_full[1:-1, 1:-1] = u_interior.reshape(M, M)
        return u_full

    # Generate samples
    rng = np.random.default_rng(42)
    all_a = np.zeros((n_samples, N, N), dtype=np.float32)
    all_u = np.zeros((n_samples, N, N), dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        a = generate_permeability(rng, N, length_scale=0.1, beta=1.0)
        u = solve_darcy(a, N, h)
        all_a[i] = a.astype(np.float32)
        all_u[i] = u.astype(np.float32)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  [{i+1}/{n_samples}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    total_time = time.time() - t0
    print(f"  Generated {n_samples} samples in {total_time:.1f}s")

    # Save as HDF5 matching PDEBench format
    dest = data_dir / target_file
    with h5py.File(str(dest), "w") as f:
        f.create_dataset("nu", data=all_a)
        f.create_dataset("tensor", data=all_u)
        f.attrs["source"] = "synthetic_finite_difference"
        f.attrs["pde"] = "darcy_flow"
        f.attrs["beta"] = 1.0
        f.attrs["resolution"] = N
        f.attrs["n_samples"] = n_samples
        f.attrs["boundary"] = "dirichlet_zero"
        f.attrs["forcing"] = "constant_1"

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Saved: {dest} ({size_mb:.1f} MB)")
    print(f"  Keys: nu={all_a.shape}, tensor={all_u.shape}")

    # Quick sanity check
    print(f"  Sanity: a range [{all_a.min():.3f}, {all_a.max():.3f}]")
    print(f"  Sanity: u range [{all_u.min():.6f}, {all_u.max():.6f}]")
    if all_u.max() < 1e-10:
        print("  WARNING: Solutions are near-zero. Check solver.")
        return False

    return True


def verify_and_repair_data(data_dir: Path, target_file: str) -> bool:
    """Verify the HDF5 file is loadable by our pipeline.

    Also repairs common issues:
    - Squeezes (N, 1, H, W) -> (N, H, W) to match PDEBench format
    """
    dest = data_dir / target_file
    if not dest.exists():
        return False
    try:
        needs_repair = False
        with h5py.File(str(dest), "r") as f:
            assert "nu" in f, f"Missing 'nu' key. Keys: {list(f.keys())}"
            assert "tensor" in f, f"Missing 'tensor' key. Keys: {list(f.keys())}"
            n = f["nu"].shape[0]
            assert n >= 10, f"Too few samples: {n}"

            # Check for extra channel dim: (N, 1, H, W) -> needs squeeze
            if f["nu"].ndim == 4 and f["nu"].shape[1] == 1:
                print(f"  REPAIR: Squeezing (N,1,H,W) -> (N,H,W)")
                needs_repair = True

            res = f["nu"].shape[-1]  # last dim is spatial
            assert res >= 64, f"Resolution too low: {res}"

        if needs_repair:
            with h5py.File(str(dest), "r") as f:
                nu = f["nu"][:].squeeze(1) if f["nu"].ndim == 4 else f["nu"][:]
                tensor = f["tensor"][:].squeeze(1) if f["tensor"].ndim == 4 else f["tensor"][:]
            # Rewrite
            with h5py.File(str(dest), "w") as f:
                f.create_dataset("nu", data=nu)
                f.create_dataset("tensor", data=tensor)
            print(f"  REPAIRED: {dest}")

        with h5py.File(str(dest), "r") as f:
            print(f"\n  VERIFIED: {dest}")
            print(f"    nu:     {f['nu'].shape} {f['nu'].dtype}")
            print(f"    tensor: {f['tensor'].shape} {f['tensor'].dtype}")
            return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Emergency Darcy Flow data fix")
    parser.add_argument("--data_dir", type=str, default="./data/pdebench")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "huggingface", "neuralop", "synthetic"])
    parser.add_argument("--n_samples", type=int, default=1200)
    parser.add_argument("--resolution", type=int, default=128,
                        help="Must match PDE_CONFIGS['darcy']['resolution'] (128)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target_file = "2D_DarcyFlow_beta1.0_Train.hdf5"

    print("=" * 60)
    print("  S2: Emergency Darcy Flow Data Fix")
    print(f"  Target: {data_dir / target_file}")
    print("=" * 60)

    # Check if already exists and valid
    if verify_and_repair_data(data_dir, target_file):
        print("\n  Data already exists and is valid. Nothing to do.")
        return

    methods = {
        "huggingface": lambda: method_huggingface(data_dir, target_file),
        "neuralop": lambda: method_neuralop(data_dir, target_file),
        "synthetic": lambda: method_synthetic(
            data_dir, target_file, args.n_samples, args.resolution
        ),
    }

    if args.method != "auto":
        ok = methods[args.method]()
        if ok and verify_and_repair_data(data_dir, target_file):
            print("\n  SUCCESS")
            sys.exit(0)
        else:
            print(f"\n  FAILED: method '{args.method}'")
            sys.exit(1)

    # Auto: try all methods in order
    for name, func in methods.items():
        try:
            ok = func()
            if ok and verify_and_repair_data(data_dir, target_file):
                print(f"\n  SUCCESS via {name}")
                sys.exit(0)
        except Exception as e:
            print(f"  {name} crashed: {e}")
            continue

    print("\n  ALL METHODS FAILED. Cannot continue.")
    sys.exit(1)


if __name__ == "__main__":
    main()
