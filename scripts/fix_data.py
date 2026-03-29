#!/usr/bin/env python3
"""
Emergency Darcy Flow data fix for RunPod.
Tries 2 methods in order (REAL DATA ONLY - NO SYNTHETIC).

Method 1: HuggingFace Hub (pdebench/PDEBench mirror)
Method 2: neuralop 2.0 built-in small Darcy dataset

Usage:
    python fix_data.py                          # default: ./data/pdebench
    python fix_data.py --data_dir /workspace/neurips2026-s2/data/pdebench
    python fix_data.py --method huggingface    # force specific method

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


(data_dir: Path, target_file: str) -> bool:
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
                        choices=["auto", "huggingface", "neuralop"])
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
    }

    if args.method != "auto":
        ok = methods[args.method]()
        if ok and verify_and_repair_data(data_dir, target_file):
            print("\n  SUCCESS")
            sys.exit(0)
        else:
            print(f"\n  FAILED: method '{args.method}'")
            sys.exit(1)

    # Auto: try all methods in order (REAL DATA ONLY - NO SYNTHETIC)
    for name, func in methods.items():
        try:
            ok = func()
            if ok and verify_and_repair_data(data_dir, target_file):
                print(f"\n  SUCCESS via {name}")
                sys.exit(0)
        except Exception as e:
            print(f"  {name} crashed: {e}")
            continue

    print("\n  ERROR: All real data methods failed.")
    print("  NO SYNTHETIC DATA ALLOWED - This is a hard requirement.")
    print("  Please check network connectivity or data sources.")
    sys.exit(1)


if __name__ == "__main__":
    main()
