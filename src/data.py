"""
Data loading utilities for neural operator experiments.

Supports two data sources:
  1. PDEBench HDF5 files (original, requires download)
  2. neuralop built-in datasets (auto-downloads from Zenodo, recommended)

Supported PDEs: 2D Darcy Flow, 1D Burgers, 2D Navier-Stokes,
1D Diffusion-Reaction, 2D Shallow Water.
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal


PDE_CONFIGS = {
    "darcy": {
        "filename": "2D_DarcyFlow_beta1.0_Train.hdf5",
        "input_key": "nu",       # permeability field
        "output_key": "tensor",  # solution field
        "spatial_dims": 2,
        "time_dependent": False,
        "resolution": 128,
    },
    "burgers": {
        "filename": "1D_Burgers_Sols_Nu0.001.hdf5",
        "input_key": "tensor",   # initial condition at t=0
        "output_key": "tensor",  # full spatiotemporal solution
        "spatial_dims": 1,
        "time_dependent": True,
        "resolution": 1024,
        "n_timesteps": 201,
    },
    "navier_stokes": {
        "filename": "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
        "input_key": "Vx",  # velocity x-component (also Vy, density, pressure)
        "output_key": "Vx",
        "spatial_dims": 2,
        "time_dependent": True,
        "resolution": 128,
        "n_timesteps": 21,
    },
    "diffusion_reaction": {
        "filename": "1D_diff-react_NA_NA.h5",
        "input_key": "tensor",
        "output_key": "tensor",
        "spatial_dims": 1,
        "time_dependent": True,
        "resolution": 1024,
        "n_timesteps": 101,
    },
    "shallow_water": {
        "filename": "2D_rdb_NA_NA.h5",
        "input_key": "tensor",
        "output_key": "tensor",
        "spatial_dims": 2,
        "time_dependent": True,
        "resolution": 128,
        "n_timesteps": 101,
    },
}


class PDEBenchDataset(Dataset):
    """Dataset wrapper for PDEBench HDF5 files.

    For static PDEs (Darcy): returns (input_field, output_field)
    For time-dependent PDEs: returns (u_t0, u_t1) pairs for operator learning,
        or full trajectory for rollout evaluation.
    """

    def __init__(
        self,
        data_dir: str,
        pde_name: str,
        mode: Literal["static", "one_step", "trajectory"] = "static",
        resolution: Optional[int] = None,
        max_samples: Optional[int] = None,
        t_in: int = 1,
        t_out: int = 1,
    ):
        """
        Args:
            data_dir: Path to PDEBench data directory
            pde_name: One of 'darcy', 'burgers', 'navier_stokes', etc.
            mode: 'static' for input->output, 'one_step' for u_t->u_{t+1},
                  'trajectory' for full time series
            resolution: Downsample to this resolution (None = use original)
            max_samples: Limit number of samples
            t_in: Number of input timesteps (for time-dependent)
            t_out: Number of output timesteps (for time-dependent)
        """
        self.config = PDE_CONFIGS[pde_name]
        self.pde_name = pde_name
        self.mode = mode
        self.resolution = resolution or self.config["resolution"]
        self.t_in = t_in
        self.t_out = t_out

        filepath = Path(data_dir) / self.config["filename"]
        if not filepath.exists():
            raise FileNotFoundError(
                f"PDEBench data not found at {filepath}. "
                f"Download from https://github.com/pdebench/PDEBench"
            )

        self.data = h5py.File(str(filepath), "r")
        self.n_samples = min(
            self.data[self.config["input_key"]].shape[0],
            max_samples or float("inf"),
        )

        # Detect actual spatial resolution from data
        # Data may be (N, H, W), (N, 1, H, W), etc.
        data_shape = self.data[self.config["input_key"]].shape
        actual_res = data_shape[-1]  # last dim is always spatial
        if actual_res != self.config["resolution"]:
            self._actual_resolution = actual_res
        else:
            self._actual_resolution = self.config["resolution"]

    def __len__(self) -> int:
        if self.mode == "one_step" and self.config["time_dependent"]:
            n_steps = self.config["n_timesteps"] - self.t_in - self.t_out + 1
            return self.n_samples * max(n_steps, 1)
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "static" or not self.config["time_dependent"]:
            return self._get_static(idx)
        elif self.mode == "one_step":
            return self._get_one_step(idx)
        else:
            return self._get_trajectory(idx)

    def _get_static(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """For static PDEs like Darcy flow."""
        x = torch.tensor(
            self.data[self.config["input_key"]][idx], dtype=torch.float32
        )
        y = torch.tensor(
            self.data[self.config["output_key"]][idx], dtype=torch.float32
        )
        # Squeeze leading channel dim if present: (1, H, W) -> (H, W)
        if x.ndim == self.config["spatial_dims"] + 1 and x.shape[0] == 1:
            x = x.squeeze(0)
        if y.ndim == self.config["spatial_dims"] + 1 and y.shape[0] == 1:
            y = y.squeeze(0)
        x = self._maybe_downsample(x)
        y = self._maybe_downsample(y)
        return x, y

    def _get_one_step(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """For time-dependent PDEs: returns (u_t, u_{t+1})."""
        n_steps = self.config["n_timesteps"] - self.t_in - self.t_out + 1
        sample_idx = idx // n_steps
        t_idx = idx % n_steps

        data = torch.tensor(
            self.data[self.config["output_key"]][sample_idx], dtype=torch.float32
        )
        x = data[t_idx : t_idx + self.t_in]
        y = data[t_idx + self.t_in : t_idx + self.t_in + self.t_out]

        if self.t_in == 1:
            x = x.squeeze(0)
        if self.t_out == 1:
            y = y.squeeze(0)

        x = self._maybe_downsample(x)
        y = self._maybe_downsample(y)
        return x, y

    def _get_trajectory(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (u_0, full_trajectory) for rollout evaluation."""
        data = torch.tensor(
            self.data[self.config["output_key"]][idx], dtype=torch.float32
        )
        # data shape: (T, *spatial) or (T, *spatial, C)
        u_0 = data[0]
        trajectory = data[1:]  # (T-1, *spatial)

        u_0 = self._maybe_downsample(u_0)
        trajectory = torch.stack([self._maybe_downsample(t) for t in trajectory])
        return u_0, trajectory

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample spatial dimensions if needed."""
        # Use actual data resolution, not config (synthetic data may differ)
        actual_res = self._actual_resolution
        if self.resolution >= actual_res:
            return x
        # Simple strided downsampling
        factor = actual_res // self.resolution
        if factor <= 1:
            return x
        if self.config["spatial_dims"] == 1:
            return x[..., ::factor]
        elif self.config["spatial_dims"] == 2:
            return x[..., ::factor, ::factor]
        return x

    def close(self):
        self.data.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_data_splits(
    data_dir: str,
    pde_name: str,
    mode: str = "static",
    resolution: Optional[int] = None,
    n_train: int = 800,
    n_cal: int = 100,
    n_test: int = 100,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/calibration/test data loaders.

    The split is important for conformal prediction validity:
    - Train: fit the neural operator
    - Calibration: compute conformal quantile (exchangeable with test)
    - Test: evaluate coverage

    For learned spectral weights, the calibration set should be further
    split using three_way_split() from conformal.py.
    """
    dataset = PDEBenchDataset(
        data_dir, pde_name, mode=mode, resolution=resolution,
        max_samples=n_train + n_cal + n_test,
    )

    # Deterministic split
    rng = torch.Generator().manual_seed(seed)
    total = len(dataset)
    indices = torch.randperm(total, generator=rng).tolist()

    train_idx = indices[:n_train]
    cal_idx = indices[n_train : n_train + n_cal]
    test_idx = indices[n_train + n_cal : n_train + n_cal + n_test]

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=32, shuffle=True,
    )
    cal_loader = DataLoader(
        torch.utils.data.Subset(dataset, cal_idx),
        batch_size=len(cal_idx), shuffle=False,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=len(test_idx), shuffle=False,
    )

    return train_loader, cal_loader, test_loader


def get_neuralop_darcy_splits(
    n_train: int = 800,
    n_cal: int = 100,
    n_test: int = 100,
    resolution: int = 16,
    data_root: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load Darcy Flow data from neuralop's built-in dataset (Zenodo).

    Auto-downloads ~10MB of .pt files. No external HDF5 needed.
    Returns train/cal/test loaders with (x, y) tuples where x, y
    have shape (*spatial) — no channel dim, matching PDEBench format.

    Available resolutions: 16, 32, 64, 128.
    """
    from neuralop.data.datasets.darcy import DarcyDataset

    if data_root is None:
        data_root = str(Path("./data/neuralop_darcy"))

    total = n_train + n_cal + n_test

    # Use DarcyDataset directly for control over resolution
    dataset = DarcyDataset(
        root_dir=data_root,
        n_train=total,
        n_tests=[1],  # minimal test set (we do our own split)
        batch_size=total,
        test_batch_sizes=[1],
        train_resolution=resolution,
        test_resolutions=[resolution],
        encode_input=False,
        encode_output=False,
        download=True,
    )

    # Extract all training data as tensors
    # neuralop stores as dict or tuple — handle both
    all_data = list(dataset.train_db)
    if isinstance(all_data[0], dict):
        all_x = torch.stack([item['x'] for item in all_data]).squeeze(1)
        all_y = torch.stack([item['y'] for item in all_data]).squeeze(1)
    else:
        all_x = torch.stack([item[0] for item in all_data]).squeeze(1)
        all_y = torch.stack([item[1] for item in all_data]).squeeze(1)

    # Deterministic split
    rng = torch.Generator().manual_seed(seed)
    n = len(all_x)
    perm = torch.randperm(n, generator=rng)

    train_x, train_y = all_x[perm[:n_train]], all_y[perm[:n_train]]
    cal_x, cal_y = all_x[perm[n_train:n_train+n_cal]], all_y[perm[n_train:n_train+n_cal]]
    test_x, test_y = all_x[perm[n_train+n_cal:n_train+n_cal+n_test]], all_y[perm[n_train+n_cal:n_train+n_cal+n_test]]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=32, shuffle=True,
    )
    cal_loader = DataLoader(
        TensorDataset(cal_x, cal_y),
        batch_size=n_cal, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=n_test, shuffle=False,
    )

    print(f"  neuralop Darcy: train={n_train}, cal={n_cal}, test={n_test}, "
          f"res={resolution}, x={train_x.shape}")

    return train_loader, cal_loader, test_loader
