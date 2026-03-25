"""
Data loading utilities for PDEBench datasets.

Supports: 2D Darcy Flow, 1D Burgers, 2D Navier-Stokes,
1D Diffusion-Reaction, 2D Shallow Water.

Data format: HDF5 files from PDEBench (https://github.com/pdebench/PDEBench)
"""

import torch
from torch.utils.data import Dataset, DataLoader
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
        orig_res = self.config["resolution"]
        if self.resolution >= orig_res:
            return x
        # Simple strided downsampling
        factor = orig_res // self.resolution
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
