"""
Neural operator model wrappers.

Supports FNO, TFNO from the neuraloperator library,
and custom DeepONet implementation.

These are thin wrappers — the actual training is done in scripts/train.py.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


def get_model(
    model_name: Literal["fno", "tfno", "deeponet", "uno"],
    spatial_dims: int = 2,
    in_channels: int = 1,
    out_channels: int = 1,
    resolution: int = 64,
    **kwargs,
) -> nn.Module:
    """Factory function for neural operator models.

    Args:
        model_name: Model architecture name
        spatial_dims: 1 or 2
        in_channels: Number of input channels
        out_channels: Number of output channels
        resolution: Spatial resolution
    """
    if model_name == "fno":
        return _build_fno(spatial_dims, in_channels, out_channels, resolution, **kwargs)
    elif model_name == "tfno":
        return _build_tfno(spatial_dims, in_channels, out_channels, resolution, **kwargs)
    elif model_name == "deeponet":
        return _build_deeponet(spatial_dims, in_channels, out_channels, resolution, **kwargs)
    elif model_name == "uno":
        return _build_uno(spatial_dims, in_channels, out_channels, resolution, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_fno(spatial_dims, in_channels, out_channels, resolution, **kwargs):
    """Build FNO from neuraloperator library."""
    try:
        from neuralop.models import FNO
    except ImportError:
        raise ImportError("Install neuraloperator: pip install neuraloperator")

    n_modes = kwargs.get("n_modes", (16,) * spatial_dims)
    hidden_channels = kwargs.get("hidden_channels", 64)
    n_layers = kwargs.get("n_layers", 4)

    model = FNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )
    return model


def _build_tfno(spatial_dims, in_channels, out_channels, resolution, **kwargs):
    """Build Tucker-Factorized FNO."""
    try:
        from neuralop.models import TFNO
    except ImportError:
        raise ImportError("Install neuraloperator: pip install neuraloperator")

    n_modes = kwargs.get("n_modes", (16,) * spatial_dims)
    hidden_channels = kwargs.get("hidden_channels", 32)
    n_layers = kwargs.get("n_layers", 4)

    model = TFNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )
    return model


def _build_deeponet(spatial_dims, in_channels, out_channels, resolution, **kwargs):
    """Build a simple DeepONet (branch + trunk architecture)."""
    branch_width = kwargs.get("branch_width", 128)
    trunk_width = kwargs.get("trunk_width", 128)
    n_basis = kwargs.get("n_basis", 64)

    class DeepONet(nn.Module):
        def __init__(self):
            super().__init__()
            input_size = resolution ** spatial_dims * in_channels

            self.branch = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, branch_width),
                nn.GELU(),
                nn.Linear(branch_width, branch_width),
                nn.GELU(),
                nn.Linear(branch_width, n_basis),
            )

            self.trunk = nn.Sequential(
                nn.Linear(spatial_dims, trunk_width),
                nn.GELU(),
                nn.Linear(trunk_width, trunk_width),
                nn.GELU(),
                nn.Linear(trunk_width, n_basis),
            )

            self.bias = nn.Parameter(torch.zeros(1))
            self._resolution = resolution
            self._spatial_dims = spatial_dims
            self._out_channels = out_channels

            # Pre-compute grid
            self._grid = None

        def _get_grid(self, device):
            if self._grid is None or self._grid.device != device:
                coords = [torch.linspace(0, 1, self._resolution, device=device)]
                if self._spatial_dims == 2:
                    grids = torch.meshgrid(*coords * 2, indexing="ij")
                    self._grid = torch.stack(grids, dim=-1).reshape(-1, 2)
                else:
                    self._grid = coords[0].unsqueeze(-1)
            return self._grid

        def forward(self, x):
            batch_size = x.shape[0]
            grid = self._get_grid(x.device)  # (n_points, spatial_dims)

            branch_out = self.branch(x)  # (batch, n_basis)
            trunk_out = self.trunk(grid)  # (n_points, n_basis)

            # Dot product + bias
            output = torch.einsum("bp,np->bn", branch_out, trunk_out) + self.bias
            # Reshape to spatial
            if self._spatial_dims == 2:
                output = output.reshape(batch_size, self._resolution, self._resolution)
            else:
                output = output.reshape(batch_size, self._resolution)

            return output

    return DeepONet()


def _build_uno(spatial_dims, in_channels, out_channels, resolution, **kwargs):
    """Build U-shaped Neural Operator (U-NO).
    Uses a U-Net-like architecture with spectral convolutions.
    """
    try:
        from neuralop.models import UNO
        return UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=kwargs.get("hidden_channels", 64),
            n_layers=kwargs.get("n_layers", 4),
        )
    except (ImportError, AttributeError):
        # Fallback: simple U-Net style with FNO blocks
        return _build_fno(spatial_dims, in_channels, out_channels, resolution, **kwargs)


def load_trained_model(
    checkpoint_path: str,
    model_name: str,
    **model_kwargs,
) -> nn.Module:
    """Load a trained model from checkpoint.

    Handles both legacy checkpoints (plain state_dict) and new format
    (dict with model_state_dict + normalization stats).

    The normalization stats are attached as model attributes:
        model.x_mean, model.x_std, model.y_mean, model.y_std
    These are None if loaded from a legacy checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Read in_channels from checkpoint to build model with correct shape
        saved_in_channels = ckpt.get("in_channels")
        if saved_in_channels is not None:
            model_kwargs["in_channels"] = saved_in_channels

        model = get_model(model_name, **model_kwargs)
        model.load_state_dict(ckpt["model_state_dict"])
        model.x_mean = ckpt.get("x_mean")
        model.x_std = ckpt.get("x_std")
        model.y_mean = ckpt.get("y_mean")
        model.y_std = ckpt.get("y_std")
        model.in_channels = saved_in_channels
    else:
        # Legacy checkpoint: plain state_dict, no normalization
        model = get_model(model_name, **model_kwargs)
        model.load_state_dict(ckpt)
        model.x_mean = None
        model.x_std = None
        model.y_mean = None
        model.y_std = None
        model.in_channels = None

    model.eval()
    return model
