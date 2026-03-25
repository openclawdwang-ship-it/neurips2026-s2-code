"""
Shared utilities for neural operator training and inference.
"""

import torch


def ensure_channel_dim(x: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Add channel dimension if missing.

    Neural operators expect (batch, channels, *spatial). Data may arrive as
    (batch, *spatial) without the channel dim. We detect this by checking
    if ndim == spatial_dims + 1 (batch + spatial, no channel).
    """
    if x.ndim == spatial_dims + 1:
        x = x.unsqueeze(1)
    return x


def append_grid(x: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Append spatial grid coordinates as extra input channels.

    Standard FNO practice from the neuraloperator library: the model receives
    (input_field, x_coord, y_coord) so it can learn position-dependent features.
    Grid is on [0, 1]^d.

    Args:
        x: (batch, C, *spatial) tensor with channel dim already present
        spatial_dims: 1 or 2
    Returns:
        (batch, C + spatial_dims, *spatial) tensor
    """
    batch_size = x.shape[0]
    if spatial_dims == 1:
        nx = x.shape[-1]
        gx = torch.linspace(0, 1, nx, device=x.device).reshape(1, 1, nx).expand(batch_size, -1, -1)
        return torch.cat([x, gx], dim=1)
    elif spatial_dims == 2:
        nx, ny = x.shape[-2], x.shape[-1]
        gx = torch.linspace(0, 1, nx, device=x.device)
        gy = torch.linspace(0, 1, ny, device=x.device)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([x, grid], dim=1)
    return x


def prepare_input(x: torch.Tensor, spatial_dims: int, use_grid: bool = True) -> torch.Tensor:
    """Full input preparation: channel dim + optional grid coordinates.

    Args:
        x: Raw input tensor (batch, *spatial) or (batch, C, *spatial)
        spatial_dims: 1 or 2
        use_grid: Whether to append grid coordinates (True for new checkpoints)
    """
    x = ensure_channel_dim(x, spatial_dims)
    if use_grid:
        x = append_grid(x, spatial_dims)
    return x
