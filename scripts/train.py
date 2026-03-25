"""
Training script for neural operator models on PDEBench.

Usage:
    python scripts/train.py --model fno --pde darcy --resolution 64 --epochs 100

Saves checkpoints to checkpoints/{model}_{pde}_{resolution}/
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.data import get_data_splits, PDE_CONFIGS
from src.utils import ensure_channel_dim, append_grid


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 loss: ||pred - target||_2 / ||target||_2.

    Standard loss for neural operators (used in neuraloperator library).
    Unlike MSE, this normalizes by solution magnitude, giving equal weight
    to solutions of different scales.
    """
    batch_size = pred.shape[0]
    diff_norm = pred.reshape(batch_size, -1).sub(target.reshape(batch_size, -1)).pow(2).sum(dim=-1).sqrt()
    target_norm = target.reshape(batch_size, -1).pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
    return (diff_norm / target_norm).mean()


def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    config = PDE_CONFIGS[args.pde]
    spatial_dims = config["spatial_dims"]
    mode = "one_step" if config["time_dependent"] else "static"

    train_loader, cal_loader, test_loader = get_data_splits(
        data_dir=args.data_dir,
        pde_name=args.pde,
        mode=mode,
        resolution=args.resolution,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
    )

    # Compute normalization statistics from training data
    print("Computing normalization statistics...")
    all_x, all_y = [], []
    for x_batch, y_batch in train_loader:
        all_x.append(x_batch)
        all_y.append(y_batch)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    x_mean, x_std = all_x.mean().item(), all_x.std().item()
    y_mean, y_std = all_y.mean().item(), all_y.std().item()
    # Clamp std to avoid division by zero for near-constant fields
    x_std = max(x_std, 1e-8)
    y_std = max(y_std, 1e-8)
    print(f"  x: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"  y: mean={y_mean:.4f}, std={y_std:.4f}")
    del all_x, all_y

    # Model — in_channels = 1 (field) + spatial_dims (grid coordinates)
    in_channels = 1 + spatial_dims
    model = get_model(
        model_name=args.model,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=1,
        resolution=args.resolution,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {n_params:,}")

    # Training
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir) / f"{args.model}_{args.pde}_{args.resolution}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Normalize
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std

            # Ensure channel dim: (batch, *spatial) -> (batch, 1, *spatial)
            x = ensure_channel_dim(x, spatial_dims)
            y = ensure_channel_dim(y, spatial_dims)

            # Append grid coordinates as extra input channels
            x = append_grid(x, spatial_dims)

            pred = model(x)
            loss = relative_l2_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs} | RelL2: {avg_loss:.6f} | Time: {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "in_channels": in_channels,
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "in_channels": in_channels,
        "epoch": args.epochs - 1,
        "loss": avg_loss,
    }, save_dir / "final.pt")
    print(f"Training complete. Best relative L2: {best_loss:.6f}")
    print(f"Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fno", choices=["fno", "tfno", "deeponet", "uno"])
    parser.add_argument("--pde", type=str, default="darcy", choices=list(PDE_CONFIGS.keys()))
    parser.add_argument("--data_dir", type=str, default="./data/pdebench")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_cal", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    args = parser.parse_args()
    train(args)
