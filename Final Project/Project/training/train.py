"""Training entry point for the OpenPhoto restoration model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.losses import RestorationLoss
from utils.dataset import DataConfig, create_dataloaders
from utils.metrics import batch_metrics
from utils.visualization import plot_losses, save_restoration_grid


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    metric_totals = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            damaged = batch["damaged"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            prediction = model(damaged)
            loss = criterion(prediction, target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            metrics = batch_metrics(prediction.detach(), target)
            for key, value in metrics.items():
                metric_totals[key] += value
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_metrics = {key: value / max(1, num_batches) for key, value in metric_totals.items()}
    return avg_loss, avg_metrics


def save_history(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    restored_dir = output_dir / "restored_images"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
    )
    train_loader, val_loader, test_loader = create_dataloaders(data_config)

    device = get_device()
    model = build_model(base_channels=args.base_channels).to(device)
    criterion = RestorationLoss(l1_weight=args.l1_weight, mse_weight=args.mse_weight)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_psnr": train_metrics["psnr"],
            "val_psnr": val_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
            "val_ssim": val_metrics["ssim"],
        }
        history.append(row)
        print(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_history(history, output_dir / "training_history.csv")
    plot_losses(history, plots_dir / "loss_curve.png")

    model.eval()
    with torch.no_grad():
        sample = next(iter(test_loader))
        damaged = sample["damaged"].to(device)
        target = sample["target"].to(device)
        restored = model(damaged)
    save_restoration_grid(damaged, restored, target, restored_dir / "test_restoration_grid.png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net for old-photo restoration.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1-weight", type=float, default=0.85)
    parser.add_argument("--mse-weight", type=float, default=0.15)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
