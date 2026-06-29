"""Visualization helpers for restored image samples and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image


def save_restoration_grid(
    damaged: torch.Tensor,
    restored: torch.Tensor,
    target: torch.Tensor,
    output_path: str | Path,
    max_samples: int = 4,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    damaged = damaged[:max_samples].cpu().clamp(0, 1)
    restored = restored[:max_samples].cpu().clamp(0, 1)
    target = target[:max_samples].cpu().clamp(0, 1)
    grid = make_grid(torch.cat([damaged, restored, target], dim=0), nrow=max_samples)
    save_image(grid, output_path)


def plot_losses(history: list[dict[str, float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
