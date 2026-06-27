"""Evaluate a trained restoration checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.train import get_device
from utils.dataset import DataConfig, create_dataloaders
from utils.metrics import batch_metrics
from utils.visualization import save_restoration_grid


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})
    model = build_model(base_channels=model_args.get("base_channels", args.base_channels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_config = DataConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        test_subset=args.test_subset,
    )
    _, _, test_loader = create_dataloaders(data_config)

    totals = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    batches = 0
    first_batch_saved = False
    output_dir = Path(args.output_dir)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            damaged = batch["damaged"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            restored = model(damaged)
            metrics = batch_metrics(restored, target)
            for key, value in metrics.items():
                totals[key] += value
            batches += 1

            if not first_batch_saved:
                save_restoration_grid(
                    damaged,
                    restored,
                    target,
                    output_dir / "restored_images" / "evaluation_grid.png",
                )
                first_batch_saved = True

    results = {key: value / max(1, batches) for key, value in totals.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained restoration model.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
