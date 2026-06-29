"""Image restoration metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(prediction, target).item()


def mae_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(prediction, target).item()


def psnr_score(prediction: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    mse = F.mse_loss(prediction, target)
    if mse.item() == 0:
        return float("inf")
    return (20 * torch.log10(torch.tensor(max_value, device=prediction.device)) - 10 * torch.log10(mse)).item()


def ssim_score(prediction: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    """Compute local-window SSIM over a batch."""

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    window_size = 11
    padding = window_size // 2

    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean().item()


def batch_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    return {
        "mse": mse_score(prediction, target),
        "mae": mae_score(prediction, target),
        "psnr": psnr_score(prediction, target),
        "ssim": ssim_score(prediction, target),
    }
