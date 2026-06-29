"""Loss functions for image restoration."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RestorationLoss(nn.Module):
    """Weighted L1 + MSE reconstruction loss.

    L1 preserves sharper details than MSE alone, while a smaller MSE term
    stabilizes pixel-level color and brightness matching.
    """

    def __init__(self, l1_weight: float = 0.85, mse_weight: float = 0.15) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_weight * F.l1_loss(prediction, target) + self.mse_weight * F.mse_loss(prediction, target)
