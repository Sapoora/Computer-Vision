"""Preprocessing transforms for paired restoration images."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import torch
from torchvision import transforms


@dataclass(frozen=True)
class RestorationTransforms:
    """Apply identical deterministic preprocessing to input and target images."""

    image_size: int = 256

    def __post_init__(self) -> None:
        if self.image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for the four-level U-Net.")
        object.__setattr__(
            self,
            "_transform",
            transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                ]
            ),
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image.convert("RGB"))
