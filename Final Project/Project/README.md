# CV Final Project: Old Photo Restoration

This project trains a supervised deep learning model that maps a damaged old-photo image to its clean restored ground truth image.

The required dataset is used exactly:

```python
from datasets import load_dataset
data = load_dataset("joshuachin/openphoto-restore-dataset")
```

The Hugging Face dataset provides paired columns named `damaged_image` and `pristine_image`, with official `train` and `test` splits.

## Architecture Choice

I selected a U-Net style convolutional autoencoder.

Why this choice:

- It is an encoder-bottleneck-decoder architecture, so it satisfies the autoencoder family requested by the project.
- Skip connections preserve edges, identity, and fine spatial details that are important in restoration.
- It is much easier and more stable to train than a GAN under course-project compute limits.
- It is more appropriate than a VAE because the task is paired image-to-image restoration, not sampling diverse new photos from a latent distribution.

Rejected alternatives:

- Plain AutoEncoder: simple and efficient, but it often loses high-frequency detail because all information must pass through the bottleneck.
- VAE: useful for generative modeling, but KL regularization can blur outputs and is unnecessary for deterministic paired restoration.
- GAN: can produce sharper images, but it is more expensive, less stable, and requires careful adversarial balancing.

Main disadvantage of U-Net:

- It can still produce slightly smooth results compared with a well-trained GAN, especially with only pixel reconstruction losses.

## Folder Structure

```text
datasets/
models/
training/
utils/
outputs/
  checkpoints/
  restored_images/
  plots/
report/
main.py
requirements.txt
README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python training/train.py --epochs 20 --batch-size 8 --image-size 256
```

For a quick smoke test:

```bash
python training/train.py --epochs 1 --train-subset 16 --val-subset 8 --test-subset 8 --base-channels 16
```

## Evaluate

```bash
python training/evaluate.py --checkpoint outputs/checkpoints/best.pt
```

Evaluation writes:

- `outputs/test_metrics.json`
- `outputs/restored_images/evaluation_grid.png`

## Preprocessing

Each damaged and pristine image is:

- converted to RGB
- resized to `256x256` by default
- converted to a PyTorch tensor
- normalized from pixel range `0..255` to `0..1`

The model output uses `Sigmoid`, so the prediction range matches the normalized target range.

## Loss and Optimizer

Loss: weighted `0.85 * L1 + 0.15 * MSE`.

Reasoning:

- L1 tends to preserve sharper image structures.
- MSE helps stabilize brightness and color matching.
- The combined loss is simpler and more stable than adversarial or perceptual losses for this course setting.

Optimizer: Adam with:

- learning rate: `1e-4`
- weight decay: `1e-5`

## Metrics

The test script reports:

- MSE
- MAE
- PSNR
- SSIM

It also saves side-by-side visualizations ordered as damaged, restored, ground truth.
