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

## Windows Setup With NVIDIA GPU

Use these steps on a Windows machine that does not have Python installed yet.

1. Install Python 3.11 from the official website:

   https://www.python.org/downloads/release/python-3119/

   During installation, enable:

   ```text
   Add python.exe to PATH
   ```

2. Open PowerShell in the project folder:

   ```powershell
   cd "path\to\CV-FinalProject-810100116-810100254"
   ```

3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this only for the current PowerShell session:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\.venv\Scripts\Activate.ps1
   ```

4. Upgrade pip:

   ```powershell
   python -m pip install --upgrade pip
   ```

5. Install PyTorch with CUDA GPU support:

   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

6. Install the remaining project dependencies:

   ```powershell
   pip install datasets Pillow matplotlib tqdm numpy
   ```

7. Verify that PyTorch can see the GPU:

   ```powershell
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU found')"
   ```

   The first line should print:

   ```text
   True
   ```

8. Run the smoke test:

   ```powershell
   python training/train.py --epochs 1 --train-subset 16 --val-subset 8 --test-subset 8 --base-channels 16
   ```

9. Run full training:

   ```powershell
   python training/train.py --epochs 20 --batch-size 8 --image-size 256
   ```

If CUDA is not detected, update the NVIDIA driver and repeat steps 5 to 7. Do not use the CPU-only PyTorch install if you want GPU training.

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
