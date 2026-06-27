# Project Plan

## Objective

The objective is supervised restoration of damaged old photographs. For each paired sample, the model receives a damaged image and predicts the corresponding clean ground-truth image.

## Supervised Learning Formulation

- Input: `damaged_image`
- Target: `pristine_image`
- Function learned: `f(damaged_image) -> restored_image`
- Training signal: direct comparison between restored output and pristine target
- Output requirement: restored image must have the same height, width, and channels as the input

## Expected Pipeline

1. Load `joshuachin/openphoto-restore-dataset` using Hugging Face Datasets.
2. Read paired `damaged_image` and `pristine_image` fields.
3. Split the official training split into train and validation subsets.
4. Keep the official test split for final evaluation.
5. Resize images to 256x256.
6. Convert images to RGB tensors in `[0, 1]`.
7. Train a U-Net restoration model with Adam.
8. Track train and validation loss.
9. Evaluate on the test split with MSE, MAE, PSNR, and SSIM.
10. Save checkpoints, plots, and visual restored examples.

## Architecture Candidates

### Plain AutoEncoder

Advantages:

- Simple and computationally cheap.
- Directly matches damaged-to-clean reconstruction.

Disadvantages:

- A narrow bottleneck can discard fine texture, facial detail, and scratch boundaries.
- Outputs are often blurry in restoration tasks.

### Variational AutoEncoder

Advantages:

- Learns a structured latent space.
- Useful when generation and sampling are project goals.

Disadvantages:

- KL regularization often encourages smooth or blurry outputs.
- The restoration target is deterministic and paired, so stochastic latent sampling is not necessary.
- More complex than needed for this dataset.

### GAN

Advantages:

- Can produce perceptually sharp and realistic outputs.
- Adversarial loss can recover plausible texture.

Disadvantages:

- More computationally expensive.
- Harder to train stably.
- Requires discriminator design, adversarial loss balancing, and more tuning.
- Can hallucinate details that do not match the ground truth.

### Selected Model: U-Net AutoEncoder

I selected a U-Net because it is the best fit for paired image-to-image restoration under course-project constraints.

Advantages:

- Encoder learns global context and damage patterns.
- Bottleneck captures semantic structure.
- Decoder reconstructs the restored RGB image.
- Skip connections copy high-resolution spatial features into the decoder, preserving edges and identity.
- Training is stable with pixel reconstruction losses.
- Compute cost is moderate and can be reduced by lowering `base_channels`.

Disadvantages:

- Pixel losses can still produce smoother outputs than adversarial methods.
- Larger image sizes such as 512x512 increase memory cost substantially.

Expected restoration quality:

- Good structural restoration and color/brightness correction.
- Better detail preservation than a plain autoencoder.
- More faithful to ground truth than a GAN in a limited-compute setting.

## Model Details

- Encoder: repeated convolution blocks and max pooling.
- Bottleneck: deepest convolution block with the largest receptive field.
- Decoder: transposed convolutions for upsampling.
- Skip connections: concatenate encoder features with decoder features at matching resolutions.
- Activations: ReLU inside blocks for stable gradients.
- Normalization: BatchNorm after convolutions to stabilize training.
- Output layer: `1x1` convolution followed by Sigmoid to produce RGB values in `[0, 1]`.

## Preprocessing Explanation

- RGB conversion avoids inconsistent grayscale/RGBA channel counts.
- Resize to 256x256 gives manageable GPU/CPU memory use while preserving meaningful image structure.
- Tensor conversion scales pixel values from `0..255` to `0..1`.
- No ImageNet mean/std normalization is used because this is image reconstruction; the output must be directly comparable to the target image.

## Loss Function

The implemented loss is:

```text
0.85 * L1 + 0.15 * MSE
```

Why:

- L1 reduces blur and preserves sharper restoration boundaries.
- MSE penalizes larger color and brightness errors.
- The combination is stable, simple, and appropriate for supervised paired restoration.

## Optimizer

Adam is used with:

- learning rate: `1e-4`
- weight decay: `1e-5`

Adam is chosen because it adapts learning rates per parameter and is reliable for convolutional image-to-image models.
