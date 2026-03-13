
#!/usr/bin/env python3
"""
Run a trained reduced U-Net model on one specified image-mask pair.

This script:
- Loads a saved U-Net checkpoint
- Loads one chosen sample by sample ID
- Resizes the image and mask to the model input size
- Converts the instance mask to a binary ground-truth mask
- Runs inference with the trained model
- Saves a 4-panel figure:
    raw image, ground truth, U-Net prediction, and prediction overlay

This is mainly used to generate a qualitative example for the report using
a specific sample chosen by the user.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

# Basic double-convolution block used throughout the U-Net.
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# Downsampling block: max-pooling followed by double convolution.
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

# Upsampling block: transposed convolution, skip connection, then double convolution.
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = TF.pad(x, [diff_x // 2, diff_y // 2, diff_x - diff_x // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# Reduced U-Net architecture used in the progress report.
class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, feats=(32, 64, 128, 256)):
        super().__init__()
        self.inc = DoubleConv(in_ch, feats[0])
        self.down1 = Down(feats[0], feats[1])
        self.down2 = Down(feats[1], feats[2])
        self.down3 = Down(feats[2], feats[3])
        self.up1 = Up(feats[3], feats[2], feats[2])
        self.up2 = Up(feats[2], feats[1], feats[1])
        self.up3 = Up(feats[1], feats[0], feats[0])
        self.outc = nn.Conv2d(feats[0], out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# Parse input paths and sample selection arguments.
def parse_args():
    p = argparse.ArgumentParser(description="Run trained U-Net on one specific sample")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--sample_id", required=True, type=str, help="Example: 001")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--out_path", required=True, type=str)
    return p.parse_args()

# Load one sample, resize it, and prepare both the image tensor and binary ground truth.
def load_sample(data_dir: Path, sample_id: str, image_size: int):
    img_path = data_dir / f"{sample_id}_img.png"
    mask_path = data_dir / f"{sample_id}_masks.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    image_pil = Image.open(img_path).convert("RGB")
    mask_arr = np.array(Image.open(mask_path))
    gt_binary_arr = (mask_arr > 0).astype(np.uint8) * 255
    gt_pil = Image.fromarray(gt_binary_arr)

    image_resized = TF.resize(image_pil, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    gt_resized = TF.resize(gt_pil, [image_size, image_size], interpolation=InterpolationMode.NEAREST)

    image_tensor = TF.to_tensor(image_resized).unsqueeze(0)
    gt_tensor = TF.to_tensor(gt_resized)
    gt_tensor = (gt_tensor > 0.5).float()

    image_np = image_tensor[0].permute(1, 2, 0).numpy()
    gt_np = gt_tensor[0].numpy()
    return image_tensor, image_np, gt_np

# Main inference function: load model, predict on one sample, and save a visualization.
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetSmall().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image_tensor, image_np, gt_np = load_sample(Path(args.data_dir), args.sample_id, args.image_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        pred = (torch.sigmoid(logits) > 0.5).float()[0, 0].cpu().numpy()

    overlay = image_np.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    m = pred > 0.5
    overlay[m] = 0.65 * overlay[m] + 0.35 * red[m]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    axes[0].imshow(image_np)
    axes[0].set_title("Raw image")
    axes[1].imshow(gt_np, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("U-Net prediction")
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Sample {args.sample_id}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
