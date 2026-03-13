#!/usr/bin/env python3

"""
Quick training script for a reduced U-Net on binary cell segmentation.

The script:
- Reads paired microscopy images and masks named *_img.png and *_masks.png
- Converts instance masks to binary foreground/background masks
- Resizes all samples to a fixed square input size
- Splits the dataset into train and validation subsets
- Trains a small U-Net for a short preliminary run
- Evaluates validation IoU and Dice after each epoch
- Saves:
    1) best model weights
    2) training history CSV
    3) summary.json with key settings and best metrics
    4) one qualitative prediction figure
    5) a learning-curve plot

This is intended as a lightweight feasibility experiment for the progress report,
not a full final training model.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# Set random seeds for reproducible dataset splitting and training behavior.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Parse training configuration and file-path arguments.
def parse_args():
    p = argparse.ArgumentParser(description="Quick U-Net training for binary cell segmentation")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Match each microscopy image with its corresponding mask file.
def find_pairs(data_dir: Path):
    image_paths = sorted(data_dir.glob("*_img.png"))
    pairs = []
    for img_path in image_paths:
        stem = img_path.name.replace("_img.png", "")
        mask_path = data_dir / f"{stem}_masks.png"
        if mask_path.exists():
            pairs.append((stem, img_path, mask_path))
    return pairs


# Dataset wrapper that loads images, binarizes masks, and resizes both for training.
class CellDataset(Dataset):
    def __init__(self, pairs, image_size=256):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_arr = np.array(Image.open(mask_path))
        mask_arr = (mask_arr > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask_arr)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask, stem


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


# Reduced U-Net architecture used for the progress report.
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


# Compute binary Dice score from model logits.
def dice_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()


# Compute binary IoU from model logits.
def iou_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


# Count trainable parameters for reporting model complexity.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Save one qualitative validation example from the trained model.
def save_prediction_figure(model, loader, device, out_path):
    model.eval()
    with torch.no_grad():
        for images, masks, stems in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            image = images[0].cpu().permute(1, 2, 0).numpy()
            gt = masks[0, 0].cpu().numpy()
            pred = preds[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
            axes[0].imshow(image)
            axes[0].set_title("Raw image")
            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("Ground truth")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("U-Net prediction")

            overlay = image.copy()
            red = np.zeros_like(overlay)
            red[..., 0] = 1.0
            m = pred > 0.5
            overlay[m] = 0.65 * overlay[m] + 0.35 * red[m]
            axes[3].imshow(overlay)
            axes[3].set_title("Prediction overlay")

            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"Validation sample {stems[0]}", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            break


# Main training loop: prepare data, train the U-Net, evaluate, and save outputs.
def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(data_dir)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    if len(pairs) < 4:
        raise RuntimeError("Need at least 4 valid image-mask pairs.")

    dataset = CellDataset(pairs, image_size=args.image_size)
    val_len = max(1, int(len(dataset) * args.val_fraction))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_dice = -1.0
    best_path = out_dir / "best_unet.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)

                val_loss_sum += loss.item()
                val_batches += 1
                iou_scores.append(iou_from_logits(logits, masks))
                dice_scores.append(dice_from_logits(logits, masks))

        val_loss = val_loss_sum / max(1, val_batches)
        val_iou = float(np.mean(iou_scores))
        val_dice = float(np.mean(dice_scores))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_path)

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_iou", "val_dice"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    summary = {
        "num_pairs_used": len(dataset),
        "train_size": train_len,
        "val_size": val_len,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "model": "UNetSmall",
        "trainable_parameters": count_parameters(model),
        "best_val_dice": best_val_dice,
        "best_epoch": int(max(history, key=lambda x: x["val_dice"])["epoch"]),
        "best_val_iou": float(max(history, key=lambda x: x["val_dice"])["val_iou"]),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    save_prediction_figure(model, val_loader, device, out_dir / "prediction_example.png")

    epochs = [r["epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["val_loss"] for r in history]
    val_dice = [r["val_dice"] for r in history]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss")
    ax.plot(epochs, val_dice, label="Val Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Done. Outputs saved in: {out_dir}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model weights: {best_path}")


if __name__ == "__main__":
    main()