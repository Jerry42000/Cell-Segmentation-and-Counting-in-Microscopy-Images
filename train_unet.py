#!/usr/bin/env python3

"""
Matured training script for the reduced U-Net on binary cell segmentation.

This version keeps the same U-Net backbone used in the progress report, but
adds stronger training and evaluation features without changing the core model:
- optional data augmentation
- BCE + Dice hybrid loss
- optional boundary-aware weighting to preserve thin gaps between touching cells
- optional target erosion to discourage boundary overgrowth and merged cells
- optional soft boundary-consistency term to reward sharper inter-cell gaps
- optional checkpoint resume for short fine-tuning cycles
- learning-rate scheduling
- early stopping
- checkpoint with optimizer/scheduler state
- optional evaluation on an external test directory or a test ZIP archive
- richer saved outputs, including per-image metrics and qualitative figures

Backward compatibility:
- the original required arguments (--data_dir and --out_dir) still work
- existing filenames are preserved
- if no test set is provided, it behaves like a train/validation script
"""

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


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


def parse_args():
    p = argparse.ArgumentParser(description="Matured reduced U-Net training for binary cell segmentation")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing training/validation image-mask pairs")
    p.add_argument("--out_dir", type=str, required=True, help="Directory for checkpoints, metrics, and figures")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_augmentation", action="store_true", help="Apply lightweight geometric and intensity augmentation to the training set")
    p.add_argument("--rotation_degrees", type=float, default=8.0, help="Max absolute rotation used in augmentation")
    p.add_argument("--brightness_jitter", type=float, default=0.04, help="Brightness jitter magnitude around 1.0")
    p.add_argument("--contrast_jitter", type=float, default=0.04, help="Contrast jitter magnitude around 1.0")
    p.add_argument("--dice_weight", type=float, default=0.5, help="Weight on Dice loss in hybrid BCE+Dice objective")
    p.add_argument("--bce_weight", type=float, default=0.5, help="Weight on BCE loss in hybrid BCE+Dice objective")
    p.add_argument("--pos_weight", type=float, default=None, help="Optional positive-class weighting for BCEWithLogitsLoss")
    p.add_argument("--early_stopping_patience", type=int, default=4)
    p.add_argument("--scheduler_patience", type=int, default=2)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary prediction")
    p.add_argument("--save_examples", type=int, default=4, help="How many qualitative example figures to save for validation/test")
    p.add_argument("--test_dir", type=str, default=None, help="Optional directory containing held-out test image-mask pairs")
    p.add_argument("--test_zip", type=str, default=None, help="Optional ZIP archive containing held-out test image-mask pairs")
    p.add_argument("--run_test_after_training", action="store_true", help="Evaluate the best checkpoint on the provided test set")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Optional checkpoint path to resume from for short fine-tuning cycles")
    p.add_argument("--boundary_weight", type=float, default=3.0, help="Extra multiplicative weight applied to BCE on boundary-band pixels")
    p.add_argument("--boundary_band", type=int, default=2, help="Half-width of the boundary band in pixels at training resolution")
    p.add_argument("--boundary_term_weight", type=float, default=0.20, help="Weight on soft boundary-consistency term")
    p.add_argument("--erode_target_radius", type=int, default=1, help="Radius for mild target erosion used inside the loss")
    p.add_argument("--comparison_threshold_high", type=float, default=0.60, help="Higher threshold used only in saved comparison figures to highlight narrow black gaps")
    return p.parse_args()


def find_pairs(data_dir: Path):
    image_paths = sorted(data_dir.glob("*_img.png"))
    pairs = []
    for img_path in image_paths:
        stem = img_path.name.replace("_img.png", "")
        mask_path = data_dir / f"{stem}_masks.png"
        if mask_path.exists():
            pairs.append((stem, img_path, mask_path))
    return pairs


def extract_zip_if_needed(zip_path: Path, extract_root: Path):
    extract_dir = extract_root / f"extracted_{zip_path.stem}"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    image_hits = list(extract_dir.rglob("*_img.png"))
    if not image_hits:
        raise RuntimeError(f"No *_img.png files found after extracting {zip_path}")
    candidate_dirs = sorted({p.parent for p in image_hits})
    best_dir = max(candidate_dirs, key=lambda d: len(list(d.glob("*_img.png"))))
    return best_dir


class CellDataset(Dataset):
    def __init__(self, pairs, image_size=256, augment=False, rotation_degrees=8.0, brightness_jitter=0.04, contrast_jitter=0.04):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __len__(self):
        return len(self.pairs)

    def _maybe_augment(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        if self.brightness_jitter > 0:
            brightness = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            image = TF.adjust_brightness(image, brightness)
        if self.contrast_jitter > 0:
            contrast = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            image = TF.adjust_contrast(image, contrast)
        return image, mask

    def __getitem__(self, idx):
        stem, img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_arr = np.array(Image.open(mask_path))
        mask_arr = (mask_arr > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask_arr)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            image, mask = self._maybe_augment(image, mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return image, mask, stem


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def soft_erode(x, radius=1):
    if radius <= 0:
        return x
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel, stride=1, padding=radius)


def make_boundary_band(targets, band=2):
    if band <= 0:
        return torch.zeros_like(targets)
    kernel = 2 * band + 1
    dil = F.max_pool2d(targets, kernel_size=kernel, stride=1, padding=band)
    ero = soft_erode(targets, radius=band)
    boundary = (dil - ero).clamp(min=0.0, max=1.0)
    return (boundary > 0).float()


def soft_boundary_map_from_probs(probs, band=1):
    kernel = 2 * band + 1
    dil = F.max_pool2d(probs, kernel_size=kernel, stride=1, padding=band)
    ero = soft_erode(probs, radius=band)
    return (dil - ero).clamp(min=0.0, max=1.0)


class HybridBCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=None,
        boundary_weight=3.0,
        boundary_band=2,
        erode_target_radius=1,
        boundary_term_weight=0.20,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.boundary_weight = boundary_weight
        self.boundary_band = boundary_band
        self.erode_target_radius = erode_target_radius
        self.boundary_term_weight = boundary_term_weight
        self.dice = DiceLoss()

    def weighted_bce(self, logits, targets):
        per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        if self.boundary_weight > 1.0 and self.boundary_band > 0:
            boundary = make_boundary_band(targets, band=self.boundary_band)
            weights = 1.0 + (self.boundary_weight - 1.0) * boundary
            per_pixel = per_pixel * weights
        return per_pixel.mean()

    def forward(self, logits, targets):
        main_targets = soft_erode(targets, radius=self.erode_target_radius) if self.erode_target_radius > 0 else targets
        bce_term = self.weighted_bce(logits, main_targets)
        dice_term = self.dice(logits, main_targets)

        total = self.bce_weight * bce_term + self.dice_weight * dice_term

        if self.boundary_term_weight > 0:
            probs = torch.sigmoid(logits)
            pred_boundary = soft_boundary_map_from_probs(probs, band=max(1, self.boundary_band // 2))
            true_boundary = make_boundary_band(targets, band=self.boundary_band)
            boundary_term = F.binary_cross_entropy(pred_boundary, true_boundary)
            total = total + self.boundary_term_weight * boundary_term

        return total


def dice_from_probs(preds, targets, eps=1e-6):
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()


def iou_from_probs(preds, targets, eps=1e-6):
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def pixel_accuracy_from_probs(preds, targets):
    return (preds == targets).float().mean().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_prediction_overlay(image, pred_mask, color=(0.95, 0.15, 0.15), alpha=0.32):
    overlay = image.copy()
    tint = np.zeros_like(overlay)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    mask = pred_mask > 0.5
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * tint[mask]
    return overlay


def save_comparison_figure(image, gt, pred_main, pred_high, prob, title, out_path):
    overlay_high = make_prediction_overlay(image, pred_high, color=(1.0, 0.55, 0.0), alpha=0.28)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2))
    panels = [
        (image, "Raw image", None),
        (gt, "Ground truth", "gray"),
        (prob, "Probability map", "viridis"),
        (pred_main, "Prediction @ main threshold", "gray"),
        (pred_high, "Prediction @ higher threshold", "gray"),
        (overlay_high, "Higher-threshold overlay", None),
    ]
    for ax, (img, title_txt, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title_txt, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_loader(
    model,
    loader,
    device,
    threshold=0.5,
    comparison_threshold_high=0.60,
    save_dir=None,
    max_examples=0,
):
    model.eval()
    metrics = []
    examples_saved = 0
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, stems in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            preds_high = (probs > comparison_threshold_high).float()

            batch_size = images.size(0)
            for i in range(batch_size):
                pred_i = preds[i:i + 1]
                mask_i = masks[i:i + 1]
                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                gt_np = masks[i, 0].cpu().numpy()
                prob_np = probs[i, 0].cpu().numpy()
                pred_np = preds[i, 0].cpu().numpy()
                pred_high_np = preds_high[i, 0].cpu().numpy()

                inter = float((pred_i * mask_i).sum().item())
                union = float(((pred_i + mask_i) > 0).float().sum().item())
                pred_pixels = float(pred_i.sum().item())
                gt_pixels = float(mask_i.sum().item())
                dice = (2 * inter + 1e-6) / (pred_pixels + gt_pixels + 1e-6)
                iou = (inter + 1e-6) / (union + 1e-6)
                acc = float((pred_i == mask_i).float().mean().item())
                abs_count_error = abs(pred_pixels - gt_pixels)
                rel_count_error = abs_count_error / (gt_pixels + 1e-6)

                metrics.append({
                    "id": stems[i],
                    "iou": iou,
                    "dice": dice,
                    "pixel_accuracy": acc,
                    "gt_foreground_pixels": gt_pixels,
                    "pred_foreground_pixels": pred_pixels,
                    "abs_foreground_pixel_error": abs_count_error,
                    "rel_foreground_pixel_error": rel_count_error,
                })

                if save_dir is not None and examples_saved < max_examples:
                    save_comparison_figure(
                        image_np,
                        gt_np,
                        pred_np,
                        pred_high_np,
                        prob_np,
                        f"Sample {stems[i]}",
                        save_dir / f"{stems[i]}_comparison.png",
                    )
                    examples_saved += 1

    if not metrics:
        raise RuntimeError("No evaluation samples were processed.")

    summary = {
        "num_samples": len(metrics),
        "mean_iou": float(np.mean([m["iou"] for m in metrics])),
        "mean_dice": float(np.mean([m["dice"] for m in metrics])),
        "mean_pixel_accuracy": float(np.mean([m["pixel_accuracy"] for m in metrics])),
        "mean_abs_foreground_pixel_error": float(np.mean([m["abs_foreground_pixel_error"] for m in metrics])),
        "mean_rel_foreground_pixel_error": float(np.mean([m["rel_foreground_pixel_error"] for m in metrics])),
        "median_iou": float(np.median([m["iou"] for m in metrics])),
        "median_dice": float(np.median([m["dice"] for m in metrics])),
    }
    return summary, metrics


def write_metrics_csv(rows, out_path):
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_weights(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return state
    model.load_state_dict(state)
    return None


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    pairs = find_pairs(data_dir)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    if len(pairs) < 4:
        raise RuntimeError("Need at least 4 valid image-mask pairs in --data_dir.")

    full_dataset = CellDataset(pairs, image_size=args.image_size, augment=False)
    val_len = max(1, int(len(full_dataset) * args.val_fraction))
    train_len = len(full_dataset) - val_len
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_len, val_len], generator=torch.Generator().manual_seed(args.seed)
    )

    train_pairs = [pairs[i] for i in train_indices.indices]
    val_pairs = [pairs[i] for i in val_indices.indices]

    train_set = CellDataset(
        train_pairs,
        image_size=args.image_size,
        augment=args.use_augmentation,
        rotation_degrees=args.rotation_degrees,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
    )
    val_set = CellDataset(val_pairs, image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)

    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)

    criterion = HybridBCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=pos_weight,
        boundary_weight=args.boundary_weight,
        boundary_band=args.boundary_band,
        erode_target_radius=args.erode_target_radius,
        boundary_term_weight=args.boundary_term_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    history = []
    best_val_dice = -1.0
    epochs_without_improvement = 0
    best_path = out_dir / "best_unet.pt"
    best_checkpoint_path = out_dir / "best_checkpoint.pt"
    start_epoch = 1

    best_source_checkpoint_path = best_checkpoint_path
    best_source_weights_path = best_path

    if args.resume_checkpoint is not None:
        resume_checkpoint_path = Path(args.resume_checkpoint)
        state = torch.load(resume_checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            best_val_dice = float(state.get("best_val_dice", -1.0))
            start_epoch = int(state.get("epoch", 0)) + 1

            torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "epoch": int(state.get("epoch", start_epoch - 1)),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "args": state.get("args", vars(args)),
                    "resumed_from": str(resume_checkpoint_path),
                },
                best_checkpoint_path,
            )
            print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}.")
        else:
            model.load_state_dict(state)
            torch.save(model.state_dict(), best_path)
            best_source_weights_path = resume_checkpoint_path
            print(f"Loaded plain weights from {args.resume_checkpoint}.")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logical_epoch = epoch - start_epoch + 1
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
        val_ious = []
        val_dices = []
        val_accs = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                probs = torch.sigmoid(logits)
                preds = (probs > args.threshold).float()

                val_loss_sum += loss.item()
                val_batches += 1
                val_ious.append(iou_from_probs(preds, masks))
                val_dices.append(dice_from_probs(preds, masks))
                val_accs.append(pixel_accuracy_from_probs(preds, masks))

        val_loss = val_loss_sum / max(1, val_batches)
        val_iou = float(np.mean(val_ious))
        val_dice = float(np.mean(val_dices))
        val_acc = float(np.mean(val_accs))
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "logical_epoch": logical_epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "val_pixel_accuracy": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {logical_epoch}/{args.epochs} | global_epoch={epoch} | lr={current_lr:.6g} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | val_dice={val_dice:.4f} | val_acc={val_acc:.4f}"
        )

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    write_metrics_csv(history, out_dir / "history.csv")

    best_epoch_row = max(history, key=lambda x: x["val_dice"])
    summary = {
        "num_pairs_used": len(pairs),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "image_size": args.image_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate_initial": args.lr,
        "device": str(device),
        "model": "UNetSmall",
        "trainable_parameters": count_parameters(model),
        "criterion": "HybridBCEDiceLoss",
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "boundary_weight": args.boundary_weight,
        "boundary_band": args.boundary_band,
        "boundary_term_weight": args.boundary_term_weight,
        "erode_target_radius": args.erode_target_radius,
        "threshold": args.threshold,
        "comparison_threshold_high": args.comparison_threshold_high,
        "augmentation_used": bool(args.use_augmentation),
        "best_val_dice": float(best_val_dice),
        "best_epoch": int(best_epoch_row["epoch"]),
        "best_val_iou": float(best_epoch_row["val_iou"]),
        "best_val_pixel_accuracy": float(best_epoch_row["val_pixel_accuracy"]),
        "best_weights": str(best_path if best_path.exists() else best_source_weights_path),
        "best_checkpoint": str(best_checkpoint_path if best_checkpoint_path.exists() else best_source_checkpoint_path),
        "resume_checkpoint": args.resume_checkpoint,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_checkpoint_path.exists():
        best_source_checkpoint_path = best_checkpoint_path
    elif args.resume_checkpoint is not None and Path(args.resume_checkpoint).exists():
        best_source_checkpoint_path = Path(args.resume_checkpoint)
    elif best_path.exists():
        best_source_checkpoint_path = best_path
    else:
        raise FileNotFoundError(
            f"No best model artifact found in {out_dir}. Expected {best_checkpoint_path} or {best_path}."
        )

    load_checkpoint_weights(model, best_source_checkpoint_path, device)
    val_summary, val_rows = evaluate_loader(
        model,
        val_loader,
        device,
        threshold=args.threshold,
        comparison_threshold_high=args.comparison_threshold_high,
        save_dir=figures_dir / "validation_examples",
        max_examples=args.save_examples,
    )
    with open(out_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)
    write_metrics_csv(val_rows, out_dir / "validation_per_image_metrics.csv")

    test_summary = None
    test_rows = None
    resolved_test_dir = None
    if args.test_dir or args.test_zip:
        if args.test_dir:
            resolved_test_dir = Path(args.test_dir)
        else:
            resolved_test_dir = extract_zip_if_needed(Path(args.test_zip), out_dir)

        test_pairs = find_pairs(resolved_test_dir)
        if not test_pairs:
            raise RuntimeError(f"No valid test image-mask pairs found in {resolved_test_dir}")

        test_set = CellDataset(test_pairs, image_size=args.image_size, augment=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        if args.run_test_after_training:
            test_summary, test_rows = evaluate_loader(
                model,
                test_loader,
                device,
                threshold=args.threshold,
                comparison_threshold_high=args.comparison_threshold_high,
                save_dir=figures_dir / "test_examples",
                max_examples=args.save_examples,
            )
            with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
                json.dump(test_summary, f, indent=2)
            write_metrics_csv(test_rows, out_dir / "test_per_image_metrics.csv")

    epochs = [r["logical_epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["val_loss"] for r in history]
    val_dices = [r["val_dice"] for r in history]
    val_ious = [r["val_iou"] for r in history]

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss")
    ax.plot(epochs, val_dices, label="Val Dice")
    ax.plot(epochs, val_ious, label="Val IoU")
    ax.set_xlabel("Fine-tuning epoch" if args.resume_checkpoint else "Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Done. Outputs saved in: {out_dir}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model weights: {best_path}")
    if resolved_test_dir is not None:
        print(f"Resolved test directory: {resolved_test_dir}")
    if test_summary is not None:
        print(
            f"Test results | mean_iou={test_summary['mean_iou']:.4f} | "
            f"mean_dice={test_summary['mean_dice']:.4f} | "
            f"mean_pixel_accuracy={test_summary['mean_pixel_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
