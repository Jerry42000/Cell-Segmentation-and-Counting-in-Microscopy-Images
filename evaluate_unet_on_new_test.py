#!/usr/bin/env python3
"""
Evaluate an already-trained U-Net checkpoint on a different held-out test set,
without changing the original project files.

This script reuses the same model family and metric definitions as the project's
training/prediction scripts, but runs in a standalone way on a new test folder
or ZIP. It can also generate one polished qualitative figure in the same 2x3
style used by predict_unet_single.py.
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import find_contours, label
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries, watershed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


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
    p = argparse.ArgumentParser(description="Evaluate trained U-Net on a new held-out test set")
    p.add_argument("--weights", required=True, type=str, help="Path to best_checkpoint.pt or best_unet.pt")
    p.add_argument("--test_dir", type=str, default=None, help="Folder containing *_img.png and *_masks.png")
    p.add_argument("--test_zip", type=str, default=None, help="ZIP containing *_img.png and *_masks.png")
    p.add_argument("--out_dir", required=True, type=str, help="Folder to save summary, per-image metrics, and figures")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--comparison_threshold_high", type=float, default=0.60)
    p.add_argument("--save_examples", type=int, default=4)
    p.add_argument("--sample_id", type=str, default=None, help="Optional sample ID for one polished 2x3 final figure")
    p.add_argument("--peak_footprint", type=int, default=17)
    p.add_argument("--min_size", type=int, default=110)
    p.add_argument("--hole_area", type=int, default=100)
    p.add_argument("--open_radius", type=int, default=1)
    p.add_argument("--close_radius", type=int, default=1)
    return p.parse_args()


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
    return max(candidate_dirs, key=lambda d: len(list(d.glob("*_img.png"))))


def find_pairs(data_dir: Path):
    image_paths = sorted(data_dir.glob("*_img.png"))
    pairs = []
    for img_path in image_paths:
        stem = img_path.name.replace("_img.png", "")
        mask_path = data_dir / f"{stem}_masks.png"
        if mask_path.exists():
            pairs.append((stem, img_path, mask_path))
    return pairs


class CellDataset(Dataset):
    def __init__(self, pairs, image_size=512):
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


def load_checkpoint(model, weights_path: Path, device):
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def make_prediction_overlay(image, pred_mask, color=(0.95, 0.15, 0.15), alpha=0.32):
    overlay = image.copy()
    tint = np.zeros_like(overlay)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
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
        ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
        ax.set_title(title_txt, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def iou_and_dice(pred_i, mask_i):
    inter = float((pred_i * mask_i).sum().item())
    union = float(((pred_i + mask_i) > 0).float().sum().item())
    pred_pixels = float(pred_i.sum().item())
    gt_pixels = float(mask_i.sum().item())
    dice = (2 * inter + 1e-6) / (pred_pixels + gt_pixels + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou, dice, pred_pixels, gt_pixels


def evaluate_loader(model, loader, device, threshold, comparison_threshold_high, save_dir=None, max_examples=0):
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

            for i in range(images.size(0)):
                pred_i = preds[i:i + 1]
                mask_i = masks[i:i + 1]
                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                gt_np = masks[i, 0].cpu().numpy()
                prob_np = probs[i, 0].cpu().numpy()
                pred_np = preds[i, 0].cpu().numpy()
                pred_high_np = preds_high[i, 0].cpu().numpy()

                iou, dice, pred_pixels, gt_pixels = iou_and_dice(pred_i, mask_i)
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
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def postprocess_binary(binary, min_size=80, hole_area=80, open_radius=1, close_radius=1):
    binary = remove_small_objects(binary.astype(bool), min_size=min_size)
    binary = remove_small_holes(binary, area_threshold=hole_area)
    if open_radius > 0:
        binary = binary_opening(binary, disk(open_radius))
    if close_radius > 0:
        binary = binary_closing(binary, disk(close_radius))
    return binary.astype(np.uint8)


def watershed_from_probability(prob_map, threshold=0.5, peak_footprint=17, min_size=110, hole_area=100, open_radius=1, close_radius=1):
    binary = postprocess_binary(prob_map > threshold, min_size=min_size, hole_area=hole_area, open_radius=open_radius, close_radius=close_radius)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary)
    footprint = np.ones((peak_footprint, peak_footprint), dtype=bool)
    coords = peak_local_max(distance, labels=binary, footprint=footprint, exclude_border=False)
    markers = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    if markers.max() == 0:
        labels = label(binary)
    else:
        markers = label(markers > 0)
        labels = watershed(-distance, markers, mask=binary)
    return labels


def count_instances(labels, min_size=40):
    cleaned = np.zeros_like(labels, dtype=np.int32)
    count = 0
    for lab in range(1, labels.max() + 1):
        region = labels == lab
        if region.sum() < min_size:
            continue
        count += 1
        cleaned[region] = count
    return cleaned, count


def boundary_overlay(image_np, labels):
    canvas = np.clip(image_np.copy(), 0, 1)
    boundaries = find_boundaries(labels, mode="outer")
    canvas[boundaries] = np.array([1.0, 0.92, 0.15])
    return canvas


def contour_overlay(ax, labels, linewidth=1.0):
    for lab in range(1, labels.max() + 1):
        region = labels == lab
        if not np.any(region):
            continue
        contours = find_contours(region.astype(float), 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth)


def save_final_panel(model, device, pair, out_path, image_size, threshold, peak_footprint, min_size, hole_area, open_radius, close_radius):
    stem, img_path, mask_path = pair
    image_pil = Image.open(img_path).convert("RGB")
    mask_arr = np.array(Image.open(mask_path))
    gt_binary_arr = (mask_arr > 0).astype(np.uint8) * 255
    gt_pil = Image.fromarray(gt_binary_arr)

    image_resized = TF.resize(image_pil, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    gt_resized = TF.resize(gt_pil, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
    image_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    gt_np = (TF.to_tensor(gt_resized)[0] > 0.5).float().numpy()

    with torch.no_grad():
        prob = torch.sigmoid(model(image_tensor))[0, 0].cpu().numpy()
    raw_binary = (prob > threshold).astype(np.uint8)
    labels = watershed_from_probability(prob, threshold=threshold, peak_footprint=peak_footprint, min_size=min_size, hole_area=hole_area, open_radius=open_radius, close_radius=close_radius)
    labels, count = count_instances(labels, min_size=max(20, min_size // 2))
    traced_overlay = boundary_overlay(image_np, labels)
    raw_overlay = image_np.copy()
    raw_overlay[raw_binary > 0] = 0.72 * raw_overlay[raw_binary > 0] + 0.28 * np.array([1.0, 0.2, 0.2])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    fig.patch.set_facecolor("white")
    axes[0, 0].imshow(image_np); axes[0, 0].set_title("Raw image", fontsize=13, fontweight="bold")
    axes[0, 1].imshow(gt_np, cmap="gray"); axes[0, 1].set_title("Ground truth", fontsize=13, fontweight="bold")
    im = axes[0, 2].imshow(prob, cmap="viridis"); axes[0, 2].set_title("U-Net probability map", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04); cbar.ax.tick_params(labelsize=9)
    axes[1, 0].imshow(raw_overlay); axes[1, 0].set_title("Raw binary overlay", fontsize=13, fontweight="bold")
    axes[1, 1].imshow(labels, cmap="nipy_spectral"); axes[1, 1].set_title(f"Separated instances ({count})", fontsize=13, fontweight="bold")
    axes[1, 2].imshow(traced_overlay); contour_overlay(axes[1, 2], labels, linewidth=0.85)
    axes[1, 2].set_title(f"Final traced boundaries | Count = {count}", fontsize=13, fontweight="bold")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Sample {stem}: segmentation, separation, and counting", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if (args.test_dir is None) == (args.test_zip is None):
        raise RuntimeError("Provide exactly one of --test_dir or --test_zip.")

    if args.test_zip:
        data_dir = extract_zip_if_needed(Path(args.test_zip), out_dir)
    else:
        data_dir = Path(args.test_dir)

    pairs = find_pairs(data_dir)
    if not pairs:
        raise RuntimeError(f"No valid image-mask pairs found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall().to(device)
    load_checkpoint(model, Path(args.weights), device)
    model.eval()

    dataset = CellDataset(pairs, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    summary, rows = evaluate_loader(
        model,
        loader,
        device,
        threshold=args.threshold,
        comparison_threshold_high=args.comparison_threshold_high,
        save_dir=figures_dir / "test_examples",
        max_examples=args.save_examples,
    )

    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_metrics_csv(rows, out_dir / "test_per_image_metrics.csv")

    if args.sample_id is not None:
        pair_lookup = {stem: triplet for stem, *triplet in pairs}
        if args.sample_id not in pair_lookup:
            raise FileNotFoundError(f"sample_id {args.sample_id} not found in {data_dir}")
        stem = args.sample_id
        img_path, mask_path = pair_lookup[stem]
        save_final_panel(
            model,
            device,
            (stem, img_path, mask_path),
            out_dir / f"{stem}_final_panel.png",
            image_size=args.image_size,
            threshold=args.threshold,
            peak_footprint=args.peak_footprint,
            min_size=args.min_size,
            hole_area=args.hole_area,
            open_radius=args.open_radius,
            close_radius=args.close_radius,
        )

    print(f"Wrote {out_dir / 'test_summary.json'}")
    print(f"Wrote {out_dir / 'test_per_image_metrics.csv'}")
    print(f"Saved example figures in {figures_dir / 'test_examples'}")
    print(
        f"Test results | mean_iou={summary['mean_iou']:.4f} | "
        f"mean_dice={summary['mean_dice']:.4f} | "
        f"mean_pixel_accuracy={summary['mean_pixel_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
