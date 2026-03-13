#!/usr/bin/env python3

"""
Classical watershed baseline for binary cell segmentation.

This script:
- Reads paired microscopy images and masks named *_img.png and *_masks.png
- Converts instance masks to binary ground-truth masks
- Applies a simple classical image processing technique:
    grayscale -> Gaussian smoothing -> Otsu thresholding -> morphology
    -> distance transform -> seed extraction -> watershed
- Evaluates the predicted binary mask using IoU and Dice score
- Saves:
    1) summary.json with overall metrics
    2) per_image_metrics.csv with per-image results
    3) a few example comparison figures

This is meant to provide a simple, reproducible non-neural baseline for comparison
with the U-Net model in the progress report.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import watershed


# Parse command-line arguments controlling input paths and baseline parameters.
def parse_args():
    parser = argparse.ArgumentParser(description="Watershed baseline for cell segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing *_img.png and *_masks.png")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder to save metrics and figures")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of image-mask pairs")
    parser.add_argument("--invert_threshold", action="store_true",
                        help="Use smoothed < otsu instead of smoothed > otsu")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur sigma")
    parser.add_argument("--min_size", type=int, default=64, help="Minimum object size to keep")
    parser.add_argument("--hole_area", type=int, default=64, help="Maximum hole area to fill")
    parser.add_argument("--peak_footprint", type=int, default=15,
                        help="Peak-local-max footprint size for watershed seeds")
    parser.add_argument("--open_radius", type=int, default=2, help="Binary opening disk radius")
    parser.add_argument("--close_radius", type=int, default=2, help="Binary closing disk radius")
    parser.add_argument("--save_examples", type=int, default=3,
                        help="How many example comparison figures to save")
    return parser.parse_args()

# Match each image file with its corresponding mask file.
def find_pairs(data_dir: Path):
    image_files = sorted(data_dir.glob("*_img.png"))
    pairs = []
    missing_masks = []
    for img_path in image_files:
        stem = img_path.name.replace("_img.png", "")
        mask_path = data_dir / f"{stem}_masks.png"
        if mask_path.exists():
            pairs.append((stem, img_path, mask_path))
        else:
            missing_masks.append(img_path.name)
    return pairs, missing_masks


def load_image(path: Path):
    return np.array(Image.open(path))


def load_mask(path: Path):
    return np.array(Image.open(path))

# Convert instance mask to binary foreground/background ground truth.
def to_binary_gt(mask):
    return (mask > 0).astype(np.uint8)

# Run the full classical baseline pipeline and return the prediction plus metadata.
def baseline_predict(image_rgb, args):
    gray = rgb2gray(image_rgb)
    smoothed = gaussian(gray, sigma=args.sigma, preserve_range=True)

    thresh = threshold_otsu(smoothed)
    binary = smoothed < thresh if args.invert_threshold else smoothed > thresh

    binary = remove_small_objects(binary, min_size=args.min_size)
    binary = remove_small_holes(binary, area_threshold=args.hole_area)

    if args.open_radius > 0:
        binary = binary_opening(binary, disk(args.open_radius))
    if args.close_radius > 0:
        binary = binary_closing(binary, disk(args.close_radius))

    distance = ndi.distance_transform_edt(binary)

    footprint = np.ones((args.peak_footprint, args.peak_footprint), dtype=bool)
    coords = peak_local_max(
        distance,
        labels=binary,
        footprint=footprint,
        exclude_border=False,
    )

    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        pred_labels = np.zeros_like(markers)
    else:
        markers = label(markers > 0)
        pred_labels = watershed(-distance, markers, mask=binary)

    pred_binary = (pred_labels > 0).astype(np.uint8)
    return {
        "threshold": float(thresh),
        "pred_labels": pred_labels,
        "pred_binary": pred_binary,
        "num_instances": int(pred_labels.max()),
    }

# Pixelwise Intersection over Union for binary masks.
def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 1.0

# Pixelwise Dice score for binary masks.
def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0

# Overlay the predicted binary mask on the original RGB image for visualization.
def make_overlay(image_rgb, mask_binary, alpha=0.35):
    img = image_rgb.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    overlay = img.copy()
    red = np.zeros_like(img)
    red[..., 0] = 1.0
    mask = mask_binary.astype(bool)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * red[mask]
    return np.clip(overlay, 0, 1)

# Save a 4-panel figure showing raw image, ground truth, prediction, and overlay.
def save_example_figure(stem, image_rgb, gt_binary, pred_binary, out_path):
    overlay = make_overlay(image_rgb, pred_binary)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Raw image")
    axes[1].imshow(gt_binary, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_binary, cmap="gray")
    axes[2].set_title("Baseline prediction")
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Sample {stem}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Main evaluation loop: run the baseline on all valid pairs and save outputs.
def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing_masks = find_pairs(data_dir)
    if args.max_images is not None:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise RuntimeError(f"No valid image-mask pairs found in {data_dir}")

    rows = []
    example_budget = args.save_examples

    for stem, img_path, mask_path in pairs:
        try:
            image_rgb = load_image(img_path)
            mask = load_mask(mask_path)

            if image_rgb.ndim != 3 or image_rgb.shape[2] not in (3, 4):
                raise ValueError(f"Expected RGB/RGBA image, got shape {image_rgb.shape}")
            if image_rgb.shape[2] == 4:
                image_rgb = image_rgb[..., :3]
            if image_rgb.shape[:2] != mask.shape[:2]:
                raise ValueError("Image and mask shapes do not match")

            gt_binary = to_binary_gt(mask)
            pred = baseline_predict(image_rgb, args)
            pred_binary = pred["pred_binary"]

            iou = compute_iou(pred_binary, gt_binary)
            dice = compute_dice(pred_binary, gt_binary)

            rows.append({
                "id": stem,
                "height": int(image_rgb.shape[0]),
                "width": int(image_rgb.shape[1]),
                "gt_pixels": int(gt_binary.sum()),
                "pred_pixels": int(pred_binary.sum()),
                "pred_instances": int(pred["num_instances"]),
                "threshold": pred["threshold"],
                "iou": iou,
                "dice": dice,
                "error": "",
            })

            if example_budget > 0:
                save_example_figure(
                    stem, image_rgb, gt_binary, pred_binary, figures_dir / f"{stem}_comparison.png"
                )
                example_budget -= 1

        except Exception as e:
            rows.append({
                "id": stem, "height": "", "width": "", "gt_pixels": "", "pred_pixels": "",
                "pred_instances": "", "threshold": "", "iou": "", "dice": "", "error": str(e),
            })

    valid_rows = [r for r in rows if isinstance(r.get("iou"), float)]
    summary = {
        "num_pairs_found": len(pairs),
        "num_missing_masks": len(missing_masks),
        "num_successful": len(valid_rows),
        "num_failed": len(rows) - len(valid_rows),
        "mean_iou": float(np.mean([r["iou"] for r in valid_rows])) if valid_rows else None,
        "mean_dice": float(np.mean([r["dice"] for r in valid_rows])) if valid_rows else None,
        "median_iou": float(np.median([r["iou"] for r in valid_rows])) if valid_rows else None,
        "median_dice": float(np.median([r["dice"] for r in valid_rows])) if valid_rows else None,
        "params": vars(args),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "height", "width", "gt_pixels", "pred_pixels", "pred_instances",
                        "threshold", "iou", "dice", "error"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'per_image_metrics.csv'}")
    print(f"Saved example figures in {figures_dir}")


if __name__ == "__main__":
    main()
