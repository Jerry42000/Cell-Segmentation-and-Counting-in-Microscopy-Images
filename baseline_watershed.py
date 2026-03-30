#!/usr/bin/env python3

"""
Classical watershed baseline for binary cell segmentation.

Matured version:
- preserves the original baseline pipeline and arguments
- adds optional ZIP extraction for held-out test sets
- adds foreground-pixel error reporting to better support counting-oriented analysis
- saves richer summary output while staying backward compatible
- saves cleaner qualitative figures with instance boundaries and counts
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
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.measure import find_contours, label
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries, watershed


def parse_args():
    parser = argparse.ArgumentParser(description="Watershed baseline for cell segmentation")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing *_img.png and *_masks.png")
    parser.add_argument("--data_zip", type=str, default=None, help="Optional ZIP archive containing *_img.png and *_masks.png")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder to save metrics and figures")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of image-mask pairs")
    parser.add_argument("--invert_threshold", action="store_true", help="Use smoothed < otsu instead of smoothed > otsu")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur sigma")
    parser.add_argument("--min_size", type=int, default=64, help="Minimum object size to keep")
    parser.add_argument("--hole_area", type=int, default=64, help="Maximum hole area to fill")
    parser.add_argument("--peak_footprint", type=int, default=15, help="Peak-local-max footprint size for watershed seeds")
    parser.add_argument("--open_radius", type=int, default=2, help="Binary opening disk radius")
    parser.add_argument("--close_radius", type=int, default=2, help="Binary closing disk radius")
    parser.add_argument("--save_examples", type=int, default=3, help="How many example comparison figures to save")
    return parser.parse_args()


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


def to_binary_gt(mask):
    return (mask > 0).astype(np.uint8)


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
    coords = peak_local_max(distance, labels=binary, footprint=footprint, exclude_border=False)

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


def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 1.0


def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def count_instances(labels, min_size=24):
    cleaned = np.zeros_like(labels, dtype=np.int32)
    count = 0
    for lab in range(1, labels.max() + 1):
        region = labels == lab
        if region.sum() < min_size:
            continue
        count += 1
        cleaned[region] = count
    return cleaned, count


def save_example_figure(stem, image_rgb, gt_binary, pred_binary, pred_labels, out_path):
    img = image_rgb.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    cleaned_labels, count = count_instances(pred_labels)
    overlay = img.copy()
    boundaries = find_boundaries(cleaned_labels, mode="outer")
    overlay[boundaries] = np.array([1.0, 0.92, 0.15])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Raw image", fontsize=12, fontweight="bold")

    axes[0, 1].imshow(gt_binary, cmap="gray")
    axes[0, 1].set_title("Ground truth", fontsize=12, fontweight="bold")

    axes[0, 2].imshow(pred_binary, cmap="gray")
    axes[0, 2].set_title("Baseline binary prediction", fontsize=12, fontweight="bold")

    axes[1, 0].imshow(cleaned_labels, cmap="nipy_spectral")
    axes[1, 0].set_title(f"Separated instances ({count})", fontsize=12, fontweight="bold")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Traced boundaries", fontsize=12, fontweight="bold")

    axes[1, 2].imshow(img)
    for lab in range(1, cleaned_labels.max() + 1):
        contours = find_contours((cleaned_labels == lab).astype(float), 0.5)
        for contour in contours:
            axes[1, 2].plot(contour[:, 1], contour[:, 0], linewidth=0.85)
    axes[1, 2].set_title(f"Final overlay | Count = {count}", fontsize=12, fontweight="bold")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Sample {stem}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is None and args.data_zip is None:
        raise RuntimeError("Provide either --data_dir or --data_zip.")

    if args.data_zip is not None:
        data_dir = extract_zip_if_needed(Path(args.data_zip), out_dir)
    else:
        data_dir = Path(args.data_dir)

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
            cleaned_labels, count = count_instances(pred["pred_labels"])

            iou = compute_iou(pred_binary, gt_binary)
            dice = compute_dice(pred_binary, gt_binary)
            gt_pixels = int(gt_binary.sum())
            pred_pixels = int(pred_binary.sum())
            abs_pixel_error = abs(pred_pixels - gt_pixels)
            rel_pixel_error = abs_pixel_error / (gt_pixels + 1e-6)

            rows.append({
                "id": stem,
                "height": int(image_rgb.shape[0]),
                "width": int(image_rgb.shape[1]),
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "abs_foreground_pixel_error": abs_pixel_error,
                "rel_foreground_pixel_error": rel_pixel_error,
                "pred_instances": int(count),
                "threshold": pred["threshold"],
                "iou": iou,
                "dice": dice,
                "error": "",
            })

            if example_budget > 0:
                save_example_figure(stem, image_rgb, gt_binary, pred_binary, cleaned_labels, figures_dir / f"{stem}_comparison.png")
                example_budget -= 1

        except Exception as e:
            rows.append({
                "id": stem,
                "height": "",
                "width": "",
                "gt_pixels": "",
                "pred_pixels": "",
                "abs_foreground_pixel_error": "",
                "rel_foreground_pixel_error": "",
                "pred_instances": "",
                "threshold": "",
                "iou": "",
                "dice": "",
                "error": str(e),
            })

    valid_rows = [r for r in rows if isinstance(r.get("iou"), float)]
    summary = {
        "data_dir": str(data_dir),
        "num_pairs_found": len(pairs),
        "num_missing_masks": len(missing_masks),
        "num_successful": len(valid_rows),
        "num_failed": len(rows) - len(valid_rows),
        "mean_iou": float(np.mean([r["iou"] for r in valid_rows])) if valid_rows else None,
        "mean_dice": float(np.mean([r["dice"] for r in valid_rows])) if valid_rows else None,
        "mean_abs_foreground_pixel_error": float(np.mean([r["abs_foreground_pixel_error"] for r in valid_rows])) if valid_rows else None,
        "mean_rel_foreground_pixel_error": float(np.mean([r["rel_foreground_pixel_error"] for r in valid_rows])) if valid_rows else None,
        "mean_pred_instances": float(np.mean([r["pred_instances"] for r in valid_rows])) if valid_rows else None,
        "median_iou": float(np.median([r["iou"] for r in valid_rows])) if valid_rows else None,
        "median_dice": float(np.median([r["dice"] for r in valid_rows])) if valid_rows else None,
        "params": vars(args),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "height",
                "width",
                "gt_pixels",
                "pred_pixels",
                "abs_foreground_pixel_error",
                "rel_foreground_pixel_error",
                "pred_instances",
                "threshold",
                "iou",
                "dice",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'per_image_metrics.csv'}")
    print(f"Saved example figures in {figures_dir}")


if __name__ == "__main__":
    main()
