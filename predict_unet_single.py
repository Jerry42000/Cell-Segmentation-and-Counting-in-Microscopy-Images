#!/usr/bin/env python3
"""
Run a trained reduced U-Net model on one specified image-mask pair.

Matured features added while preserving the original use case:
- can read either plain state_dict weights or a richer checkpoint dict
- optional watershed-style instance separation from the probability map
- optional nucleus-marker detection path kept available but disabled by default
- saves a more polished multi-panel figure with traced boundaries and count
- configurable threshold
"""
import argparse
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
    p = argparse.ArgumentParser(description="Run trained U-Net on one specific sample")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--sample_id", required=True, type=str, help="Example: 001")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--use_watershed", action="store_true", help="Apply watershed separation to the predicted probability map")
    p.add_argument("--peak_footprint", type=int, default=17)
    p.add_argument("--min_size", type=int, default=110)
    p.add_argument("--hole_area", type=int, default=100)
    p.add_argument("--open_radius", type=int, default=1)
    p.add_argument("--close_radius", type=int, default=1)
    p.add_argument("--show_nuclei", action="store_true", help="Optional later-use flag to visualize simple nuclei markers")
    p.add_argument("--nuclei_threshold_percentile", type=float, default=97.0)
    p.add_argument("--out_path", required=True, type=str)
    return p.parse_args()


def load_checkpoint(model, weights_path: Path, device):
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


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


def detect_nuclei_markers(image_np, percentile=97.0, min_size=6):
    red = image_np[..., 0]
    smooth = ndi.gaussian_filter(red, sigma=1.0)
    thresh = np.percentile(smooth, percentile)
    nuclei = smooth >= thresh
    nuclei = remove_small_objects(nuclei, min_size=min_size)
    nuclei_labels = label(nuclei)
    centers = []
    for lab in range(1, nuclei_labels.max() + 1):
        coords = np.argwhere(nuclei_labels == lab)
        if coords.size == 0:
            continue
        centers.append(tuple(np.round(coords.mean(axis=0)).astype(int)))
    return nuclei_labels, centers


def postprocess_binary(binary, min_size=80, hole_area=80, open_radius=1, close_radius=1):
    binary = remove_small_objects(binary.astype(bool), min_size=min_size)
    binary = remove_small_holes(binary, area_threshold=hole_area)
    if open_radius > 0:
        binary = binary_opening(binary, disk(open_radius))
    if close_radius > 0:
        binary = binary_closing(binary, disk(close_radius))
    return binary.astype(np.uint8)


def watershed_from_probability(
    prob_map,
    threshold=0.5,
    peak_footprint=15,
    min_size=80,
    hole_area=80,
    open_radius=1,
    close_radius=1,
    nuclei_centers=None,
):
    binary = postprocess_binary(prob_map > threshold, min_size=min_size, hole_area=hole_area, open_radius=open_radius, close_radius=close_radius)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.uint8), np.zeros_like(binary, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)
    markers = np.zeros(distance.shape, dtype=np.int32)

    if nuclei_centers:
        for idx, (r, c) in enumerate(nuclei_centers, start=1):
            if 0 <= r < markers.shape[0] and 0 <= c < markers.shape[1] and binary[r, c]:
                markers[r, c] = idx
    else:
        footprint = np.ones((peak_footprint, peak_footprint), dtype=bool)
        coords = peak_local_max(distance, labels=binary, footprint=footprint, exclude_border=False)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

    if markers.max() == 0:
        labels = label(binary)
    else:
        markers = label(markers > 0)
        labels = watershed(-distance, markers, mask=binary)
    return (labels > 0).astype(np.uint8), labels


def boundary_overlay(image_np, labels):
    canvas = np.clip(image_np.copy(), 0, 1)
    boundaries = find_boundaries(labels, mode="outer")
    canvas[boundaries] = np.array([1.0, 0.92, 0.15])
    return canvas, boundaries


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


def contour_overlay(ax, labels, linewidth=1.0):
    for lab in range(1, labels.max() + 1):
        region = labels == lab
        if not np.any(region):
            continue
        contours = find_contours(region.astype(float), 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetSmall().to(device)
    load_checkpoint(model, Path(args.weights), device)
    model.eval()

    image_tensor, image_np, gt_np = load_sample(Path(args.data_dir), args.sample_id, args.image_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        raw_binary = (prob > args.threshold).astype(np.uint8)

    nuclei_labels, nuclei_centers = detect_nuclei_markers(image_np, percentile=args.nuclei_threshold_percentile)
    separated_binary, labels = watershed_from_probability(
        prob,
        threshold=args.threshold,
        peak_footprint=args.peak_footprint,
        min_size=args.min_size,
        hole_area=args.hole_area,
        open_radius=args.open_radius,
        close_radius=args.close_radius,
        nuclei_centers=nuclei_centers if args.show_nuclei else None,
    )
    labels, count = count_instances(labels, min_size=max(20, args.min_size // 2))
    traced_overlay, boundaries = boundary_overlay(image_np, labels)

    raw_overlay = image_np.copy()
    raw_overlay[raw_binary > 0] = 0.72 * raw_overlay[raw_binary > 0] + 0.28 * np.array([1.0, 0.2, 0.2])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    fig.patch.set_facecolor("white")

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Raw image", fontsize=13, fontweight="bold")

    axes[0, 1].imshow(gt_np, cmap="gray")
    axes[0, 1].set_title("Ground truth", fontsize=13, fontweight="bold")

    im = axes[0, 2].imshow(prob, cmap="viridis")
    axes[0, 2].set_title("U-Net probability map", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    axes[1, 0].imshow(raw_overlay)
    axes[1, 0].set_title("Raw binary overlay", fontsize=13, fontweight="bold")

    axes[1, 1].imshow(labels, cmap="nipy_spectral")
    axes[1, 1].set_title(f"Separated instances ({count})", fontsize=13, fontweight="bold")

    axes[1, 2].imshow(traced_overlay)
    contour_overlay(axes[1, 2], labels, linewidth=0.85)
    if args.show_nuclei:
        for r, c in nuclei_centers:
            axes[1, 2].plot(c, r, marker="o", markersize=2.8)
    axes[1, 2].set_title(f"Final traced boundaries | Count = {count}", fontsize=13, fontweight="bold")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(f"Sample {args.sample_id}: segmentation, separation, and counting", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_path}")
    print(f"Cell count: {count}")


if __name__ == "__main__":
    main()
