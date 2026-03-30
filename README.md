
# Cell Segmentation and Counting in Microscopy Images

This repository contains the final codebase for binary cell segmentation and counting in fluorescence microscopy images. The project compares a classical watershed baseline against a lightweight U-Net segmentation model, then extends the learned model with post-processing for instance separation, boundary tracing, and approximate cell counting.

## Repository Files

### `baseline_watershed.py`
Implements the classical image-processing baseline for binary cell segmentation.  
Pipeline:
- grayscale conversion
- Gaussian smoothing
- Otsu thresholding
- morphological cleanup
- distance transform
- watershed segmentation

Outputs:
- quantitative metrics
- per-image CSV results
- qualitative example figures
- traced boundaries and approximate counts

### `train_unet.py`
Trains the final lightweight U-Net model for binary cell segmentation.

Main features:
- reduced encoder-decoder U-Net backbone
- BCE + Dice hybrid loss
- boundary-aware weighting
- optional target erosion
- optional augmentation
- learning-rate scheduling
- early stopping
- checkpoint saving
- validation evaluation during training
- optional held-out test-set evaluation after training

Outputs:
- best model weights
- checkpoint file
- validation summary
- optional test summary
- learning curve
- qualitative validation/test examples

### `predict_unet.py`
Runs inference on one specified image-mask pair using a trained U-Net model.

Main features:
- loads either plain weights or checkpoint dicts
- produces probability map
- optional watershed-based instance separation
- traced boundary overlay
- final approximate cell count
- polished multi-panel qualitative figure

### `evaluate_test.py`
Evaluates a trained U-Net model on a separate unseen test dataset without retraining.

Main features:
- accepts either a test folder or test ZIP archive
- computes test IoU, Dice, and pixel accuracy
- saves per-image metrics
- can also generate one final qualitative panel for a chosen test sample

---

## Project Objective

The objective of this project is to evaluate whether a lightweight deep learning segmentation model can outperform a classical watershed baseline on microscopy cell images, while also supporting boundary tracing and approximate counting after post-processing.

---

## Expected Data Format

All scripts assume paired image and mask files of the form:

- `XXX_img.png`
- `XXX_masks.png`

Example:
- `000_img.png`
- `000_masks.png`

Masks are assumed to be binary foreground/background annotations.

---

## Final Model

The final learned flow is:

1. Preprocess microscopy image and mask pairs  
2. Train U-Net for binary cell segmentation  
3. Produce a cell probability map  
4. Threshold and clean the predicted mask  
5. Apply watershed instance separation  
6. Trace final boundaries on the original image  
7. Output an approximate cell count

---
### Sample Output Image:
<img width="1537" height="884" alt="image" src="https://github.com/user-attachments/assets/cb53cfda-670b-48b2-a33c-898a6365ec26" />

## How to Run

### Train the U-Net
```bash
python train_unet.py \
  --data_dir /path/to/train_data \
  --out_dir /path/to/output_dir \
  --use_augmentation
