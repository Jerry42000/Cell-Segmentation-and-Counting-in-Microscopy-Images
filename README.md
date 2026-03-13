
# Cell Segmentation and Counting in Microscopy Images

This repository contains APS360 project code for microscopy-based cell segmentation, including both a classical image-processing baseline and a preliminary deep learning model.

## Files

- **`baseline_watershed.py`**  
  Implements a classical watershed-based baseline for binary cell segmentation using grayscale conversion, Gaussian smoothing, thresholding, morphology, distance transform, and watershed. Saves quantitative metrics and example outputs.

- **`train_unet_quick.py`**  
  Trains a reduced U-Net prototype for binary cell segmentation. Saves model weights, validation metrics, a learning-curve plot, and an example prediction.

- **`predict_unet_single.py`**  
  Loads a trained U-Net model and runs inference on one specified sample to generate a qualitative comparison figure.

## Objective

The goal of this project is to evaluate whether a neural segmentation model can outperform a simple rule-based baseline on microscopy images of cells.

## Expected Data Format

All scripts assume paired image and mask files of the form:


*XXX_img.png
*XXX_masks.png

for instance:

*000_img.png
*005_masks.png
