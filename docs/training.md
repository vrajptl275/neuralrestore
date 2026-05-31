# 🔬 NeuralRestore — Model Training Documentation

This document summarizes the training configurations, dataset splits, losses, and metrics for each model.

---

## 🗂️ Dataset Splits & Sizes

The models are trained using synthetic damage overlayed on clean high-resolution source datasets (DIV2K, Flickr2K, FFHQ, and BSD500).

| Model | Task | Train Size | Val Size | Test Size | Total |
|---|---|---|---|---|---|
| **Model 1** | Damage Detector | 37,920 | 4,740 | 4,740 | 47,400 |
| **Model 2** | Inpainting | 37,920 | 4,740 | 4,740 | 47,400 |
| **Model 3** | Denoising | 6,320 | 790 | 790 | 7,900 |
| **Model 4** | Deblurring | 6,320 | 790 | 790 | 7,900 |

---

## ⚙️ Model Training Configurations

### Model 1: Damage Detector
* **Loss Function**: $\text{Loss}_{\text{total}} = (0.5 \times \text{BCE} + 0.5 \times \text{Dice}) + 0.5 \times \text{BCEWithLogitsLoss}$ (joint mask segmentation and multi-label classification).
* **Metrics**: IoU (Intersection over Union) & Dice coefficient for masks, Accuracy & F1 score for label head.
* **Target**: IoU > 0.85.

### Model 2: Inpainting
* **Loss Function**: $L_1\text{Loss} + 0.2 \times \text{PerceptualLoss} + 0.2 \times \text{FFTLoss} + 0.05 \times \text{PatchGANLoss}$ (incorporates spatial, frequency, and adversarial metrics for realistic gap filling).
* **Metrics**: Peak Signal-to-Noise Ratio (PSNR) & Structural Similarity Index (SSIM).
* **Target**: PSNR > 30dB, SSIM > 0.90.

### Model 3: Denoising
* **Loss Function**: $L_1\text{Loss} + 0.2 \times \text{PerceptualLoss}$ (trained on Gaussian, Poisson, and mixed noise overlays).
* **Metrics**: PSNR & SSIM.
* **Target**: PSNR > 35dB, SSIM > 0.95.

### Model 4: Deblurring
* **Loss Function**: PSNR loss (L1 loss in the frequency domain) + Perceptual Loss (trained on Gaussian, motion, and defocus blurs).
* **Metrics**: PSNR & SSIM.
* **Target**: PSNR > 32dB, SSIM > 0.92.

### Model 5: Super Resolution
* **Training**: None. Uses pretrained Real-ESRGAN weights.

---

## 💻 Hardware Setup
* **Platform**: Kaggle T4 GPU (15GB VRAM) or P100 GPU (16GB VRAM) with session durations up to 12 hours.
* **DataLoader settings**: Batch size 8–12, 2 workers, pin memory disabled to prevent VRAM overflow.