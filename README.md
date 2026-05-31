# 🖼️ NeuralRestore — AI-Powered Image Restoration

> Automatically detect and repair damage in old, degraded, or damaged images using a 5-model deep learning pipeline.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Automatic damage detection** — no manual masking needed
- **5-model pipeline** — detects, inpaints, denoises, deblurs, and upscales
- **Handles any image** — color, grayscale, sepia, old photos
- **Runs fully local** — no internet or cloud dependency
- **Beautiful Gradio UI** — drag-and-drop with before/after comparison

## 🏗️ Architecture

```
Damaged Image
     ↓
[Model 1] Damage Detector  → EfficientNet-B4 + UNet (mask + labels)
     ↓
[Model 2] Inpainting       → LaMa FFC Generator (if damage found)
     ↓
[Model 3] Denoising        → DnCNN (always)
     ↓
[Model 4] Deblurring       → NAFNet (always)
     ↓
[Model 5] Super Resolution → Real-ESRGAN 4x (always)
     ↓
✅ Restored HD Image
```

## 📁 Project Structure

```
neuralrestore/
├── run.py                  # Entry point — launches UI
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore configuration
│
├── src/                    # Main source package
│   ├── app.py              # Gradio web interface
│   ├── pipeline.py         # Pipeline orchestrator
│   ├── config.py           # Central configuration
│   ├── models/             # Model architectures
│   │   ├── damage_detector.py
│   │   ├── inpainting.py
│   │   ├── denoising.py
│   │   ├── deblurring.py
│   │   └── super_resolution.py
│   └── utils/              # Shared utilities
│       └── image.py
│
├── weights/                # Model weight files (.pth)
├── data/                   # Dataset directory
│   └── dataset/            # Training datasets
└── docs/                   # Documentation
    ├── architecture.md     # Pipeline and models reference
    └── training.md         # Model training documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

Place the following files in the `weights/` directory:

| File | Model | Source |
|---|---|---|
| `model1_best_v2.pth` | Damage Detector | Trained on Kaggle |
| `model2_best_v3.pth` | Inpainting | Trained on Kaggle |
| `model3_best.pth` | Denoising | Trained on Kaggle |
| `model4_best.pth` | Deblurring | Trained on Kaggle |
| `RealESRGAN_x4.pth` | Super Resolution | Prepretrained (HuggingFace Hub) |

### 3. Launch the App

```bash
python run.py
```

The Gradio UI will open at [http://localhost:7860](http://localhost:7860).

### CLI Usage

```bash
python -m src.pipeline --image path/to/damaged.png --output outputs/restored.png
python -m src.pipeline --image path/to/damaged.png --no_sr  # Skip super resolution
```

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `mask_threshold` | 0.25 | Lower = detect more damage (try 0.10–0.15 for subtle) |
| `mask_dilation` | 5 | Expand mask in pixels (helps inpainting blend) |
| `use_tta` | True | Test-time augmentation (4 flip variants) |
| `apply_sr` | True | Apply 4x super resolution |

## 🧪 Models

| # | Model | Architecture | Task | Input |
|---|---|---|---|---|
| 1 | Damage Detector | EfficientNet-B4 + UNet | Mask + 4 labels | 256×256 RGB |
| 2 | Inpainting | LaMa (FFC ResNet) | Fill damaged regions | 256×256 + mask |
| 3 | Denoising | DnCNN (17 layers) | Remove noise | 256×256 RGB |
| 4 | Deblurring | NAFNet | Remove blur | 256×256 RGB |
| 5 | Super Resolution | Real-ESRGAN (RRDBNet) | 4x upscale | Any size RGB |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
