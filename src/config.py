"""
config.py — Central Configuration
===================================
All project-wide constants and default settings.
"""

import torch
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = PROJECT_ROOT / "weights"
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
SAMPLES_DIR  = DATA_DIR / "samples"

# ── Device Selection ───────────────────────────────────────────────────
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ── Model Weight Filenames ─────────────────────────────────────────────
WEIGHT_FILES = {
    "model1": "model1_best_v2.pth",
    "model2": "model2_best_v3.pth",
    "model3": "model3_best.pth",
    "model4": "model4_best.pth",
    "model5": "RealESRGAN_x4.pth",
}

# ── Damage Labels ──────────────────────────────────────────────────────
LABEL_NAMES = ["scratch", "stain", "crack", "dust"]
LABEL_ICONS = {
    "scratch": "✏️",
    "stain"  : "💧",
    "crack"  : "⚡",
    "dust"   : "🌫️",
}

# ── Default Inference Settings ─────────────────────────────────────────
DEFAULT_IMG_SIZE       = 256
DEFAULT_MASK_THRESHOLD = 0.25
DEFAULT_MASK_DILATION  = 5
DEFAULT_USE_TTA        = True
DEFAULT_APPLY_SR       = True
