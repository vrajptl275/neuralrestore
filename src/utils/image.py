"""
image.py — Image Processing Utilities
=======================================
Shared helper functions for loading, padding, tensor conversion,
and resizing images throughout the pipeline.
"""

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


def load_image(path_or_array):
    """Load image from path, PIL, or numpy array → RGB numpy uint8."""
    if isinstance(path_or_array, str):
        return np.array(Image.open(path_or_array).convert("RGB"))
    elif isinstance(path_or_array, Image.Image):
        return np.array(path_or_array.convert("RGB"))
    elif isinstance(path_or_array, np.ndarray):
        return path_or_array.astype(np.uint8)
    raise ValueError(f"Unsupported input type: {type(path_or_array)}")


def pad_to_square(image: Image.Image):
    """
    Pad image to square with black borders (preserves aspect ratio).
    Returns (padded_img, pad_info) — pad_info used to unpad after restoration.
    """
    w, h  = image.size
    size  = max(w, h)
    pad_w = (size - w) // 2
    pad_h = (size - h) // 2
    padded = Image.new('RGB', (size, size), (0, 0, 0))
    padded.paste(image, (pad_w, pad_h))
    pad_info = {'orig_w': w, 'orig_h': h,
                'pad_w': pad_w, 'pad_h': pad_h, 'size': size}
    return padded, pad_info


def unpad_from_square(image: Image.Image, pad_info: dict) -> Image.Image:
    """Crop padded square image back to original aspect ratio."""
    pw, ph = pad_info['pad_w'], pad_info['pad_h']
    ow, oh = pad_info['orig_w'], pad_info['orig_h']
    sz     = pad_info['size']
    return image.crop((pw, ph, sz - (sz - ow - pw), sz - (sz - oh - ph)))


def to_tensor(img_np, size=256):
    """numpy uint8 (H,W,3) → normalized tensor (1,3,size,size) in [-1,1]."""
    pil = Image.fromarray(img_np).resize((size, size), Image.LANCZOS)
    t   = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])(pil)
    return t.unsqueeze(0)


def to_numpy(tensor):
    """normalized tensor (1,3,H,W) in [-1,1] → uint8 numpy (H,W,3)."""
    t = tensor[0].cpu() * 0.5 + 0.5
    t = t.permute(1, 2, 0).numpy()
    return (t * 255).clip(0, 255).astype(np.uint8)


def resize_to(img_np, target_size):
    """Resize numpy image to (W, H) tuple."""
    pil = Image.fromarray(img_np)
    return np.array(pil.resize(target_size, Image.LANCZOS))
