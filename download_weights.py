"""
download_weights.py — Download NeuralRestore Model Weights
============================================================
Downloads all pre-trained model weights from HuggingFace Hub
into the weights/ directory. Run this once before using the app.

Usage:
    python download_weights.py
"""

import os
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
REPO_ID = "sp533/NeuralRestore"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

WEIGHT_FILES = [
    "model1_best_v2.pth",   # Damage Detector  (EfficientNet-B4 + UNet)
    "model2_best_v3.pth",   # Inpainting       (LaMa FFC Generator)
    "model3_best.pth",      # Denoising        (DnCNN)
    "model4_best.pth",      # Deblurring       (NAFNet)
    "RealESRGAN_x4.pth",    # Super Resolution (Real-ESRGAN 4x)
]


def download_with_hub():
    """Download using huggingface_hub (preferred — shows progress bar)."""
    from huggingface_hub import hf_hub_download

    for filename in WEIGHT_FILES:
        dest = WEIGHTS_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} (already exists, skipping)")
            continue

        print(f"  ↓ Downloading {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=WEIGHTS_DIR,
        )
        print(f"  ✓ {filename}")


def download_with_urllib():
    """Fallback download using only stdlib (no dependencies needed)."""
    import urllib.request
    import shutil

    BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

    for filename in WEIGHT_FILES:
        dest = WEIGHTS_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} (already exists, skipping)")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"  ↓ Downloading {filename}...")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NeuralRestore/1.0"})
            with urllib.request.urlopen(req) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 8192

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total / (1024 * 1024)
                            print(f"\r    {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

                print()  # newline after progress
            print(f"  ✓ {filename}")

        except Exception as e:
            # Clean up partial file on failure
            if dest.exists():
                dest.unlink()
            print(f"\n  ✗ Failed to download {filename}: {e}")
            return False

    return True


def main():
    print("=" * 55)
    print("  NeuralRestore — Model Weight Downloader")
    print("=" * 55)
    print(f"\n  Source:  huggingface.co/{REPO_ID}")
    print(f"  Target:  {WEIGHTS_DIR}/")
    print(f"  Files:   {len(WEIGHT_FILES)} models\n")

    # Create weights directory
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first, fall back to urllib
    try:
        import huggingface_hub  # noqa: F401
        print("  Using: huggingface_hub\n")
        download_with_hub()
    except ImportError:
        print("  Using: urllib (install 'huggingface_hub' for faster downloads)\n")
        if not download_with_urllib():
            sys.exit(1)

    # Verify all files exist
    print("\n" + "-" * 55)
    missing = [f for f in WEIGHT_FILES if not (WEIGHTS_DIR / f).exists()]
    if missing:
        print("  ⚠ Missing files:")
        for f in missing:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print("  ✅ All weights downloaded successfully!")
        print(f"  📂 Location: {WEIGHTS_DIR}/")
        print("\n  Run the app:  python run.py")
    print()


if __name__ == "__main__":
    main()
