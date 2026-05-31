"""
pipeline.py — NeuralRestore Pipeline
====================================
Connects all 5 models in the correct order:
  Model 1 → Damage Detector (EfficientNet-B4 + UNet)
  Model 2 → Inpainting (LaMa-inspired FFC)
  Model 3 → Denoising (DnCNN)
  Model 4 → Deblurring (NAFNet)
  Model 5 → Super Resolution (Real-ESRGAN)
"""

import sys
import gc
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import cv2

# Import custom modules
from src.config import DEVICE, WEIGHTS_DIR, WEIGHT_FILES, LABEL_NAMES
from src.models import (
    MultiTaskDamageDetector,
    LaMaGenerator,
    DnCNN,
    NAFNet,
    RealESRGAN
)
from src.models.damage_detector import SMP_AVAILABLE
from src.utils.image import (
    load_image,
    pad_to_square,
    unpad_from_square,
    to_tensor,
    to_numpy,
    resize_to
)


class NeuralRestorePipeline:
    """
    NeuralRestore — Full Image Restoration Pipeline

    Loads all 5 models and runs them in correct order:
      1. Damage Detection  (always)
      2. Inpainting        (only if damage found)
      3. Denoising         (always)
      4. Deblurring        (always)
      5. Super Resolution  (always, final step)
    """

    LABEL_NAMES = LABEL_NAMES

    def __init__(self, weights_dir=None, device=None):
        self.device      = device or DEVICE
        self.weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
        self.models      = {}

        print("\n" + "="*55)
        print("  🚀  NeuralRestore Pipeline — Loading Models")
        print("="*55)

        self._load_model1()
        self._load_model2()
        self._load_model3()
        self._load_model4()
        self._load_model5()

        print("="*55)
        print("  ✅  All models loaded — Pipeline ready!")
        print("="*55 + "\n")

    # ── Model Loading ────────────────────────────────────────────────

    def _load_model1(self):
        path = self.weights_dir / WEIGHT_FILES["model1"]
        print(f"\n[1/5] Loading Model 1 (Damage Detector)...")
        if not SMP_AVAILABLE:
            print("   ⚠️  Skipped — smp not installed")
            return
        try:
            m = MultiTaskDamageDetector(
                encoder="tu-efficientnet_b4", num_classes=4
            ).to(self.device)
            ckpt = torch.load(str(path), map_location=self.device,
                              weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models["model1"] = m
            print(f"   ✅ Loaded from {path}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    def _load_model2(self):
        path = self.weights_dir / WEIGHT_FILES["model2"]
        print(f"\n[2/5] Loading Model 2 (Inpainting)...")
        try:
            m = LaMaGenerator(
                input_nc=4, output_nc=3, ngf=64,
                n_downsampling=2, n_blocks=9,
                ratio_gin=0.75, ratio_gout=0.75,
            ).to(self.device)
            ckpt = torch.load(str(path), map_location=self.device,
                              weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models["model2"] = m
            print(f"   ✅ Loaded from {path}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    def _load_model3(self):
        path = self.weights_dir / WEIGHT_FILES["model3"]
        print(f"\n[3/5] Loading Model 3 (Denoising)...")
        try:
            m = DnCNN(in_channels=3, num_layers=17, num_features=64
                      ).to(self.device)
            ckpt = torch.load(str(path), map_location=self.device,
                              weights_only=False)
            state_dict = ckpt["model_state_dict"]
            state_dict = {k.replace("net.", "dncnn."): v for k, v in state_dict.items()}
            m.load_state_dict(state_dict)
            m.eval()
            self.models["model3"] = m
            print(f"   ✅ Loaded from {path}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    def _load_model4(self):
        path = self.weights_dir / WEIGHT_FILES["model4"]
        print(f"\n[4/5] Loading Model 4 (Deblurring)...")
        try:
            m = NAFNet(
                img_channel=3, width=32, middle_blk_num=12,
                enc_blks=[2,2,4,8], dec_blks=[2,2,2,2],
            ).to(self.device)
            ckpt = torch.load(str(path), map_location=self.device,
                              weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models["model4"] = m
            print(f"   ✅ Loaded from {path}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    def _load_model5(self):
        path = self.weights_dir / WEIGHT_FILES["model5"]
        print(f"\n[5/5] Loading Model 5 (Super Resolution)...")
        try:
            self.models["model5"] = RealESRGAN(str(path), self.device)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print("   ℹ️  Install: pip install realesrgan basicsr")

    # ── Inference Steps ──────────────────────────────────────────────

    def _run_model1(self, img_np,
                    threshold=0.25, dilation=5, use_tta=True):
        if "model1" not in self.models:
            return None, {k: 0 for k in self.LABEL_NAMES}

        pil_orig        = Image.fromarray(img_np)
        padded_pil, _   = pad_to_square(pil_orig)
        padded_np       = np.array(padded_pil.resize((256, 256), Image.LANCZOS),
                                   dtype=np.uint8)

        def _infer(arr):
            inp = to_tensor(arr).to(self.device)
            with torch.no_grad():
                mask_logit, lbl_logit = self.models["model1"](inp)
            mp = torch.sigmoid(mask_logit[0, 0]).cpu().numpy()
            lp = torch.sigmoid(lbl_logit[0]).cpu().numpy()
            return mp, lp

        if use_tta:
            variants = [
                padded_np,
                np.fliplr(padded_np).copy(),
                np.flipud(padded_np).copy(),
                np.fliplr(np.flipud(padded_np)).copy(),
            ]
            mask_preds, label_preds = [], []
            for idx, v in enumerate(variants):
                mp, lp = _infer(v)
                if idx == 1: mp = np.fliplr(mp)
                elif idx == 2: mp = np.flipud(mp)
                elif idx == 3: mp = np.fliplr(np.flipud(mp))
                mask_preds.append(mp)
                label_preds.append(lp)
            mask_prob   = np.max(mask_preds,  axis=0)
            label_probs = np.mean(label_preds, axis=0)
        else:
            mask_prob, label_probs = _infer(padded_np)

        mask_bin = (mask_prob > threshold).astype(np.uint8)

        if dilation > 0:
            kernel   = np.ones((dilation, dilation), np.uint8)
            mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

        mask_bin = mask_bin.astype(np.float32)

        lbl_prob = label_probs
        labels   = {}
        for i, name in enumerate(self.LABEL_NAMES):
            labels[name]                 = int(lbl_prob[i] > 0.5)
            labels[f"{name}_confidence"] = float(lbl_prob[i])

        return mask_bin, labels

    def _run_model2(self, img_np, mask_np):
        if "model2" not in self.models:
            return img_np

        original_w, original_h = img_np.shape[1], img_np.shape[0]
        SIZE = 256

        pil_orig        = Image.fromarray(img_np)
        padded_pil, pad_info = pad_to_square(pil_orig)
        img_resized_np  = np.array(padded_pil.resize((SIZE, SIZE), Image.LANCZOS),
                                   dtype=np.uint8)

        mask_256 = mask_np

        img_t  = torch.from_numpy(img_resized_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_256).unsqueeze(0)

        masked_rgb  = img_t * (1.0 - mask_t)
        model_input = torch.cat([masked_rgb, mask_t], dim=0).unsqueeze(0)
        model_input = model_input.to(self.device)

        img_t_dev   = img_t.unsqueeze(0).to(self.device)
        mask_t_dev  = mask_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            model_input_norm = model_input.clone()
            model_input_norm[:, :3] = model_input_norm[:, :3] * 2.0 - 1.0
            restored_t = self.models["model2"](
                img_t_dev * 2.0 - 1.0,
                mask_t_dev
            )
            restored_01 = (restored_t.clamp(-1, 1) + 1.0) / 2.0

        composite = img_t_dev * (1.0 - mask_t_dev) + restored_01 * mask_t_dev
        composite_np = (composite[0].cpu().permute(1, 2, 0).numpy() * 255) \
                       .clip(0, 255).astype(np.uint8)

        composite_pil   = Image.fromarray(composite_np)
        padded_restored = composite_pil.resize(
            (pad_info['size'], pad_info['size']), Image.LANCZOS
        )
        final_pil = unpad_from_square(padded_restored, pad_info)
        final_pil = final_pil.resize((original_w, original_h), Image.LANCZOS)
        return np.array(final_pil)

    def _run_model3(self, img_np):
        if "model3" not in self.models:
            return img_np

        original_size = (img_np.shape[1], img_np.shape[0])
        inp_t = to_tensor(img_np).to(self.device)

        with torch.no_grad():
            out_t = self.models["model3"](inp_t)

        out_np = to_numpy(out_t)
        return resize_to(out_np, original_size)

    def _run_model4(self, img_np):
        if "model4" not in self.models:
            return img_np

        original_size = (img_np.shape[1], img_np.shape[0])
        inp_t = to_tensor(img_np).to(self.device)

        with torch.no_grad():
            out_t = self.models["model4"](inp_t)

        out_np = to_numpy(out_t)
        return resize_to(out_np, original_size)

    def _run_model5(self, img_np):
        if "model5" not in self.models:
            return img_np
        try:
            return self.models["model5"].enhance(img_np)
        except Exception as e:
            print(f"   ⚠️  SR failed: {e} — skipping")
            return img_np

    # ── Main Restore Function ────────────────────────────────────────

    def restore(self, image_input, apply_sr=True, denoise_strength=None,
                mask_threshold=0.25, mask_dilation=5, use_tta=True, **kwargs):
        steps = []

        if kwargs:
            print(f"   ⚠️  Ignoring unsupported restore kwargs: {list(kwargs.keys())}")

        img_np = load_image(image_input)
        original_np = img_np.copy()
        print(f"\n[Pipeline] Input: {img_np.shape}")

        print("[Pipeline] Step 1: Damage Detection...")
        mask_np, labels = self._run_model1(
            img_np,
            threshold=mask_threshold,
            dilation=mask_dilation,
            use_tta=use_tta,
        )
        damage_found = any(labels.get(k, 0) == 1 for k in self.LABEL_NAMES)
        steps.append("damage_detection")

        detected = [k for k in self.LABEL_NAMES if labels.get(k, 0) == 1]
        if damage_found:
            print(f"   Damage detected: {detected}")
        else:
            print(f"   No damage detected — skipping inpainting")

        if damage_found and mask_np is not None:
            print("[Pipeline] Step 2: Inpainting...")
            img_np = self._run_model2(img_np, mask_np)
            steps.append("inpainting")
            print(f"   Done → {img_np.shape}")
        else:
            print("[Pipeline] Step 2: Inpainting — SKIPPED (no damage)")

        print("[Pipeline] Step 3: Denoising...")
        img_np = self._run_model3(img_np)
        steps.append("denoising")
        print(f"   Done → {img_np.shape}")

        print("[Pipeline] Step 4: Deblurring...")
        img_np = self._run_model4(img_np)
        steps.append("deblurring")
        print(f"   Done → {img_np.shape}")

        if apply_sr:
            print("[Pipeline] Step 5: Super Resolution (4x)...")
            img_np = self._run_model5(img_np)
            steps.append("super_resolution")
            print(f"   Done → {img_np.shape}")
        else:
            print("[Pipeline] Step 5: Super Resolution — SKIPPED")

        print(f"[Pipeline] ✅ Complete! Steps: {steps}")
        print(f"[Pipeline] Output shape: {img_np.shape}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

        return {
            "restored"      : img_np,
            "original"      : original_np,
            "mask"          : mask_np,
            "labels"        : labels,
            "damage_found"  : damage_found,
            "steps_applied" : steps,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuralRestore Pipeline Test")
    parser.add_argument("--image",       type=str, required=True,
                        help="Path to damaged image")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Directory containing model weights")
    parser.add_argument("--output",      type=str, default="restored.png",
                        help="Output path for restored image")
    parser.add_argument("--no_sr",       action="store_true",
                        help="Skip super resolution")
    args = parser.parse_args()

    pipeline = NeuralRestorePipeline(weights_dir=args.weights_dir)
    result = pipeline.restore(args.image, apply_sr=not args.no_sr)

    out_path = Path(args.output)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(result["restored"]).save(str(out_path))
    print(f"\n✅ Restored image saved → {out_path}")
    print(f"   Damage found  : {result['damage_found']}")
    print(f"   Labels        : {result['labels']}")
    print(f"   Steps applied : {result['steps_applied']}")
    print(f"   Output size   : {result['restored'].shape}")
