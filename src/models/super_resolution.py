"""
super_resolution.py — Model 5: Real-ESRGAN
============================================
Real-ESRGAN 4x Super Resolution
Plug & play — no training needed.

Uses pretrained RRDBNet with Real-ESRGAN upsampler.
"""

import cv2
import numpy as np


class RealESRGAN:
    """
    Real-ESRGAN 4x Super Resolution wrapper.

    Args:
        weights_path : Path to RealESRGAN_x4.pth
        device       : torch device
    """

    def __init__(self, weights_path, device):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_arch = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        self.upsampler = RealESRGANer(
            scale        = 4,
            model_path   = weights_path,
            model        = model_arch,
            tile         = 0,
            tile_pad     = 10,
            pre_pad      = 0,
            half         = device.type == "cuda",
            device       = device,
        )
        print(f"   ✅ Real-ESRGAN loaded from {weights_path}")

    def enhance(self, img_np):
        """
        Upscale image by 4x.

        Args:
            img_np : np.ndarray (H, W, 3) uint8 RGB
        Returns:
            enhanced : np.ndarray (H*4, W*4, 3) uint8 RGB
        """
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        enhanced_bgr, _ = self.upsampler.enhance(img_bgr, outscale=4)
        return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
