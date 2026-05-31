"""
damage_detector.py — Model 1: Damage Detector
================================================
Multi-task model:
  - Mask Head  (UNet decoder)  → binary segmentation mask
  - Label Head (FC classifier) → damage type classification

Architecture: EfficientNet-B4 encoder + UNet decoder + FC label head
Classes: scratch, stain, crack, dust
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


class MultiTaskDamageDetector(nn.Module):
    """
    Model 1: EfficientNet-B4 + UNet
    Output 1: binary mask  (WHERE is damage)
    Output 2: 4-class labels (WHAT type)
               scratch, stain, crack, dust
    """
    LABEL_NAMES = ["scratch", "stain", "crack", "dust"]

    def __init__(self, encoder="tu-efficientnet_b4", num_classes=4):
        super().__init__()
        if not SMP_AVAILABLE:
            raise RuntimeError("segmentation_models_pytorch required for Model 1")

        self.unet = smp.Unet(
            encoder_name          = encoder,
            encoder_weights       = None,
            in_channels           = 3,
            classes               = 1,
            activation            = None,
            decoder_channels      = (256, 128, 64, 32, 16),
            decoder_use_batchnorm = True,
        )
        encoder_out_ch = self.unet.encoder.out_channels[-1]
        self.label_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(encoder_out_ch),
            nn.Linear(encoder_out_ch, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features     = self.unet.encoder(x)
        decoder_out  = self.unet.decoder(*features)
        mask_logits  = self.unet.segmentation_head(decoder_out)
        label_logits = self.label_head(features[-1])
        return mask_logits, label_logits
