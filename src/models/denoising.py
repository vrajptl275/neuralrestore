"""
denoising.py — Model 3: DnCNN Denoiser
========================================
DnCNN: Beyond a Gaussian Denoiser
Handles: Gaussian noise, Poisson noise, Mixed noise

Approach: Residual learning — predicts noise, subtracts from input.
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN denoising network.

    Args:
        in_channels  : Number of input channels (default: 3 for RGB)
        num_layers   : Total number of conv layers (default: 17)
        num_features : Number of intermediate feature channels (default: 64)
    """

    def __init__(self, in_channels=3, num_layers=17, num_features=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        layers.append(
            nn.Conv2d(num_features, in_channels, 3, padding=1, bias=True)
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning: predict noise, subtract from input
        noise = self.dncnn(x)
        return torch.clamp(x - noise, -1.0, 1.0)
