"""
deblurring.py — Model 4: NAFNet Deblurring
============================================
NAFNet: Nonlinear Activation Free Network for Image Restoration
Handles: Gaussian blur, motion blur, defocus blur

Key innovations:
  - SimpleGate replaces activation functions
  - Simple Channel Attention (SCA) for feature recalibration
  - Residual U-Net architecture with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies — replaces activation functions."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    """Lightweight channel attention via global average pooling."""
    def __init__(self, num_features):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, 1, bias=True),
        )

    def forward(self, x):
        return x * self.sca(x)


class NAFBlock(nn.Module):
    """Core building block of NAFNet."""
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel  = c * dw_expand
        self.conv1  = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2  = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3  = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sg     = SimpleGate()
        self.sca    = SimpleChannelAttention(dw_channel // 2)
        ffn_channel = ffn_expand * c
        self.conv4  = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5  = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1  = nn.GroupNorm(1, c)
        self.norm2  = nn.GroupNorm(1, c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x); x = self.conv2(x); x = self.sg(x)
        x = self.sca(x);   x = self.conv3(x); x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x); x = self.sg(x); x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet — U-Net with NAFBlocks for image deblurring.

    Args:
        img_channel    : Input/output image channels (default: 3)
        width          : Base feature width (default: 32)
        middle_blk_num : Number of middle NAFBlocks (default: 12)
        enc_blks       : Blocks per encoder stage (default: [2,2,4,8])
        dec_blks       : Blocks per decoder stage (default: [2,2,2,2])
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=12,
                 enc_blks=[2,2,4,8], dec_blks=[2,2,2,2]):
        super().__init__()
        self.intro   = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending  = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders    = nn.ModuleList()
        self.decoders    = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups         = nn.ModuleList()
        self.downs       = nn.ModuleList()
        chan = width
        for num in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan*2, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan*2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(enc_blks)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self._check_image_size(inp)
        x   = self.intro(inp)
        enc_skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x); enc_skips.append(x); x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(enc_skips)):
            x = up(x); x = x + skip; x = decoder(x)
        x = self.ending(x) + inp
        return x[:, :, :H, :W]

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
