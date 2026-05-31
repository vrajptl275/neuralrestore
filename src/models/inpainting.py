"""
inpainting.py — Model 2: LaMa Inpainting
==========================================
LaMa-inspired architecture with Fast Fourier Convolutions (FFC)
for large-mask inpainting.

Components:
  - FourierUnit      → FFT-based global convolution
  - SpectralTransform → multi-scale frequency processing
  - FFC              → 4-path (l2l, l2g, g2l, g2g) convolutions
  - FFC_BN_ACT       → FFC + BatchNorm + ReLU
  - FFCResnetBlock   → residual FFC blocks
  - LaMaGenerator    → full generator model
"""

import torch
import torch.nn as nn


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels * 2, out_channels * 2,
            kernel_size=1, groups=groups, bias=False
        )
        self.bn  = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        fft     = torch.fft.rfft2(x, norm="ortho")
        fft_cat = torch.cat([fft.real, fft.imag], dim=1)
        fft_cat = self.act(self.bn(self.conv_layer(fft_cat)))
        real, imag = torch.chunk(fft_cat, 2, dim=1)
        return torch.fft.irfft2(
            torch.complex(real, imag), s=(H, W), norm="ortho"
        )


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super().__init__()
        self.stride = stride
        if stride == 2:
            self.downsample = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu    = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, groups=groups, bias=False)

    def forward(self, x):
        if self.stride == 2:
            x = self.downsample(x)
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=0,
                 groups=1, bias=False, enable_lfu=True):
        super().__init__()
        in_cg  = int(in_channels  * ratio_gin)
        in_cl  = in_channels  - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin  = ratio_gin
        self.ratio_gout = ratio_gout
        self.in_cl = in_cl; self.in_cg = in_cg
        self.out_cl = out_cl; self.out_cg = out_cg
        self.convl2l = nn.Conv2d(in_cl, out_cl, kernel_size, padding=padding, stride=stride, groups=groups, bias=bias) if in_cl > 0 and out_cl > 0 else nn.Identity()
        self.convl2g = nn.Conv2d(in_cl, out_cg, kernel_size, padding=padding, stride=stride, groups=groups, bias=bias) if in_cl > 0 and out_cg > 0 else nn.Identity()
        self.convg2l = nn.Conv2d(in_cg, out_cl, kernel_size, padding=padding, stride=stride, groups=groups, bias=bias) if in_cg > 0 and out_cl > 0 else nn.Identity()
        self.convg2g = SpectralTransform(in_cg, out_cg, stride, groups, enable_lfu) if in_cg > 0 and out_cg > 0 else nn.Identity()

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl = out_xg = 0
        if self.out_cl > 0:
            out_xl = self.convl2l(x_l)
            if isinstance(x_g, torch.Tensor) and self.in_cg > 0:
                out_xl = out_xl + self.convg2l(x_g)
        if self.out_cg > 0:
            out_xg = self.convl2g(x_l)
            if isinstance(x_g, torch.Tensor) and self.in_cg > 0:
                out_xg = out_xg + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=0,
                 groups=1, bias=False, enable_lfu=True):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                       stride=stride, padding=padding,
                       groups=groups, bias=bias, enable_lfu=enable_lfu)
        out_cl = out_channels - int(out_channels * ratio_gout)
        out_cg = int(out_channels * ratio_gout)
        self.bn_l  = nn.BatchNorm2d(out_cl) if out_cl > 0 else nn.Identity()
        self.bn_g  = nn.BatchNorm2d(out_cg) if out_cg > 0 else nn.Identity()
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l)) if isinstance(x_l, torch.Tensor) else x_l
        x_g = self.act_g(self.bn_g(x_g)) if isinstance(x_g, torch.Tensor) else x_g
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=True):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, 3, padding=1, ratio_gin=ratio_gin,  ratio_gout=ratio_gout, enable_lfu=enable_lfu)
        self.conv2 = FFC_BN_ACT(dim, dim, 3, padding=1, ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=enable_lfu)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        if isinstance(x_l, torch.Tensor) and isinstance(id_l, torch.Tensor):
            if x_l.shape == id_l.shape:
                x_l = x_l + id_l
        if isinstance(x_g, torch.Tensor) and isinstance(id_g, torch.Tensor):
            if x_g.shape == id_g.shape:
                x_g = x_g + id_g
        return x_l, x_g


class LaMaGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64,
                 n_downsampling=2, n_blocks=9,
                 ratio_gin=0.75, ratio_gout=0.75, enable_lfu=True):
        super().__init__()
        self.model = nn.Sequential(
            FFC_BN_ACT(input_nc, ngf,     7, padding=3, ratio_gin=0, ratio_gout=0, enable_lfu=enable_lfu),
            FFC_BN_ACT(ngf,     ngf*2,   3, padding=1, stride=2, ratio_gin=0, ratio_gout=0, enable_lfu=enable_lfu),
            FFC_BN_ACT(ngf*2,   ngf*4,   3, padding=1, stride=2, ratio_gin=0, ratio_gout=0, enable_lfu=enable_lfu),
            FFC_BN_ACT(ngf*4,   ngf*8,   3, padding=1, ratio_gin=0, ratio_gout=ratio_gout, enable_lfu=enable_lfu),
            *[FFCResnetBlock(ngf*8, ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=enable_lfu) for _ in range(n_blocks)],
            nn.ConvTranspose2d(ngf*8, ngf*4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*2), nn.ReLU(inplace=True),
            nn.Conv2d(ngf*2, output_nc, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, damaged, mask):
        x = torch.cat([damaged, mask], dim=1)
        out = x
        for layer in self.model:
            if isinstance(layer, (FFCResnetBlock, FFC_BN_ACT)):
                out = layer(out) if isinstance(out, tuple) else layer((out, 0))
            elif isinstance(out, tuple):
                out_l, out_g = out
                out = torch.cat([out_l, out_g], dim=1) if isinstance(out_g, torch.Tensor) else out_l
                out = layer(out)
            else:
                out = layer(out)
        if isinstance(out, tuple):
            out_l, out_g = out
            out = torch.cat([out_l, out_g], dim=1) if isinstance(out_g, torch.Tensor) else out_l
        restored = damaged * (1 - mask) + out * mask
        return torch.clamp(restored, -1.0, 1.0)
