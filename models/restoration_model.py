"""NAFSR: a NAFNet-inspired unified denoiser + super-resolution network.

=====================================================================================
ARCHITECTURE SELECTION RATIONALE (expanded version in README.md "Architecture")
=====================================================================================
Candidates considered: NAFNet, Restormer, SwinIR, RCAN, EDSR, plain U-Net, Residual
Dense Network (RRDB), lightweight hybrid CNN/Transformer.

- SwinIR / Restormer: strong quality via (windowed / channel) self-attention, but
  attention adds real latency and memory overhead, and semiconductor inspection
  images are dominated by local, repetitive texture (periodic patterns, edges)
  rather than long-range global relationships -- the case where full/window
  attention pays for itself the most. For a benchmarked, real-time-ish H100
  inference budget, that cost is not justified here.
- RCAN / EDSR: excellent super-resolution baselines, but designed for clean
  bicubic-downsampled inputs -- they have no explicit denoising capacity for
  speckle + Gaussian noise, so a second network would be needed (violates the
  "single unified model" preference).
- Plain U-Net: good denoiser, weak at generating genuine high-frequency detail
  during upsampling (tends to blur), and offers no attention-based feature
  recalibration.
- NAFNet ("Nonlinear Activation Free Network"): state-of-the-art *denoising*
  quality at a fraction of the FLOPs of attention-based restorers, achieved by
  replacing attention/activation nonlinearities with depthwise convs + a
  "SimpleGate" (channel-split multiplicative gate) + Simplified Channel
  Attention (SCA, a single global-pool + 1x1-conv gate instead of full softmax
  attention). This is the best available quality/speed trade-off for the
  denoising half of the problem.

DECISION: use a NAFNet-style encoder-decoder (U-Net topology, NAFBlocks) as the
*denoising backbone*, then attach a PixelShuffle super-resolution tail with a
few extra residual refinement blocks for the *upsampling* half. This keeps
everything as ONE forward pass / one set of weights (not a two-stage cascade
of separate models), reuses NAFNet's cheap building blocks throughout (so the
SR tail stays cheap too), and lets scale_factor be any power of two via a
configurable number of PixelShuffle stages (2 -> 128->256 or 256->512,
4 -> 128->512, 1 -> denoising-only ablation baseline).

Global residual learning: the network predicts a residual that is added to a
bicubic upsample of the (normalized) degraded input, rather than reconstructing
the whole image from scratch. This is well established to speed convergence
and reduce over-smoothing because low-frequency content is passed through
almost for free, leaving the network's capacity for genuine high-frequency
recovery and noise removal.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor (as in NAFNet)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies -- a cheap, effective, activation-free
    nonlinearity used throughout NAFNet in place of ReLU/GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global-average-pool + 1x1 conv gate: a cheap substitute for full channel
    self-attention that still lets the network recalibrate feature importance."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """Core building block: depthwise conv + SimpleGate + SCA, then an FFN-style
    1x1-conv branch, each with a learnable per-channel residual scale."""

    def __init__(self, channels: int, expand_ratio: int = 2, drop_path: float = 0.0):
        super().__init__()
        hidden = channels * expand_ratio

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden // 2)
        self.conv2 = nn.Conv2d(hidden // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(hidden // 2, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_path
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = residual + self._drop_path(y) * self.beta

        residual = x
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.gate2(y)
        y = self.conv4(y)
        x = residual + self._drop_path(y) * self.gamma
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class PixelShuffleUpBlock(nn.Module):
    """One 2x super-resolution stage: 1x1 conv to 4C channels + PixelShuffle(2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class NAFSR(nn.Module):
    """Unified denoising + super-resolution network.

    Pipeline: input projection -> NAFNet-style multi-scale encoder/decoder
    (denoising backbone, with skip connections) -> PixelShuffle SR tail with
    residual refinement blocks -> output projection -> global residual add of
    a bicubic-upsampled copy of the input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        enc_blocks: Optional[List[int]] = None,
        middle_blocks: int = 4,
        dec_blocks: Optional[List[int]] = None,
        scale_factor: int = 2,
        refine_blocks: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        enc_blocks = enc_blocks or [2, 2, 4]
        dec_blocks = dec_blocks or [2, 2, 2]
        assert len(enc_blocks) == len(dec_blocks), "enc_blocks and dec_blocks must have equal length"
        if scale_factor > 1 and (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError(f"scale_factor must be a power of 2 (or 1), got {scale_factor}")

        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_channels
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch, drop_path=drop_path_rate) for _ in range(n)]))
            self.downs.append(Downsample(ch))
            ch *= 2

        # ---- Bottleneck ----
        self.middle = nn.Sequential(*[NAFBlock(ch, drop_path=drop_path_rate) for _ in range(middle_blocks)])

        # ---- Decoder ----
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()
        for n in reversed(dec_blocks):
            self.ups.append(Upsample(ch))
            ch //= 2
            self.skip_fuse.append(nn.Conv2d(ch * 2, ch, kernel_size=1))
            self.decoders.append(nn.Sequential(*[NAFBlock(ch, drop_path=drop_path_rate) for _ in range(n)]))

        # ---- Super-resolution tail ----
        num_up_stages = int(math.log2(scale_factor)) if scale_factor > 1 else 0
        self.sr_ups = nn.ModuleList([PixelShuffleUpBlock(base_channels) for _ in range(num_up_stages)])
        self.refine = nn.Sequential(*[NAFBlock(base_channels, drop_path=drop_path_rate) for _ in range(refine_blocks)])

        self.output_proj = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the last conv so the network starts close to identity + bicubic.
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) normalized degraded input.
        Returns: (B, out_channels, H*scale, W*scale) restored image (GT-normalized space).
        """
        b, c, h, w = x.shape
        num_downs = len(self.encoders)
        multiple = 2 ** num_downs
        x_padded, pad_h, pad_w = self._pad_to_multiple(x, multiple)

        # Global residual base: bicubic upsample of the (unclamped) input.
        if self.scale_factor > 1:
            base = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        else:
            base = x

        feat = self.input_proj(x_padded)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, fuse, dec, skip in zip(self.ups, self.skip_fuse, self.decoders, reversed(skips)):
            feat = up(feat)
            feat = torch.cat([feat, skip], dim=1)
            feat = fuse(feat)
            feat = dec(feat)

        # Undo the reflect-padding used for the encoder/decoder downsampling.
        if pad_h or pad_w:
            feat = feat[:, :, : h, : w]

        for sr_up in self.sr_ups:
            feat = sr_up(feat)
        feat = self.refine(feat)
        residual = self.output_proj(feat)

        out = base + residual
        return out


def build_model(model_cfg) -> NAFSR:
    """Construct a NAFSR model from a config namespace/dict (see config/config.yaml -> model)."""
    return NAFSR(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 48),
        enc_blocks=list(model_cfg.get("enc_blocks", [2, 2, 4])),
        middle_blocks=model_cfg.get("middle_blocks", 4),
        dec_blocks=list(model_cfg.get("dec_blocks", [2, 2, 2])),
        scale_factor=model_cfg.get("scale_factor", 2),
        refine_blocks=model_cfg.get("refine_blocks", 4),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = NAFSR(base_channels=48, enc_blocks=[2, 2, 4], middle_blocks=4, dec_blocks=[2, 2, 2], scale_factor=2)
    dummy = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input: {tuple(dummy.shape)} -> Output: {tuple(out.shape)}")
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
