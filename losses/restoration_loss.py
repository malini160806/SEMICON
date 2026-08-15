"""Composite restoration loss.

    L_total = w1 * L_charbonnier + w2 * L_ssim + w3 * L_gradient
            + w4 * L_frequency  + w5 * L_perceptual

All weights are configurable (config/config.yaml -> loss.*). Each component is
individually justified below; `L_perceptual` defaults to weight 0.0 (disabled)
because standard perceptual networks (VGG16/19) are trained on natural RGB
photographs, not grayscale semiconductor inspection imagery -- their learned
features are not a reliable domain match, and channel-replicating a
single-channel image to fake RGB does not fix the underlying distribution
mismatch. It is left available (channel-replicated VGG) for experimentation,
but the recommended default relies on the gradient + frequency losses instead,
which are domain-agnostic and directly target the failure modes described in
the problem statement (over-smoothing, loss of fine structure).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth, robust L1 variant: sqrt((x - y)^2 + eps^2).

    Preferred over plain MSE because it is far less sensitive to the residual
    outlier pixels that speckle noise leaves behind, while still being
    differentiable everywhere (unlike L1's kink at zero).
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d.unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian window (single-channel images).

    SSIM directly optimizes structural/luminance/contrast similarity, which
    complements the purely pixel-wise Charbonnier term and helps avoid the
    "smoothed average" look that pure L1/L2 optimization tends to produce.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.c1 = c1
        self.c2 = c2
        self._window_cache: Dict[str, torch.Tensor] = {}

    def _get_window(self, device, dtype) -> torch.Tensor:
        key = f"{device}_{dtype}"
        if key not in self._window_cache:
            self._window_cache[key] = _gaussian_window(self.window_size, self.sigma, device, dtype)
        return self._window_cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self._get_window(pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, window, padding=pad) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        ssim_map = ((2 * mu_pt + self.c1) * (2 * sigma_pt + self.c2)) / (
            (mu_p2 + mu_t2 + self.c1) * (sigma_p2 + sigma_t2 + self.c2)
        )
        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    """L1 distance between Sobel-gradient magnitudes of pred/target.

    Directly penalizes edge/structure mismatch, which is exactly what matters
    for preserving real semiconductor patterns and avoiding over-smoothing.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(pred), self._grad(target))


class FrequencyLoss(nn.Module):
    """L1 distance between 2D FFT magnitudes of pred/target.

    Encourages the network to match the high-frequency spectral content of
    the ground truth (fine texture / sharp edges) rather than only minimizing
    a spatial-domain average, which tends to under-recover high frequencies.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class VGGPerceptualLoss(nn.Module):
    """Optional channel-replicated VGG16 perceptual loss.

    Disabled by default (see module docstring): grayscale semiconductor
    imagery is out-of-domain for ImageNet-trained VGG features. Enable only
    for experimentation (config: loss.perceptual.weight > 0).
    """

    def __init__(self, layers: Optional[list[str]] = None):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        layer_map = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}
        layers = layers or ["relu2_2", "relu3_3"]
        self.layer_indices = sorted(layer_map[l] for l in layers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred3 = (pred.repeat(1, 3, 1, 1) - self.mean) / self.std
        target3 = (target.repeat(1, 3, 1, 1) - self.mean) / self.std
        loss = 0.0
        x, y = pred3, target3
        last = 0
        for idx in self.layer_indices:
            x = self.vgg[last:idx + 1](x)
            with torch.no_grad():
                y = self.vgg[last:idx + 1](y)
            loss = loss + F.l1_loss(x, y)
            last = idx + 1
        return loss


class CompositeLoss(nn.Module):
    """Weighted sum of all restoration loss components. Returns (total, dict-of-parts)."""

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        charbonnier_eps: float = 1e-3,
        ssim_weight: float = 0.3,
        ssim_window: int = 11,
        gradient_weight: float = 0.2,
        frequency_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        perceptual_layers: Optional[list[str]] = None,
    ):
        super().__init__()
        self.w_char = charbonnier_weight
        self.w_ssim = ssim_weight
        self.w_grad = gradient_weight
        self.w_freq = frequency_weight
        self.w_perc = perceptual_weight

        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps) if self.w_char > 0 else None
        self.ssim = SSIMLoss(window_size=ssim_window) if self.w_ssim > 0 else None
        self.gradient = GradientLoss() if self.w_grad > 0 else None
        self.frequency = FrequencyLoss() if self.w_freq > 0 else None
        self.perceptual = VGGPerceptualLoss(perceptual_layers) if self.w_perc > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        parts: Dict[str, float] = {}
        total = pred.new_zeros(())

        if self.charbonnier is not None:
            v = self.charbonnier(pred, target)
            total = total + self.w_char * v
            parts["charbonnier"] = v.item()
        if self.ssim is not None:
            v = self.ssim(pred, target)
            total = total + self.w_ssim * v
            parts["ssim"] = v.item()
        if self.gradient is not None:
            v = self.gradient(pred, target)
            total = total + self.w_grad * v
            parts["gradient"] = v.item()
        if self.frequency is not None:
            v = self.frequency(pred, target)
            total = total + self.w_freq * v
            parts["frequency"] = v.item()
        if self.perceptual is not None:
            v = self.perceptual(pred, target)
            total = total + self.w_perc * v
            parts["perceptual"] = v.item()

        parts["total"] = total.item()
        return total, parts


def build_loss(loss_cfg) -> CompositeLoss:
    """Construct a CompositeLoss from a config namespace/dict (config.yaml -> loss)."""
    return CompositeLoss(
        charbonnier_weight=loss_cfg.get("charbonnier", {}).get("weight", 1.0),
        charbonnier_eps=loss_cfg.get("charbonnier", {}).get("eps", 1e-3),
        ssim_weight=loss_cfg.get("ssim", {}).get("weight", 0.3),
        ssim_window=loss_cfg.get("ssim", {}).get("window_size", 11),
        gradient_weight=loss_cfg.get("gradient", {}).get("weight", 0.2),
        frequency_weight=loss_cfg.get("frequency", {}).get("weight", 0.1),
        perceptual_weight=loss_cfg.get("perceptual", {}).get("weight", 0.0),
        perceptual_layers=loss_cfg.get("perceptual", {}).get("layers", None),
    )
