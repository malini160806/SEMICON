"""Standalone evaluation metrics: PSNR, SSIM, LPIPS, plus model/inference
benchmarking helpers (parameter count, size on disk, latency, throughput).

PSNR/SSIM operate on numpy arrays in [0, 1] (single channel, HxW).
LPIPS operates on torch tensors: since the reference `lpips` package's
AlexNet/VGG backbones expect 3-channel input normalized to [-1, 1], grayscale
images are channel-replicated (L -> RGB by repetition) and rescaled from
[0, 1] to [-1, 1] before being passed in. This is documented explicitly
because LPIPS silently producing wrong values for 1-channel input is an easy
mistake -- the repetition avoids that failure mode while keeping the metric's
correlation-with-perceived-quality behavior (LPIPS itself is still a natural
image prior, so treat absolute LPIPS values on semiconductor imagery as a
*relative* comparison tool between model variants, not an absolute quality
guarantee -- this caveat is called out again in README.md).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR between two single-channel arrays in the same normalized range."""
    pred = np.clip(pred, 0.0, data_range)
    target = np.clip(target, 0.0, data_range)
    return float(peak_signal_noise_ratio(target, pred, data_range=data_range))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """SSIM between two single-channel arrays in the same normalized range."""
    pred = np.clip(pred, 0.0, data_range)
    target = np.clip(target, 0.0, data_range)
    return float(structural_similarity(target, pred, data_range=data_range))


class LPIPSMetric:
    """Lazy-loaded LPIPS wrapper (grayscale-safe via channel replication)."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips  # deferred import: heavyweight, only needed when LPIPS is used

        self.device = device
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """pred/target: (1, 1, H, W) or (1, H, W) or (H, W) tensors in [0, 1]."""
        pred = self._prep(pred)
        target = self._prep(target)
        d = self.model(pred, target)
        return float(d.item())

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device).float()
        x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        return x * 2.0 - 1.0  # [0,1] -> [-1,1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Approximate serialized size (float32 params + buffers), in megabytes."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


class InferenceBenchmark:
    """Measures warm-up time, per-image latency, and throughput for a model.

    Model loading time is measured separately by the caller (it happens once,
    outside the hot loop) -- see evaluate.py for the full breakdown.
    """

    def __init__(self, model: nn.Module, device: str, use_amp: bool = True):
        self.model = model
        self.device = device
        self.use_amp = use_amp and device == "cuda"

    @torch.inference_mode()
    def run(self, sample_input: torch.Tensor, warmup_iters: int = 5, timed_iters: int = 20) -> dict:
        sample_input = sample_input.to(self.device)

        for _ in range(warmup_iters):
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                self.model(sample_input)
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(timed_iters):
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                self.model(sample_input)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_image_s = elapsed / timed_iters
        peak_mem_mb: Optional[float] = None
        if self.device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return {
            "avg_inference_time_s": per_image_s,
            "images_per_second": 1.0 / per_image_s if per_image_s > 0 else float("inf"),
            "peak_gpu_memory_mb": peak_mem_mb,
        }
