"""Degradation-aware augmentation pipeline for paired (degraded, ground_truth) crops.

Two families of augmentation are applied, both configurable via config.yaml:

1. Geometric augmentations (applied identically to both LR degraded and HR GT
   crops so pairing stays exact): random crop, horizontal/vertical flip,
   90-degree rotations.

2. Synthetic degradation augmentation (applied ONLY to the degraded crop, with
   probability `synthetic_prob`): additional Gaussian noise, additional
   multiplicative speckle noise, mild Gaussian blur, and brightness/contrast
   jitter. This is layered ON TOP of the real degraded/gt pairs already
   provided by the dataset -- it never invents a synthetic pair from scratch,
   it just diversifies the noise statistics the model sees so it generalizes
   to out-of-distribution noise levels/sources at test time.

We deliberately avoid unrealistic augmentations (heavy color jitter, cutout,
mixup, elastic warps) that would distort real semiconductor structures and
teach the network to hallucinate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AugmentConfig:
    enabled: bool = True
    random_crop: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotate90: float = 0.5
    synthetic_prob: float = 0.3
    gauss_enabled: bool = True
    gauss_sigma_range: Tuple[float, float] = (0.0, 0.05)
    speckle_enabled: bool = True
    speckle_var_range: Tuple[float, float] = (0.0, 0.05)
    blur_enabled: bool = True
    blur_prob: float = 0.2
    blur_kernel_range: Tuple[int, int] = (3, 5)
    blur_sigma_range: Tuple[float, float] = (0.2, 1.0)
    intensity_enabled: bool = True
    brightness_range: Tuple[float, float] = (-0.05, 0.05)
    contrast_range: Tuple[float, float] = (0.9, 1.1)

    @staticmethod
    def from_dict(d: dict) -> "AugmentConfig":
        gn = d.get("gaussian_noise", {})
        sp = d.get("speckle_noise", {})
        bl = d.get("blur", {})
        it = d.get("intensity_jitter", {})
        return AugmentConfig(
            enabled=d.get("enabled", True),
            random_crop=d.get("random_crop", True),
            horizontal_flip=d.get("horizontal_flip", 0.5),
            vertical_flip=d.get("vertical_flip", 0.5),
            rotate90=d.get("rotate90", 0.5),
            synthetic_prob=d.get("synthetic_prob", 0.3),
            gauss_enabled=gn.get("enabled", True),
            gauss_sigma_range=tuple(gn.get("sigma_range", (0.0, 0.05))),
            speckle_enabled=sp.get("enabled", True),
            speckle_var_range=tuple(sp.get("variance_range", (0.0, 0.05))),
            blur_enabled=bl.get("enabled", True),
            blur_prob=bl.get("prob", 0.2),
            blur_kernel_range=tuple(bl.get("kernel_range", (3, 5))),
            blur_sigma_range=tuple(bl.get("sigma_range", (0.2, 1.0))),
            intensity_enabled=it.get("enabled", True),
            brightness_range=tuple(it.get("brightness_range", (-0.05, 0.05))),
            contrast_range=tuple(it.get("contrast_range", (0.9, 1.1))),
        )


def random_paired_crop(
    lr: np.ndarray, hr: np.ndarray, lr_patch: int, scale: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a matching (lr_patch, lr_patch*scale) pair at a random aligned location."""
    lh, lw = lr.shape[:2]
    if lh < lr_patch or lw < lr_patch:
        raise ValueError(f"LR image ({lh}x{lw}) smaller than requested patch {lr_patch}")
    top = int(rng.integers(0, lh - lr_patch + 1))
    left = int(rng.integers(0, lw - lr_patch + 1))
    lr_crop = lr[top:top + lr_patch, left:left + lr_patch]
    hr_top, hr_left = top * scale, left * scale
    hr_patch = lr_patch * scale
    hr_crop = hr[hr_top:hr_top + hr_patch, hr_left:hr_left + hr_patch]
    return lr_crop, hr_crop


def apply_geometric_pair(
    lr: np.ndarray, hr: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical flips/rotations to both crops so pairing stays exact."""
    if rng.random() < cfg.horizontal_flip:
        lr = np.ascontiguousarray(lr[:, ::-1])
        hr = np.ascontiguousarray(hr[:, ::-1])
    if rng.random() < cfg.vertical_flip:
        lr = np.ascontiguousarray(lr[::-1, :])
        hr = np.ascontiguousarray(hr[::-1, :])
    if rng.random() < cfg.rotate90:
        k = int(rng.integers(1, 4))
        lr = np.ascontiguousarray(np.rot90(lr, k))
        hr = np.ascontiguousarray(np.rot90(hr, k))
    return lr, hr


def _gaussian_kernel1d(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(kernel_size, dtype=np.float32) - (kernel_size - 1) / 2.0
    kernel = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _separable_gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    k = _gaussian_kernel1d(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode="reflect")
    # Horizontal pass
    tmp = np.zeros_like(img)
    for i, w in enumerate(k):
        tmp += w * padded[pad:pad + img.shape[0], i:i + img.shape[1]]
    padded2 = np.pad(tmp, pad, mode="reflect")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * padded2[i:i + img.shape[0], pad:pad + img.shape[1]]
    return out


def add_synthetic_degradation(
    lr_norm: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    """Layer additional realistic degradation onto an already-normalized LR crop
    (values roughly in [0, 1], but not clamped -- see utils/image_utils.py).

    Applied ONLY to the input, never to the ground truth, and only with
    probability `cfg.synthetic_prob` so the model still sees plenty of the
    real, unmodified degraded/gt pairs.
    """
    out = lr_norm.copy()

    if cfg.blur_enabled and rng.random() < cfg.blur_prob:
        k = int(rng.integers(cfg.blur_kernel_range[0], cfg.blur_kernel_range[1] + 1))
        if k % 2 == 0:
            k += 1
        sigma = float(rng.uniform(*cfg.blur_sigma_range))
        out = _separable_gaussian_blur(out, k, sigma)

    if cfg.speckle_enabled:
        var = float(rng.uniform(*cfg.speckle_var_range))
        if var > 0:
            noise = rng.normal(0.0, np.sqrt(var), size=out.shape).astype(np.float32)
            out = out + out * noise  # multiplicative speckle

    if cfg.gauss_enabled:
        sigma = float(rng.uniform(*cfg.gauss_sigma_range))
        if sigma > 0:
            out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)

    if cfg.intensity_enabled:
        brightness = float(rng.uniform(*cfg.brightness_range))
        contrast = float(rng.uniform(*cfg.contrast_range))
        out = out * contrast + brightness

    return out.astype(np.float32)
