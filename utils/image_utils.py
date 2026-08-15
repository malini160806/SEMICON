"""Image I/O and intensity-normalization utilities.

=====================================================================================
INTENSITY HANDLING STRATEGY (see README.md "Intensity Handling" for full discussion)
=====================================================================================
Degraded semiconductor inspection images can contain pixel intensities that exceed
the ground-truth dynamic range (multiplicative speckle noise produces bright spikes
well above the clean signal ceiling). Naively clamping or min-max normalizing the
degraded input to [0, 1] BEFORE the model sees it would silently destroy exactly the
information a denoiser needs to recognize "this is a noise spike, not a real bright
structure".

We therefore use two DIFFERENT, but individually consistent, normalization rules:

1. Ground truth (clean, canonical images):
   Normalized with a FIXED, bit-depth-derived divisor (`fixed_range` / native max,
   e.g. 255.0 for 8-bit, 65535.0 for 16-bit). GT is clean by construction, so a
   fixed scale is safe and keeps the target space consistent and stable for the
   loss functions and PSNR/SSIM computation across the whole dataset.

2. Degraded input (noisy, possibly out-of-range):
   Normalized with PER-IMAGE robust statistics computed from percentiles of that
   image (default 0.5th / 99.5th percentile), WITHOUT clamping the result:

       x_norm = (x - p_low) / (p_high - p_low + eps)

   Because we do not clip, a speckle spike that sits above the 99.5th percentile
   maps to a value > 1.0 (not clipped to 1.0), and a rare dark outlier can map to
   a value < 0.0. The network's first convolution therefore receives the true
   relative magnitude of every pixel instead of a truncated version of it. This
   directly implements the requirement: "Do NOT blindly normalize/clamp the
   degraded input before the model has a chance to use the information."

   Percentile (rather than raw min/max) is used because a single hot pixel would
   otherwise dominate the whole normalization scale; percentiles are robust to a
   small fraction of extreme outliers while still leaving them unclamped in the
   normalized signal.

The model is trained to regress directly into the GT's fixed normalized space, so
its output can be safely clipped to [0, 1] only at the very last step (after
restoration), then rescaled to the native bit depth for saving. This is the ONLY
clipping point in the whole pipeline.

`method` is configurable (percentile | minmax | fixed) via config/config.yaml ->
normalization.method, and the same function is used identically at train time and
at inference time in evaluate.py / inference.py, guaranteeing train/test consistency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


@dataclass
class NormStats:
    """Stats needed to invert a normalization back to native intensity space."""
    low: float
    high: float
    method: str


def native_max_for_dtype(dtype: np.dtype) -> float:
    """Return the canonical maximum intensity value for a given image dtype."""
    if dtype == np.uint8:
        return 255.0
    if dtype == np.uint16:
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 1.0
    return 255.0


def read_image_grayscale_meta(path: str | Path) -> Tuple[np.ndarray, float]:
    """Read an image from disk as a float32 single-channel array, and also
    return its native max intensity (255.0 for 8-bit, 65535.0 for 16-bit)
    inferred from the on-disk dtype -- used to save restored outputs back
    into the same bit-depth convention as the corresponding input.

    Preserves native bit depth (8-bit / 16-bit) without rescaling. Supports
    PNG, JPG/JPEG, BMP (via Pillow), TIFF (via tifffile if available, else
    Pillow fallback which supports most single-page TIFFs), and raw NumPy
    `.npy` arrays (e.g. pre-normalized float32 dumps, as used by the KLA
    dataset -- see README.md "Dataset Structure"). For `.npy` files whose
    dtype is already floating point, `native_max` is 1.0 (identity scale),
    matching `native_max_for_dtype`'s convention for float arrays.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(str(path))
        elif suffix in {".tif", ".tiff"} and _HAS_TIFFFILE:
            arr = tifffile.imread(str(path))
        else:
            with Image.open(path) as im:
                if im.mode not in ("L", "I", "I;16"):
                    im = im.convert("L")
                arr = np.array(im)
    except Exception as exc:  # noqa: BLE001 - re-raise with context
        raise ValueError(f"Failed to decode image '{path}': {exc}") from exc

    arr = np.asarray(arr)
    native_max = native_max_for_dtype(arr.dtype)
    if arr.ndim == 3:
        if suffix == ".npy" and 1 in arr.shape:
            # Squeeze a redundant singleton channel dim, e.g. (1,H,W) or (H,W,1).
            arr = arr.squeeze()
        elif arr.shape[2] >= 3:
            # Multi-channel image slipped through (e.g. RGB jpg): convert to luminance.
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected image array shape {arr.shape} for '{path}'")

    return arr.astype(np.float32), native_max


def read_image_grayscale(path: str | Path) -> np.ndarray:
    """Read an image from disk as a float32 single-channel array (see
    `read_image_grayscale_meta` if the native bit depth is also needed)."""
    arr, _ = read_image_grayscale_meta(path)
    return arr


def save_image_grayscale(path: str | Path, array: np.ndarray, bit_depth: int = 8) -> None:
    """Save a float array as a grayscale image, clipping to the target bit depth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_val = 255.0 if bit_depth == 8 else 65535.0
    clipped = np.clip(array, 0.0, max_val)
    if bit_depth == 8:
        out = clipped.astype(np.uint8)
        Image.fromarray(out, mode="L").save(path)
    else:
        out = clipped.astype(np.uint16)
        if path.suffix.lower() in {".tif", ".tiff"} and _HAS_TIFFFILE:
            tifffile.imwrite(str(path), out)
        else:
            Image.fromarray(out).save(path)


def save_restored(path: str | Path, array: np.ndarray, native_max: float, reference_ext: Optional[str] = None) -> None:
    """Save a restored (GT-normalized, [0,1]) prediction back to disk in the
    SAME format as the corresponding input file, so `evaluate.py`/`inference.py`
    can preserve filenames exactly (including extension).

    - `.npy` inputs -> saved as float32 `.npy`, clipped to [0, native_max]
      (native_max is 1.0 for pre-normalized float datasets like KLA's, so this
      is just a clip to [0, 1]; the ONLY clipping point in the pipeline).
    - Image inputs (png/jpg/tif/bmp) -> saved via `save_image_grayscale` at
      8-bit or 16-bit depth, chosen from `native_max`.

    `reference_ext` lets the caller pass the source file's extension when
    `path` doesn't already carry the right one; if omitted, `path.suffix` is
    used directly.
    """
    path = Path(path)
    ext = (reference_ext or path.suffix).lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".npy":
        clipped = np.clip(array, 0.0, native_max).astype(np.float32)
        np.save(str(path.with_suffix(".npy")), clipped)
    else:
        bit_depth = 16 if native_max > 255.0 else 8
        save_image_grayscale(path, np.clip(array, 0.0, native_max), bit_depth=bit_depth)


def normalize_degraded(
    image: np.ndarray,
    method: str = "percentile",
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    fixed_range: Tuple[float, float] = (0.0, 255.0),
    eps: float = 1e-6,
) -> Tuple[np.ndarray, NormStats]:
    """Normalize a degraded input image WITHOUT clamping out-of-range values.

    Returns the normalized array (float32, nominally ~[0, 1] but may exceed
    it for outlier pixels) plus the stats needed to invert the transform.
    """
    image = image.astype(np.float32)

    if method == "percentile":
        low = float(np.percentile(image, lower_percentile))
        high = float(np.percentile(image, upper_percentile))
    elif method == "minmax":
        low = float(image.min())
        high = float(image.max())
    elif method == "fixed":
        low, high = float(fixed_range[0]), float(fixed_range[1])
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    denom = max(high - low, eps)
    normalized = (image - low) / denom  # intentionally NOT clipped
    return normalized.astype(np.float32), NormStats(low=low, high=high, method=method)


def denormalize(array: np.ndarray, stats: NormStats, eps: float = 1e-6) -> np.ndarray:
    """Invert `normalize_degraded` back to native intensity space."""
    denom = max(stats.high - stats.low, eps)
    return array * denom + stats.low


def normalize_gt(image: np.ndarray, native_max: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Normalize a clean ground-truth image with a FIXED, bit-depth divisor.

    GT images are clean by construction, so a dataset-consistent fixed scale
    (rather than per-image percentiles) keeps the target space stable, which
    matters for loss computation and for PSNR/SSIM to be comparable across
    the whole validation set.
    """
    if native_max is None:
        native_max = native_max_for_dtype(image.dtype)
    normalized = image.astype(np.float32) / native_max
    return normalized, native_max


def output_to_native(pred_normalized: np.ndarray, native_max: float) -> np.ndarray:
    """Convert model output (in GT-normalized [0,1] space) back to native intensity.

    This is the ONLY point in the pipeline where clipping is applied, and it is
    applied to the RESTORED output (not the raw degraded input), which is safe
    because a correctly restored image should indeed lie within the clean
    dynamic range.
    """
    clipped = np.clip(pred_normalized, 0.0, 1.0)
    return clipped * native_max


def bicubic_resize(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bicubic resize for a single-channel float array (used for the global
    residual skip connection and for the bicubic baseline comparison)."""
    im = Image.fromarray(image.astype(np.float32), mode="F")
    resized = im.resize((out_w, out_h), resample=Image.BICUBIC)
    return np.array(resized, dtype=np.float32)
