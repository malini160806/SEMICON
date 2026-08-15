"""Paired (degraded, ground_truth) dataset with validation, robust intensity
normalization, and degradation-aware augmentation.

Filenames are matched by stem (filename without extension) so `degraded/001.png`
pairs with `ground_truth/001.tif` etc. -- extensions do not need to match, and
nothing is hard-coded about naming conventions.

The degraded/ground-truth SUBFOLDER NAMES themselves are also auto-detected
(`detect_split_folders`) rather than assumed, since different dataset drops
use different conventions (`degraded`/`ground_truth`, `noisy`/`clean`,
`NoisyLR`/`GT`, `LR`/`HR`, ...). See README.md "Dataset Structure".
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.augmentations import (
    AugmentConfig,
    add_synthetic_degradation,
    apply_geometric_pair,
    random_paired_crop,
)
from utils.image_utils import (
    SUPPORTED_EXTENSIONS,
    normalize_degraded,
    normalize_gt,
    read_image_grayscale,
)

logger = logging.getLogger("kla_restoration")


@dataclass
class PairValidationReport:
    num_degraded_found: int = 0
    num_gt_found: int = 0
    num_matched_pairs: int = 0
    missing_gt: List[str] = field(default_factory=list)
    missing_degraded: List[str] = field(default_factory=list)
    corrupted_files: List[str] = field(default_factory=list)
    resolution_mismatches: List[str] = field(default_factory=list)
    non_grayscale_warnings: List[str] = field(default_factory=list)
    valid_pairs: List[Tuple[Path, Path]] = field(default_factory=list)
    detected_scales: set = field(default_factory=set)

    def summary(self) -> str:
        lines = [
            "Dataset validation report",
            "=" * 40,
            f"Degraded files found:   {self.num_degraded_found}",
            f"GT files found:         {self.num_gt_found}",
            f"Matched valid pairs:    {self.num_matched_pairs}",
            f"Missing GT:             {len(self.missing_gt)}",
            f"Missing degraded:       {len(self.missing_degraded)}",
            f"Corrupted files:        {len(self.corrupted_files)}",
            f"Resolution mismatches:  {len(self.resolution_mismatches)}",
            f"Detected scale factors: {sorted(self.detected_scales) if self.detected_scales else 'n/a'}",
        ]
        return "\n".join(lines)


def _list_images(directory: Path) -> Dict[str, Path]:
    """Map filename stem -> path for all supported image files in a directory."""
    result: Dict[str, Path] = {}
    if not directory.exists():
        return result
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            result[p.stem] = p
    return result


# Keyword vocabularies used to auto-detect which subfolder of a train split
# holds degraded/noisy/low-res input vs. clean/ground-truth/high-res target.
# Matched case-insensitively against the FULL subfolder name (not substrings
# of unrelated words), so e.g. "GT" matches but "Digital" does not.
_DEGRADED_KEYWORDS = {"degraded", "noisy", "noisylr", "lr", "low", "input", "corrupted", "distorted"}
_GT_KEYWORDS = {"ground_truth", "groundtruth", "gt", "clean", "hr", "high", "target", "label", "reference"}


def detect_split_folders(train_dir: str | Path) -> Tuple[Path, Path, str]:
    """Auto-detect which immediate subfolders of `train_dir` hold degraded
    (noisy/low-res) vs. ground-truth (clean/high-res) images, from common
    naming conventions -- without assuming a fixed folder-name pair.

    Returns (degraded_dir, gt_dir, layout_name).

    Raises RuntimeError with a descriptive message if detection is ambiguous,
    so the caller can be told exactly what was found instead of silently
    guessing wrong.
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise RuntimeError(f"Training directory not found: {train_dir}")

    subdirs = [p for p in sorted(train_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    subdir_names = [p.name for p in subdirs]

    degraded_dir, gt_dir = None, None
    for p in subdirs:
        key = p.name.lower().replace("-", "_")
        if key in _DEGRADED_KEYWORDS and degraded_dir is None:
            degraded_dir = p
        elif key in _GT_KEYWORDS and gt_dir is None:
            gt_dir = p

    if degraded_dir is not None and gt_dir is not None:
        layout = f"{degraded_dir.name}/{gt_dir.name}"
        logger.info("Detected dataset layout under %s: %s", train_dir, layout)
        return degraded_dir, gt_dir, layout

    raise RuntimeError(
        f"Could not auto-detect degraded/ground-truth subfolders under '{train_dir}'. "
        f"Found subfolders: {subdir_names}. Recognized degraded-side names: "
        f"{sorted(_DEGRADED_KEYWORDS)}; recognized GT-side names: {sorted(_GT_KEYWORDS)}. "
        f"Rename your subfolders to one of these, or pass explicit "
        f"degraded_dir/gt_dir paths directly."
    )


def split_pairs(
    pairs: List[Tuple[Path, Optional[Path]]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Tuple[Path, Optional[Path]]]]:
    """Deterministically split a list of (degraded, gt) pairs into train/val.

    Sorted by filename first (so the split is independent of filesystem
    iteration order), then shuffled with a seeded RNG and sliced -- every
    pair lands in exactly one split, guaranteeing no leakage between them.
    """
    ordered = sorted(pairs, key=lambda pair: str(pair[0]))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = int(round(len(ordered) * val_ratio))
    val_pairs = ordered[:n_val]
    train_pairs = ordered[n_val:]
    return train_pairs, val_pairs


def validate_and_pair(
    degraded_dir: str | Path,
    gt_dir: Optional[str | Path],
    scale_factor: Optional[int] = None,
    strict_resolution: bool = False,
) -> PairValidationReport:
    """Validate a dataset split and build the list of matched (degraded, gt) pairs.

    If `gt_dir` is None (test-time inference on degraded-only data), every
    valid degraded image is treated as a "pair" with `gt` set to None.

    The scale factor is DETECTED per-pair from each image's actual shape
    (`report.detected_scales`) rather than assumed uniform across the
    dataset. `scale_factor`, if given, is only used to flag pairs whose
    detected ratio doesn't match the expected one (recorded in
    `resolution_mismatches`; dropped only if `strict_resolution=True`) --
    it never overrides what was actually measured from the files.
    """
    degraded_dir = Path(degraded_dir)
    report = PairValidationReport()

    degraded_files = _list_images(degraded_dir)
    report.num_degraded_found = len(degraded_files)

    if gt_dir is None:
        for stem, dpath in degraded_files.items():
            try:
                img = read_image_grayscale(dpath)
                if img.ndim != 2:
                    report.non_grayscale_warnings.append(str(dpath))
            except (FileNotFoundError, ValueError) as exc:
                report.corrupted_files.append(f"{dpath}: {exc}")
                continue
            report.valid_pairs.append((dpath, None))
        report.num_matched_pairs = len(report.valid_pairs)
        return report

    gt_dir = Path(gt_dir)
    gt_files = _list_images(gt_dir)
    report.num_gt_found = len(gt_files)

    all_stems = sorted(set(degraded_files) | set(gt_files))
    for stem in all_stems:
        dpath = degraded_files.get(stem)
        gpath = gt_files.get(stem)

        if dpath is None:
            report.missing_degraded.append(stem)
            continue
        if gpath is None:
            report.missing_gt.append(stem)
            continue

        try:
            d_img = read_image_grayscale(dpath)
            g_img = read_image_grayscale(gpath)
        except (FileNotFoundError, ValueError) as exc:
            report.corrupted_files.append(f"{stem}: {exc}")
            continue

        if d_img.ndim != 2:
            report.non_grayscale_warnings.append(str(dpath))
        if g_img.ndim != 2:
            report.non_grayscale_warnings.append(str(gpath))

        # Detect this pair's actual scale factor from its measured shapes,
        # rather than assuming every pair in the dataset shares one scale.
        h_ratio = g_img.shape[0] / d_img.shape[0] if d_img.shape[0] else 0
        w_ratio = g_img.shape[1] / d_img.shape[1] if d_img.shape[1] else 0
        if h_ratio == w_ratio and h_ratio == int(h_ratio) and h_ratio >= 1:
            report.detected_scales.add(int(h_ratio))
        else:
            report.resolution_mismatches.append(
                f"{stem}: non-integer or non-uniform scale, degraded {d_img.shape} vs gt {g_img.shape}"
            )
            if strict_resolution:
                continue

        if scale_factor is not None and int(h_ratio) != scale_factor:
            msg = f"{stem}: degraded {d_img.shape} * expected scale {scale_factor} != gt {g_img.shape}"
            report.resolution_mismatches.append(msg)
            if strict_resolution:
                continue

        report.valid_pairs.append((dpath, gpath))

    report.num_matched_pairs = len(report.valid_pairs)
    return report


class PairedRestorationDataset(Dataset):
    """Training/validation dataset yielding normalized (degraded, gt) tensor pairs.

    Normalization strategy: see utils/image_utils.py docstring. GT uses a fixed
    bit-depth divisor; degraded input uses per-image robust percentile scaling
    WITHOUT clamping, so out-of-range speckle spikes remain visible to the model.
    """

    def __init__(
        self,
        degraded_dir: Optional[str | Path] = None,
        gt_dir: Optional[str | Path] = None,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
        scale_factor: int = 2,
        patch_size: int = 128,
        augment: bool = True,
        augment_cfg: Optional[AugmentConfig] = None,
        norm_method: str = "percentile",
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        fixed_range: Tuple[float, float] = (0.0, 255.0),
        eps: float = 1e-6,
        strict_resolution: bool = False,
        seed: int = 42,
        split_name: str = "dataset",
    ):
        self.degraded_dir = Path(degraded_dir) if degraded_dir is not None else None
        self.gt_dir = Path(gt_dir) if gt_dir is not None else None
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.norm_method = norm_method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.fixed_range = fixed_range
        self.eps = eps
        self.rng = np.random.default_rng(seed)

        if pairs is not None:
            # Pairs were already validated + split upstream (e.g. by
            # build_train_val_split), so no leakage-prone re-scanning here.
            self.pairs: List[Tuple[Path, Path]] = pairs
        else:
            if self.degraded_dir is None or self.gt_dir is None:
                raise ValueError("PairedRestorationDataset requires either `pairs` or both `degraded_dir`/`gt_dir`.")
            report = validate_and_pair(
                self.degraded_dir, self.gt_dir, scale_factor=scale_factor, strict_resolution=strict_resolution
            )
            if report.num_matched_pairs == 0:
                raise RuntimeError(
                    f"No valid pairs found between '{degraded_dir}' and '{gt_dir}'.\n{report.summary()}"
                )
            if report.missing_gt or report.corrupted_files:
                logger.warning(
                    "Dataset validation found issues (missing_gt=%d, corrupted=%d, "
                    "res_mismatch=%d). See validate_dataset.py for full detail.",
                    len(report.missing_gt), len(report.corrupted_files), len(report.resolution_mismatches),
                )
            self.pairs = report.valid_pairs

        logger.info("Loaded %d %s pairs", len(self.pairs), split_name)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d_path, g_path = self.pairs[idx]
        degraded = read_image_grayscale(d_path)
        gt = read_image_grayscale(g_path)

        # Detect this pair's own scale factor from its measured shapes rather
        # than assuming every pair shares self.scale_factor.
        pair_scale = self.scale_factor
        if degraded.shape[0] > 0 and gt.shape[0] % degraded.shape[0] == 0:
            pair_scale = gt.shape[0] // degraded.shape[0]

        if self.augment and self.augment_cfg.enabled and self.augment_cfg.random_crop:
            if degraded.shape[0] >= self.patch_size and degraded.shape[1] >= self.patch_size:
                degraded, gt = random_paired_crop(degraded, gt, self.patch_size, pair_scale, self.rng)

        if self.augment and self.augment_cfg.enabled:
            degraded, gt = apply_geometric_pair(degraded, gt, self.augment_cfg, self.rng)

        degraded_norm, _stats = normalize_degraded(
            degraded, self.norm_method, self.lower_percentile, self.upper_percentile, self.fixed_range, self.eps
        )
        gt_norm, _gt_max = normalize_gt(gt)

        if self.augment and self.augment_cfg.enabled:
            if self.rng.random() < self.augment_cfg.synthetic_prob:
                degraded_norm = add_synthetic_degradation(degraded_norm, self.augment_cfg, self.rng)

        degraded_t = torch.from_numpy(degraded_norm).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_norm).unsqueeze(0).float()
        return {"degraded": degraded_t, "gt": gt_t, "filename": d_path.name}


@dataclass
class TrainValSplitResult:
    train_pairs: List[Tuple[Path, Path]]
    val_pairs: List[Tuple[Path, Path]]
    degraded_dir: Path
    gt_dir: Path
    layout: str
    report: PairValidationReport


def build_train_val_split(
    train_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    scale_factor: Optional[int] = None,
    strict_resolution: bool = False,
    max_pairs: Optional[int] = None,
) -> TrainValSplitResult:
    """One-stop dataset builder for a `datasets/train/` style directory:

    1. Auto-detects which subfolder is degraded vs. ground-truth
       (`detect_split_folders` -- handles `degraded/ground_truth`,
       `noisy/clean`, `NoisyLR/GT`, `LR/HR`, etc.).
    2. Validates and pairs every file by stem, detecting each pair's actual
       scale factor from its measured shape (no fixed-resolution assumption).
    3. Splits the matched pairs into train/val with a seeded, deterministic
       shuffle (`split_pairs`) -- every pair goes to exactly one split, so
       there is no leakage between train and validation.

    `max_pairs` (config: `data.max_pairs`, default None = use everything)
    optionally caps the total pool BEFORE splitting, for fast iteration on a
    subset of a large dataset (e.g. quick smoke runs on CPU) -- the cap is
    applied to a seeded shuffle, so it's deterministic and still leak-free.
    """
    degraded_dir, gt_dir, layout = detect_split_folders(train_dir)
    report = validate_and_pair(degraded_dir, gt_dir, scale_factor=scale_factor, strict_resolution=strict_resolution)
    if report.num_matched_pairs == 0:
        raise RuntimeError(f"No valid pairs found under '{train_dir}' (layout: {layout}).\n{report.summary()}")

    pool = report.valid_pairs
    if max_pairs is not None and max_pairs < len(pool):
        ordered = sorted(pool, key=lambda pair: str(pair[0]))
        random.Random(seed).shuffle(ordered)
        pool = ordered[:max_pairs]
        logger.info("Capped dataset pool to %d/%d pairs (data.max_pairs)", len(pool), report.num_matched_pairs)

    train_pairs, val_pairs = split_pairs(pool, val_ratio, seed)
    logger.info(
        "Train/val split (seed=%d, val_ratio=%.2f): %d train, %d val, from %d total pairs (layout: %s)",
        seed, val_ratio, len(train_pairs), len(val_pairs), len(pool), layout,
    )
    return TrainValSplitResult(
        train_pairs=train_pairs, val_pairs=val_pairs,
        degraded_dir=degraded_dir, gt_dir=gt_dir, layout=layout, report=report,
    )


class InferenceDataset(Dataset):
    """Degraded-only dataset for evaluate.py / inference.py (no ground truth required).

    Optionally pairs with ground truth if a `gt_dir` is supplied, so evaluate.py
    can compute metrics when GT is available while still working with pure
    degraded-only test folders.
    """

    def __init__(
        self,
        degraded_dir: str | Path,
        gt_dir: Optional[str | Path] = None,
        norm_method: str = "percentile",
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        fixed_range: Tuple[float, float] = (0.0, 255.0),
        eps: float = 1e-6,
    ):
        self.degraded_dir = Path(degraded_dir)
        self.gt_dir = Path(gt_dir) if gt_dir is not None else None
        self.norm_method = norm_method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.fixed_range = fixed_range
        self.eps = eps

        report = validate_and_pair(self.degraded_dir, self.gt_dir)
        if report.num_matched_pairs == 0:
            raise RuntimeError(f"No valid images found in '{degraded_dir}'.\n{report.summary()}")
        self.pairs = report.valid_pairs
        logger.info("Found %d images for inference in %s", len(self.pairs), degraded_dir)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        d_path, g_path = self.pairs[idx]
        degraded = read_image_grayscale(d_path)
        degraded_norm, stats = normalize_degraded(
            degraded, self.norm_method, self.lower_percentile, self.upper_percentile, self.fixed_range, self.eps
        )
        degraded_t = torch.from_numpy(degraded_norm).unsqueeze(0).float()

        item: Dict[str, object] = {
            "degraded": degraded_t,
            "filename": d_path.name,
            "orig_h": degraded.shape[0],
            "orig_w": degraded.shape[1],
            "norm_low": stats.low,
            "norm_high": stats.high,
        }
        if g_path is not None:
            gt = read_image_grayscale(g_path)
            gt_norm, gt_max = normalize_gt(gt)
            item["gt"] = torch.from_numpy(gt_norm).unsqueeze(0).float()
            item["gt_max"] = gt_max
        return item
