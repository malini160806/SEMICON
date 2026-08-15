#!/usr/bin/env python
"""Standalone dataset validation / inspection CLI.

Inspects `datasets/train/` (auto-detecting its degraded/ground-truth subfolder
layout -- e.g. NoisyLR/GT, degraded/ground_truth, noisy/clean, LR/HR -- rather
than assuming one) and `datasets/Test_NoisyLR/` (degraded-only, inference-only,
never mixed into train/val), and prints a DATASET SUMMARY covering pairing,
corruption, resolution/scale detection, format, and intensity statistics.

Usage:
    python validate_dataset.py --config config/config.yaml

    # Or point at directories directly, bypassing the config:
    python validate_dataset.py --train_dir datasets/train --test_dir datasets/Test_NoisyLR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from datasets.paired_dataset import (
    build_train_val_split,
    detect_split_folders,
    validate_and_pair,
)
from utils.config import load_config
from utils.image_utils import read_image_grayscale_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect and validate the KLA restoration dataset")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--train_dir", type=str, default=None, help="Override config.data.train_dir")
    p.add_argument("--test_dir", type=str, default=None, help="Override config.data.test_dir")
    p.add_argument("--validation_split", type=float, default=None, help="Override config.data.validation_split")
    p.add_argument("--seed", type=int, default=None, help="Override config.data.seed")
    p.add_argument("--max_stats_images", type=int, default=500,
                    help="Cap how many images per group are sampled for intensity statistics (speed). Use 0 for all.")
    return p.parse_args()


def _resolution_summary(shapes: Iterable[tuple[int, int]]) -> str:
    shapes = list(shapes)
    if not shapes:
        return "n/a"
    unique = sorted(set(shapes))
    if len(unique) == 1:
        h, w = unique[0]
        return f"{h}x{w}"
    return f"{len(unique)} distinct resolutions, e.g. " + ", ".join(f"{h}x{w}" for h, w in unique[:5])


def _intensity_stats(paths: list[Path], max_images: int) -> dict:
    if max_images and len(paths) > max_images:
        rng = np.random.default_rng(0)
        sample = [paths[i] for i in rng.choice(len(paths), size=max_images, replace=False)]
    else:
        sample = paths

    mins, maxs, means, stds, shapes, exts = [], [], [], [], [], set()
    corrupted = []
    for p in sample:
        try:
            arr, _native_max = read_image_grayscale_meta(p)
        except (FileNotFoundError, ValueError) as exc:
            corrupted.append(f"{p}: {exc}")
            continue
        mins.append(float(arr.min()))
        maxs.append(float(arr.max()))
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        shapes.append(arr.shape)
        exts.add(p.suffix.lower())

    if not mins:
        return {"n": 0, "corrupted": corrupted}
    return {
        "n": len(mins),
        "sampled_of": len(paths),
        "min": min(mins), "max": max(maxs),
        "mean": float(np.mean(means)), "std": float(np.mean(stds)),
        "shapes": shapes, "extensions": exts, "corrupted": corrupted,
    }


def main() -> None:
    args = parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir
    val_split = args.validation_split
    seed = args.seed
    scale_factor = None

    if Path(args.config).exists():
        cfg = load_config(args.config)
        train_dir = train_dir or cfg.data.train_dir
        test_dir = test_dir or cfg.data.test_dir
        val_split = val_split if val_split is not None else cfg.data.validation_split
        seed = seed if seed is not None else cfg.data.seed
        scale_factor = cfg.data.scale_factor
    else:
        train_dir = train_dir or "datasets/train"
        test_dir = test_dir or "datasets/Test_NoisyLR"
        val_split = val_split if val_split is not None else 0.2
        seed = seed if seed is not None else 42

    print(f"Inspecting train_dir = {train_dir}")
    print(f"Inspecting test_dir  = {test_dir}\n")

    # ---- Detect train layout + pair + split ----
    try:
        degraded_dir, gt_dir, layout = detect_split_folders(train_dir)
        print(f"Detected training layout: {layout}  (degraded='{degraded_dir.name}', gt='{gt_dir.name}')")
    except RuntimeError as exc:
        print(f"FAILED to detect train layout: {exc}")
        sys.exit(2)

    report = validate_and_pair(degraded_dir, gt_dir, scale_factor=scale_factor)
    split = None
    if report.num_matched_pairs > 0:
        split = build_train_val_split(train_dir, val_ratio=val_split, seed=seed, scale_factor=scale_factor)

    # ---- Detect test set (degraded-only) ----
    test_report = None
    if Path(test_dir).exists():
        test_report = validate_and_pair(test_dir, None)
    else:
        print(f"NOTE: test_dir '{test_dir}' does not exist -- skipping test inspection.\n")

    # ---- Resolutions ----
    train_degraded_paths = [p[0] for p in report.valid_pairs]
    train_gt_paths = [p[1] for p in report.valid_pairs]
    test_paths = [p[0] for p in test_report.valid_pairs] if test_report else []

    def sample_shapes(paths: list[Path], n: int = 50) -> list[tuple[int, int]]:
        if not paths:
            return []
        idx = np.random.default_rng(0).choice(len(paths), size=min(n, len(paths)), replace=False)
        shapes = []
        for i in idx:
            try:
                arr, _ = read_image_grayscale_meta(paths[i])
                shapes.append(arr.shape)
            except (FileNotFoundError, ValueError):
                continue
        return shapes

    train_res = _resolution_summary(sample_shapes(train_degraded_paths))
    gt_res = _resolution_summary(sample_shapes(train_gt_paths))
    test_res = _resolution_summary(sample_shapes(test_paths))

    formats = sorted({p.suffix.lower() for p in (train_degraded_paths + train_gt_paths + test_paths)})

    n_train = len(split.train_pairs) if split else 0
    n_val = len(split.val_pairs) if split else 0
    n_test = test_report.num_matched_pairs if test_report else 0

    # ---- Required summary block ----
    print("=" * 40)
    print("DATASET SUMMARY")
    print("=" * 40)
    print()
    print(f"Training images:        {n_train}")
    print(f"Validation images:      {n_val}")
    print(f"Test images:            {n_test}")
    print()
    print(f"Training resolution:    {train_res}")
    print(f"Ground truth resolution:{gt_res}")
    print(f"Test resolution:        {test_res}")
    print()
    print(f"Image format:           {', '.join(formats) if formats else 'n/a'}")
    print("Channels:               1 (grayscale)")
    print()
    print(f"Paired images:          {report.num_matched_pairs}")
    print(f"Missing pairs:          {len(report.missing_gt) + len(report.missing_degraded)}")
    print(f"Corrupted images:       {len(report.corrupted_files) + (len(test_report.corrupted_files) if test_report else 0)}")
    print()
    print("=" * 40)

    # ---- Extra detail: pairing / resolution issues ----
    if report.missing_gt:
        print(f"\nMissing GT for {len(report.missing_gt)} file(s), e.g.: {report.missing_gt[:5]}")
    if report.missing_degraded:
        print(f"Missing degraded for {len(report.missing_degraded)} file(s), e.g.: {report.missing_degraded[:5]}")
    if report.corrupted_files:
        print(f"Corrupted training files: {report.corrupted_files[:5]}")
    if report.resolution_mismatches:
        print(f"Resolution/scale anomalies (first 5): {report.resolution_mismatches[:5]}")
    print(f"\nDetected scale factor(s) in training pairs: {sorted(report.detected_scales) if report.detected_scales else 'n/a'}")
    if split:
        overlap = set(str(p[0]) for p in split.train_pairs) & set(str(p[0]) for p in split.val_pairs)
        print(f"Train/val leakage check: {'FAIL' if overlap else 'PASS (no overlap)'}  (seed={seed}, val_ratio={val_split})")

    if test_report and test_report.corrupted_files:
        print(f"Corrupted test files: {test_report.corrupted_files[:5]}")

    # ---- Intensity statistics (section 10) ----
    max_stats = args.max_stats_images if args.max_stats_images > 0 else None
    print("\n--- Intensity statistics ---")
    for label, paths in [
        ("Training degraded", train_degraded_paths),
        ("Training ground truth", train_gt_paths),
        ("Test degraded", test_paths),
    ]:
        stats = _intensity_stats(paths, max_stats or len(paths))
        if stats["n"] == 0:
            print(f"{label}: no readable images")
            continue
        sampled_note = f" (sampled {stats['n']}/{stats['sampled_of']})" if stats.get("sampled_of", stats["n"]) != stats["n"] else ""
        print(f"{label}{sampled_note}: min={stats['min']:.4f} max={stats['max']:.4f} "
              f"mean={stats['mean']:.4f} std={stats['std']:.4f}")
        if stats["min"] < 0 or stats["max"] > 1.0:
            print(f"  (values extend outside [0,1] -- expected for noisy input; NOT clipped, see README 'Intensity Handling')")

    exit_code = 0 if (report.num_matched_pairs > 0) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
