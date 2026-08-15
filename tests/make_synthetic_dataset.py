#!/usr/bin/env python
"""Generate a tiny synthetic paired dataset for smoke-testing the pipeline.

Not part of the production pipeline -- used only to verify train.py /
evaluate.py / validate_dataset.py run correctly end-to-end before real KLA
data is available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_gt_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic "semiconductor-like" pattern: grid lines + rectangles."""
    img = Image.new("L", (size, size), color=40)
    draw = ImageDraw.Draw(img)
    step = size // 8
    for i in range(0, size, step):
        draw.line([(i, 0), (i, size)], fill=200, width=1)
        draw.line([(0, i), (size, i)], fill=200, width=1)
    for _ in range(5):
        x0, y0 = rng.integers(0, size - size // 4, size=2)
        w, h = rng.integers(size // 8, size // 4, size=2)
        fill = int(rng.integers(120, 255))
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=fill)
    return np.array(img, dtype=np.float32)


def degrade(gt: np.ndarray, lr_size: int, rng: np.random.Generator) -> np.ndarray:
    lr = np.array(Image.fromarray(gt.astype(np.uint8)).resize((lr_size, lr_size), Image.BICUBIC), dtype=np.float32)
    speckle = lr * rng.normal(0, 0.15, lr.shape).astype(np.float32)
    gauss = rng.normal(0, 8.0, lr.shape).astype(np.float32)
    degraded = lr + speckle + gauss
    return degraded  # intentionally NOT clipped -- may exceed [0, 255]


def build_split(root: Path, split: str, n: int, hr_size: int, scale: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    gt_dir = root / split / "ground_truth"
    deg_dir = root / split / "degraded"
    gt_dir.mkdir(parents=True, exist_ok=True)
    deg_dir.mkdir(parents=True, exist_ok=True)
    lr_size = hr_size // scale

    for i in range(n):
        gt = make_gt_image(hr_size, rng)
        degraded = degrade(gt, lr_size, rng)
        Image.fromarray(gt.astype(np.uint8), mode="L").save(gt_dir / f"{i:03d}.png")
        clipped_for_save = np.clip(degraded, 0, 255).astype(np.uint8)
        Image.fromarray(clipped_for_save, mode="L").save(deg_dir / f"{i:03d}.png")


def main() -> None:
    """Mirrors the real KLA layout: ONE `train/` dir (split 80/20 internally
    by build_train_val_split, not pre-split here) + a degraded-only test dir.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="tests/synthetic_dataset")
    p.add_argument("--hr_size", type=int, default=64)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n_train_total", type=int, default=16, help="Combined pool later split 80/20 into train/val")
    p.add_argument("--n_test", type=int, default=3)
    args = p.parse_args()

    root = Path(args.root)
    build_split(root, "train", args.n_train_total, args.hr_size, args.scale, seed=1)

    # Test split: degraded only, no ground_truth dir -- exercises evaluate.py's GT-optional path.
    test_dir = root / "Test_NoisyLR"
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(args.n_test):
        gt = make_gt_image(args.hr_size, rng)
        degraded = degrade(gt, args.hr_size // args.scale, rng)
        Image.fromarray(np.clip(degraded, 0, 255).astype(np.uint8), mode="L").save(test_dir / f"{i:03d}.png")

    print(f"Synthetic dataset created at {root} (train/ will be split 80/20 by build_train_val_split)")


if __name__ == "__main__":
    main()
