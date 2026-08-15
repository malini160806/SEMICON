#!/usr/bin/env python
"""Simple bicubic-upsampling baseline: Degraded -> Bicubic -> Output.

Gives a meaningful, zero-training reference point against which the learned
AI restoration model's PSNR/SSIM/LPIPS/inference-time gains are measured.
This is also the first row of the ablation study (scripts/ablation_study.py).

By default this runs on the SAME validation split train.py uses (derived from
config.data.train_dir / validation_split / seed), so the baseline is directly
comparable to the trained model's validation metrics -- pass --degraded_dir/
--gt_dir explicitly to point at a different directory instead.

Usage:
    python scripts/bicubic_baseline.py --config config/config.yaml
    python scripts/bicubic_baseline.py --degraded_dir datasets/train/NoisyLR --gt_dir datasets/train/GT --scale_factor 2
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.paired_dataset import build_train_val_split, validate_and_pair  # noqa: E402
from metrics.metrics import compute_psnr, compute_ssim  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.image_utils import bicubic_resize, normalize_degraded, normalize_gt, read_image_grayscale, save_restored  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bicubic upsampling baseline")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--degraded_dir", type=str, default=None, help="Override: explicit degraded dir instead of the config-driven val split")
    p.add_argument("--gt_dir", type=str, default=None, help="Override: explicit GT dir (required if --degraded_dir is given)")
    p.add_argument("--scale_factor", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="outputs/restored/bicubic_baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config) if Path(args.config).exists() else None
    scale_factor = args.scale_factor or (cfg.data.scale_factor if cfg else 2)
    norm_method = cfg.normalization.method if cfg else "fixed"
    lower_pct = cfg.normalization.lower_percentile if cfg else 0.5
    upper_pct = cfg.normalization.upper_percentile if cfg else 99.5
    fixed_range = tuple(cfg.normalization.fixed_range) if cfg else (0.0, 1.0)
    eps = cfg.normalization.eps if cfg else 1e-6

    if args.degraded_dir:
        report = validate_and_pair(args.degraded_dir, args.gt_dir, scale_factor=scale_factor)
        pairs = report.valid_pairs
    else:
        if cfg is None:
            raise RuntimeError("No --config found and no --degraded_dir/--gt_dir given; cannot locate data.")
        split = build_train_val_split(cfg.data.train_dir, val_ratio=cfg.data.validation_split, seed=cfg.data.seed, scale_factor=scale_factor)
        pairs = split.val_pairs
        print(f"Using validation split from {cfg.data.train_dir} ({len(pairs)} pairs, seed={cfg.data.seed})")

    if not pairs:
        raise RuntimeError("No images found for the bicubic baseline.")

    rows = []
    times = []
    for d_path, g_path in pairs:
        degraded = read_image_grayscale(d_path)
        degraded_norm, _ = normalize_degraded(degraded, norm_method, lower_pct, upper_pct, fixed_range, eps)

        t0 = time.perf_counter()
        out_h, out_w = degraded.shape[0] * scale_factor, degraded.shape[1] * scale_factor
        restored_norm = bicubic_resize(degraded_norm, out_h, out_w)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        save_restored(output_dir / d_path.name, restored_norm, 1.0, reference_ext=d_path.suffix)

        row = {"filename": d_path.name, "inference_time_s": elapsed}
        if g_path is not None:
            gt = read_image_grayscale(g_path)
            gt_norm, _ = normalize_gt(gt)
            pred_clamped = np.clip(restored_norm, 0.0, 1.0)
            row["psnr"] = compute_psnr(pred_clamped, gt_norm)
            row["ssim"] = compute_ssim(pred_clamped, gt_norm)
        rows.append(row)

    print(f"Processed {len(rows)} images | avg time {np.mean(times) * 1000:.3f} ms")
    if "psnr" in rows[0]:
        print(f"Bicubic baseline PSNR: {np.mean([r['psnr'] for r in rows]):.3f} dB")
        print(f"Bicubic baseline SSIM: {np.mean([r['ssim'] for r in rows]):.4f}")

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
