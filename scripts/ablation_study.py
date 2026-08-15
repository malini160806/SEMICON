#!/usr/bin/env python
"""Ablation study harness: compares multiple restoration variants on the same
validation set and prints/saves a single comparison table:

    Model | PSNR | SSIM | LPIPS | Inference Time

This does NOT fabricate results -- it is a runnable harness. Populate the
`--model` list with whatever checkpoints you have actually trained. Useful
variants for the hackathon "Innovation" slide, and how to produce each one
with the existing config/config.yaml:

  1. Bicubic baseline        -> use `name=bicubic` (no weights/training needed)
  2. Denoising-only          -> train with model.scale_factor=1 (no SR tail
                                 upsampling) on the same noisy/clean pairs
  3. Super-resolution-only   -> train on clean-but-downsampled pairs (disable
                                 augmentation.gaussian_noise / speckle_noise)
  4. Unified restoration     -> the default config.yaml as shipped
  5. Unified + proposed loss -> same as (4) but with loss.ssim / gradient /
                                 frequency weights enabled (already default);
                                 compare against a run with those weights
                                 zeroed out to isolate their contribution

By default this runs on the SAME validation split train.py uses (derived from
config.data.train_dir / validation_split / seed) -- pass --degraded_dir/
--gt_dir explicitly to point at a different directory instead.

Usage:
    python scripts/ablation_study.py --config config/config.yaml \\
        --model bicubic:none \\
        --model unified:weights/best_model.pth \\
        --output_csv outputs/ablation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.paired_dataset import build_train_val_split, validate_and_pair  # noqa: E402
from evaluate import load_model_and_config, resolve_device  # noqa: E402
from metrics.metrics import LPIPSMetric, compute_psnr, compute_ssim, count_parameters  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.image_utils import bicubic_resize, normalize_degraded, normalize_gt, read_image_grayscale  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation comparison across restoration variants")
    p.add_argument("--degraded_dir", type=str, default=None, help="Override: explicit degraded dir instead of the config-driven val split")
    p.add_argument("--gt_dir", type=str, default=None, help="Override: explicit GT dir (required if --degraded_dir is given)")
    p.add_argument("--scale_factor", type=int, default=None)
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--model", action="append", required=True,
                    help="name:weights_path pair, repeatable. Use name:bicubic (or path 'none') for the baseline.")
    p.add_argument("--output_csv", type=str, default="outputs/ablation_results.csv")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max_images", type=int, default=None)
    return p.parse_args()


def evaluate_variant(name: str, weights_path: str, pairs: list, scale_factor: int, config_path: str, device: str,
                      lpips_metric: LPIPSMetric | None,
                      default_norm: tuple) -> dict:
    is_bicubic = weights_path.lower() in ("none", "bicubic")
    model = None
    n_params = 0
    norm_method, lower_pct, upper_pct, fixed_range, eps = default_norm
    if not is_bicubic:
        model, variant_cfg, checkpoint = load_model_and_config(Path(weights_path), config_path, device)
        if isinstance(checkpoint, dict) and checkpoint.get("ema_state_dict"):
            model.load_state_dict(checkpoint["ema_state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        n_params = count_parameters(model)
        # Use THIS checkpoint's own training-time normalization, not the caller's default,
        # so each variant is evaluated exactly the way it was trained.
        norm_cfg = variant_cfg["normalization"] if isinstance(variant_cfg, dict) else variant_cfg.normalization
        norm_method = norm_cfg.get("method", norm_method) if isinstance(norm_cfg, dict) else norm_cfg.method
        lower_pct = norm_cfg.get("lower_percentile", lower_pct) if isinstance(norm_cfg, dict) else norm_cfg.lower_percentile
        upper_pct = norm_cfg.get("upper_percentile", upper_pct) if isinstance(norm_cfg, dict) else norm_cfg.upper_percentile
        fixed_range = tuple(norm_cfg.get("fixed_range", fixed_range) if isinstance(norm_cfg, dict) else norm_cfg.fixed_range)
        eps = norm_cfg.get("eps", eps) if isinstance(norm_cfg, dict) else norm_cfg.eps

    psnr_vals, ssim_vals, lpips_vals, times = [], [], [], []
    for d_path, g_path in pairs:
        degraded = read_image_grayscale(d_path)
        gt = read_image_grayscale(g_path)
        degraded_norm, _ = normalize_degraded(degraded, norm_method, lower_pct, upper_pct, fixed_range, eps)
        gt_norm, _ = normalize_gt(gt)

        t0 = time.perf_counter()
        if is_bicubic:
            pred = bicubic_resize(degraded_norm, gt.shape[0], gt.shape[1])
        else:
            x = torch.from_numpy(degraded_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.inference_mode():
                pred = model(x).squeeze().float().cpu().numpy()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        pred_clamped = np.clip(pred, 0.0, 1.0)
        psnr_vals.append(compute_psnr(pred_clamped, gt_norm))
        ssim_vals.append(compute_ssim(pred_clamped, gt_norm))
        if lpips_metric is not None:
            lpips_vals.append(lpips_metric(torch.from_numpy(pred_clamped), torch.from_numpy(gt_norm)))

    return {
        "model": name,
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
        "inference_time_ms": float(np.mean(times)) * 1000,
        "params_m": n_params / 1e6,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = load_config(args.config) if Path(args.config).exists() else None
    scale_factor = args.scale_factor or (cfg.data.scale_factor if cfg else 2)
    default_norm = (
        (cfg.normalization.method, cfg.normalization.lower_percentile, cfg.normalization.upper_percentile,
         tuple(cfg.normalization.fixed_range), cfg.normalization.eps)
        if cfg else ("fixed", 0.5, 99.5, (0.0, 1.0), 1e-6)
    )

    if args.degraded_dir:
        report = validate_and_pair(args.degraded_dir, args.gt_dir, scale_factor=scale_factor)
        pairs = report.valid_pairs
    else:
        if cfg is None:
            raise RuntimeError("No --config found and no --degraded_dir/--gt_dir given; cannot locate data.")
        split = build_train_val_split(cfg.data.train_dir, val_ratio=cfg.data.validation_split, seed=cfg.data.seed, scale_factor=scale_factor)
        pairs = split.val_pairs
        print(f"Using validation split from {cfg.data.train_dir} ({len(pairs)} pairs, seed={cfg.data.seed})")
    if args.max_images:
        pairs = pairs[: args.max_images]

    lpips_metric = None
    try:
        lpips_metric = LPIPSMetric(device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"LPIPS unavailable ({exc}); skipping LPIPS column.")

    results = []
    for spec in args.model:
        if ":" not in spec:
            raise ValueError(f"--model must be name:weights_path, got '{spec}'")
        name, weights_path = spec.split(":", 1)
        print(f"\nEvaluating variant: {name} ({weights_path})")
        result = evaluate_variant(
            name, weights_path, pairs, scale_factor,
            args.config, device, lpips_metric, default_norm,
        )
        results.append(result)
        print(f"  PSNR={result['psnr']:.3f}  SSIM={result['ssim']:.4f}  "
              f"LPIPS={result['lpips']:.4f}  time={result['inference_time_ms']:.2f}ms  "
              f"params={result['params_m']:.2f}M")

    print("\n" + "=" * 78)
    print(f"{'Model':<22}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}{'Time(ms)':>14}{'Params(M)':>12}")
    print("=" * 78)
    for r in results:
        print(f"{r['model']:<22}{r['psnr']:>10.3f}{r['ssim']:>10.4f}{r['lpips']:>10.4f}"
              f"{r['inference_time_ms']:>14.2f}{r['params_m']:>12.2f}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    main()
