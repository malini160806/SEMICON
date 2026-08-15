#!/usr/bin/env python
"""Validate a trained checkpoint against its own held-out validation split.

This is THE script for "how good is my trained model" during development:

    datasets/train/ -> same 80/20 split used during training (same seed) ->
    weights/best_model.pth -> inference on validation images -> compare with
    ground truth -> PSNR / SSIM / LPIPS -> outputs/comparisons/

It never touches datasets/Test_NoisyLR/ (no ground truth exists there --
use evaluate.py for that, which reports timing/throughput only).

By default it reconstructs the validation split from the CHECKPOINT'S OWN
embedded config (train_dir / validation_split / seed / max_pairs) -- not
config/config.yaml, which may have changed since that checkpoint was
trained -- so metrics are always computed on data that specific model
genuinely never saw. Pass --degraded_dir/--gt_dir to point at a different
directory instead.

Metrics (PSNR/SSIM/LPIPS + per-image inference time) are computed over ALL
validation pairs by default (cap with --max_eval_images for speed on a large
validation set); visual comparison grids (Noisy LR -> Restored -> Ground
Truth) are saved for only the first --num_samples of those, to --output_dir
(default outputs/comparisons/). The full per-image CSV (filename, psnr, ssim,
lpips, inference_time_ms) is written to --metrics_csv (default
outputs/validation_metrics.csv).

Usage (PowerShell or bash -- no line-continuation characters needed):
    python scripts/generate_comparisons.py --weights weights/best_model.pth
    python scripts/generate_comparisons.py --weights weights/best_model.pth --num_samples 8 --max_eval_images 200
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.paired_dataset import build_train_val_split  # noqa: E402
from evaluate import load_model_and_config, resolve_device  # noqa: E402
from metrics.metrics import LPIPSMetric, compute_psnr, compute_ssim  # noqa: E402
from utils.image_utils import normalize_degraded, normalize_gt, read_image_grayscale  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a checkpoint on its own held-out split: metrics + comparison grids")
    p.add_argument("--degraded_dir", type=str, default=None, help="Override: explicit degraded dir instead of the checkpoint's own val split")
    p.add_argument("--gt_dir", type=str, default=None, help="Override: explicit GT dir (required if --degraded_dir is given)")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--config", type=str, default="config/config.yaml", help="Fallback config if the checkpoint has none embedded")
    p.add_argument("--output_dir", type=str, default="outputs/comparisons", help="Where comparison PNGs are saved")
    p.add_argument("--metrics_csv", type=str, default="outputs/validation_metrics.csv", help="Where the per-image metrics CSV is saved")
    p.add_argument("--num_samples", type=int, default=8, help="How many validation images get a saved comparison PNG")
    p.add_argument("--max_eval_images", type=int, default=None, help="Cap how many validation pairs are used for the PSNR/SSIM/LPIPS average (default: all)")
    p.add_argument("--crop_size", type=int, default=64, help="Zoomed-crop size in GT pixel space")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def _find_high_detail_crop(gt: np.ndarray, crop: int) -> tuple[int, int]:
    """Pick the crop location with the highest local gradient energy (i.e.
    the most texture/detail), so zoomed comparisons highlight fine structure
    rather than a flat background region."""
    if gt.shape[0] <= crop or gt.shape[1] <= crop:
        return 0, 0
    gy, gx = np.gradient(gt.astype(np.float32))
    energy = gy ** 2 + gx ** 2
    # Downsample the search grid for speed on large images.
    stride = max(crop // 4, 1)
    best_score, best_yx = -1.0, (0, 0)
    for y in range(0, gt.shape[0] - crop, stride):
        for x in range(0, gt.shape[1] - crop, stride):
            score = energy[y:y + crop, x:x + crop].sum()
            if score > best_score:
                best_score, best_yx = score, (y, x)
    return best_yx


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    model, cfg, checkpoint = load_model_and_config(Path(args.weights), args.config, device)
    if isinstance(checkpoint, dict) and checkpoint.get("ema_state_dict"):
        model.load_state_dict(checkpoint["ema_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    norm_cfg = cfg["normalization"] if isinstance(cfg, dict) else cfg.normalization
    norm_method = norm_cfg.get("method", "percentile") if isinstance(norm_cfg, dict) else norm_cfg.method
    lower_pct = norm_cfg.get("lower_percentile", 0.5) if isinstance(norm_cfg, dict) else norm_cfg.lower_percentile
    upper_pct = norm_cfg.get("upper_percentile", 99.5) if isinstance(norm_cfg, dict) else norm_cfg.upper_percentile
    fixed_range = tuple(norm_cfg.get("fixed_range", (0.0, 255.0)) if isinstance(norm_cfg, dict) else norm_cfg.fixed_range)
    eps = norm_cfg.get("eps", 1e-6) if isinstance(norm_cfg, dict) else norm_cfg.eps

    if args.degraded_dir:
        from utils.image_utils import SUPPORTED_EXTENSIONS
        degraded_dir, gt_dir = Path(args.degraded_dir), Path(args.gt_dir)
        files = sorted(p for p in degraded_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        pairs = []
        for d_path in files:
            gt_path = gt_dir / d_path.name
            if not gt_path.exists():
                candidates = list(gt_dir.glob(d_path.stem + ".*"))
                if not candidates:
                    continue
                gt_path = candidates[0]
            pairs.append((d_path, gt_path))
    else:
        # Reconstruct the split from the CHECKPOINT's own embedded config (not
        # config/config.yaml, which may have since changed) so these samples
        # are guaranteed to be from that specific model's held-out validation
        # set -- e.g. a demo/debug checkpoint trained with data.max_pairs set
        # would otherwise leak training images into "validation" comparisons.
        data_cfg = cfg["data"] if isinstance(cfg, dict) else cfg.data
        train_dir = data_cfg.get("train_dir")
        val_ratio = data_cfg.get("validation_split")
        seed = data_cfg.get("seed")
        max_pairs = data_cfg.get("max_pairs")
        split = build_train_val_split(train_dir, val_ratio=val_ratio, seed=seed, max_pairs=max_pairs)
        pairs = split.val_pairs
        print(f"Using validation split from {train_dir} (this checkpoint's own config: "
              f"seed={seed}, val_ratio={val_ratio}, max_pairs={max_pairs}) -- {len(pairs)} val pairs total")

    if not pairs:
        raise RuntimeError("No validation pairs found -- check --degraded_dir/--gt_dir or the checkpoint's data config.")

    eval_pairs = pairs[: args.max_eval_images] if args.max_eval_images else pairs
    visual_stems = {p[0].stem for p in pairs[: args.num_samples]}

    lpips_metric = None
    try:
        lpips_metric = LPIPSMetric(device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"LPIPS unavailable ({exc}); skipping LPIPS metric.")

    # ---- Warm-up (excluded from per-image timing) ----
    warm_degraded = read_image_grayscale(eval_pairs[0][0])
    warm_norm, _ = normalize_degraded(warm_degraded, norm_method, lower_pct, upper_pct, fixed_range, eps)
    warm_x = torch.from_numpy(warm_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        for _ in range(3):
            model(warm_x)
    if device == "cuda":
        torch.cuda.synchronize()

    metric_rows = []
    psnr_vals, ssim_vals, lpips_vals, time_vals_ms = [], [], [], []

    for d_path, gt_path in eval_pairs:
        degraded = read_image_grayscale(d_path)
        gt = read_image_grayscale(gt_path)

        degraded_norm, _ = normalize_degraded(degraded, norm_method, lower_pct, upper_pct, fixed_range, eps)
        gt_norm, _ = normalize_gt(gt)

        x = torch.from_numpy(degraded_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        t_start = time.perf_counter()
        with torch.inference_mode():
            pred_t = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        time_vals_ms.append(elapsed_ms)

        pred = pred_t.squeeze().float().cpu().numpy()
        pred_clamped = np.clip(pred, 0.0, 1.0)

        psnr = compute_psnr(pred_clamped, gt_norm)
        ssim = compute_ssim(pred_clamped, gt_norm)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        lp = float("nan")
        if lpips_metric is not None:
            lp = lpips_metric(torch.from_numpy(pred_clamped), torch.from_numpy(gt_norm))
            lpips_vals.append(lp)
        row = {
            "filename": d_path.name, "psnr": psnr, "ssim": ssim, "lpips": lp,
            "inference_time_ms": elapsed_ms,
        }
        metric_rows.append(row)

        if d_path.stem in visual_stems:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
            for ax, img, title in zip(
                axes, [degraded_norm, pred_clamped, gt_norm],
                [f"Noisy LR ({degraded.shape[1]}x{degraded.shape[0]})",
                 f"Restored ({pred_clamped.shape[1]}x{pred_clamped.shape[0]})",
                 f"Ground Truth ({gt.shape[1]}x{gt.shape[0]})"],
            ):
                ax.imshow(np.clip(img, 0, 1), cmap="gray")
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            fig.suptitle(f"{d_path.name}  |  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  |  Noisy LR -> Restored -> Ground Truth")
            fig.tight_layout()
            fig.savefig(output_dir / f"{d_path.stem}_comparison.png", dpi=150)
            plt.close(fig)

            crop = min(args.crop_size, gt.shape[0] - 1, gt.shape[1] - 1)
            if crop > 4:
                y, x0 = _find_high_detail_crop(gt_norm, crop)
                fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
                axes[0].imshow(pred_clamped[y:y + crop, x0:x0 + crop], cmap="gray")
                axes[0].set_title("Restored (zoom)", fontsize=10)
                axes[1].imshow(gt_norm[y:y + crop, x0:x0 + crop], cmap="gray")
                axes[1].set_title("Ground Truth (zoom)", fontsize=10)
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(output_dir / f"{d_path.stem}_zoom.png", dpi=150)
                plt.close(fig)

            print(f"[{len(metric_rows)}/{len(eval_pairs)}] {d_path.name}: PSNR={psnr:.2f}dB SSIM={ssim:.4f} ({elapsed_ms:.1f}ms) (comparison saved)")
        else:
            print(f"[{len(metric_rows)}/{len(eval_pairs)}] {d_path.name}: PSNR={psnr:.2f}dB SSIM={ssim:.4f} ({elapsed_ms:.1f}ms)")

    csv_path = Path(args.metrics_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "lpips", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(metric_rows)

    psnr_arr = np.array(psnr_vals)
    avg_time_ms = float(np.mean(time_vals_ms))
    total_time_s = float(np.sum(time_vals_ms)) / 1000.0

    print("\n" + "=" * 40)
    print("VALIDATION RESULTS")
    print("=" * 40)
    print(f"Images evaluated: {len(eval_pairs)}")
    print()
    print("PSNR:")
    print(f"  Mean: {psnr_arr.mean():.3f} dB")
    print(f"  Min:  {psnr_arr.min():.3f} dB")
    print(f"  Max:  {psnr_arr.max():.3f} dB")
    print(f"  Std:  {psnr_arr.std():.3f} dB")
    print()
    print("SSIM:")
    print(f"  Mean: {np.mean(ssim_vals):.4f}")
    print()
    print("LPIPS:")
    if lpips_vals:
        print(f"  Mean: {np.mean(lpips_vals):.4f}")
    else:
        print("  Mean: n/a (LPIPS not available)")
    print()
    print("Inference:")
    print(f"  Average/image: {avg_time_ms:.2f} ms")
    print(f"  Total:         {total_time_s:.3f} s")
    print("=" * 40)
    print(f"Per-image metrics: {csv_path}")
    print(f"Comparison images: {output_dir} ({len(visual_stems)} samples)")


if __name__ == "__main__":
    main()
