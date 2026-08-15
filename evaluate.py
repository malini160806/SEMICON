#!/usr/bin/env python
"""Standalone evaluation / restoration script. Works as-is, no code edits required.

    python evaluate.py \\
        --input_dir ./datasets/Test_NoisyLR \\
        --output_dir ./outputs/test_restored \\
        --weights ./weights/best_model.pth

`datasets/Test_NoisyLR/` has no ground-truth folder, so this reports pure
inference statistics (images processed, timing, throughput) rather than
PSNR/SSIM/LPIPS -- pass --gt_dir explicitly if you do have a paired
ground-truth directory (e.g. for re-checking validation-split performance).

What it does:
  1. Loads the trained model from `--weights` (architecture is read from the
     config embedded in the checkpoint, falling back to `--config` if absent).
  2. Runs on GPU automatically if available, falls back to CPU otherwise.
  3. For every image in `--input_dir`, detects its resolution, normalizes it
     with the exact same (train-time-consistent) intensity normalization,
     runs a learned forward pass (NOT bicubic) to produce the restored
     high-resolution image, and saves it to `--output_dir` under the same
     filename.
  4. If `--gt_dir` is supplied, also computes PSNR / SSIM / LPIPS per image
     and writes a metrics.csv + prints an aggregate summary.
  5. Reports model loading time, warm-up time, per-image inference time, and
     throughput separately, plus parameter count and model size.

This script is fully independent of train.py / any notebook.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from datasets.paired_dataset import validate_and_pair
from metrics.metrics import LPIPSMetric, compute_psnr, compute_ssim, count_parameters, model_size_mb
from models.restoration_model import build_model
from utils.config import load_config
from utils.image_utils import (
    normalize_degraded,
    normalize_gt,
    read_image_grayscale_meta,
    save_restored,
)
from utils.logger import get_logger

logger = get_logger(name="evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / run inference with a trained restoration model")
    p.add_argument("--input_dir", type=str, required=True, help="Directory of degraded input images")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save restored images")
    p.add_argument("--weights", type=str, required=True, help="Path to a .pth checkpoint")
    p.add_argument("--config", type=str, default="config/config.yaml",
                    help="Fallback config if the checkpoint has none embedded")
    p.add_argument("--gt_dir", type=str, default=None, help="Optional ground-truth dir to compute metrics")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--use_amp", action="store_true", default=None, help="Force-enable mixed precision")
    p.add_argument("--no_amp", dest="use_amp", action="store_false", help="Force-disable mixed precision")
    p.add_argument("--use_tta", action="store_true", help="Flip/rotate self-ensemble (slower, optional)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (off by default)")
    p.add_argument("--channels_last", action="store_true", help="Use channels-last memory format")
    p.add_argument("--bit_depth_out", type=int, default=None, choices=[8, 16],
                    help="Force output bit depth; defaults to matching each input file")
    p.add_argument("--prefer_raw", action="store_true",
                    help="Load raw (non-EMA) weights even if an EMA copy is present in the checkpoint")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_model_and_config(weights_path: Path, fallback_config_path: str, device: str):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint.get("config") is not None:
        cfg = checkpoint["config"]
        logger.info("Loaded model configuration embedded in checkpoint.")
    else:
        cfg = load_config(fallback_config_path)
        logger.info("Checkpoint has no embedded config; using %s", fallback_config_path)

    model = build_model(cfg["model"] if isinstance(cfg, dict) else cfg.model)
    return model, cfg, checkpoint


def tta_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Lightweight 4-way flip self-ensemble (no rotation, so output orientation
    stays trivially correct). Roughly 4x slower; opt-in via --use_tta."""
    outputs = []
    for hflip in (False, True):
        for vflip in (False, True):
            xi = x
            if hflip:
                xi = torch.flip(xi, dims=[3])
            if vflip:
                xi = torch.flip(xi, dims=[2])
            yi = model(xi)
            if vflip:
                yi = torch.flip(yi, dims=[2])
            if hflip:
                yi = torch.flip(yi, dims=[3])
            outputs.append(yi)
    return torch.stack(outputs, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(args.weights)

    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    # ---- Model loading (timed separately from inference) ----
    t0 = time.perf_counter()
    model, cfg, checkpoint = load_model_and_config(weights_path, args.config, device)

    prefer_ema = not args.prefer_raw
    if isinstance(checkpoint, dict) and prefer_ema and checkpoint.get("ema_state_dict"):
        model.load_state_dict(checkpoint["ema_state_dict"])
        logger.info("Loaded EMA weights.")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded raw model weights.")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw state_dict.")

    model.to(device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)
    model_load_time = time.perf_counter() - t0

    norm_cfg = cfg["normalization"] if isinstance(cfg, dict) else cfg.normalization
    norm_method = norm_cfg.get("method", "percentile") if isinstance(norm_cfg, dict) else norm_cfg.method
    lower_pct = norm_cfg.get("lower_percentile", 0.5) if isinstance(norm_cfg, dict) else norm_cfg.lower_percentile
    upper_pct = norm_cfg.get("upper_percentile", 99.5) if isinstance(norm_cfg, dict) else norm_cfg.upper_percentile
    fixed_range = tuple(norm_cfg.get("fixed_range", (0.0, 255.0)) if isinstance(norm_cfg, dict) else norm_cfg.fixed_range)
    eps = norm_cfg.get("eps", 1e-6) if isinstance(norm_cfg, dict) else norm_cfg.eps

    inf_cfg = cfg["inference"] if isinstance(cfg, dict) else cfg.inference
    use_amp = args.use_amp if args.use_amp is not None else (inf_cfg.get("use_amp", True) if isinstance(inf_cfg, dict) else inf_cfg.use_amp)
    use_amp = use_amp and device == "cuda"

    n_params = count_parameters(model)
    size_mb = model_size_mb(model)
    logger.info("Model parameters: %.2fM | Size: %.2f MB", n_params / 1e6, size_mb)

    report = validate_and_pair(input_dir, args.gt_dir)
    if report.num_matched_pairs == 0:
        raise RuntimeError(f"No usable images found in {input_dir}.\n{report.summary()}")
    logger.info("Found %d input images.", report.num_matched_pairs)

    lpips_metric = None
    if args.gt_dir is not None:
        try:
            lpips_metric = LPIPSMetric(device=device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LPIPS unavailable (%s); skipping LPIPS metric.", exc)

    # ---- Warm-up (excluded from per-image timing) ----
    first_img, _ = read_image_grayscale_meta(report.valid_pairs[0][0])
    dummy_norm, _ = normalize_degraded(first_img, norm_method, lower_pct, upper_pct, fixed_range, eps)
    dummy = torch.from_numpy(dummy_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    if args.channels_last:
        dummy = dummy.to(memory_format=torch.channels_last)
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start

    # ---- Main inference loop ----
    inference_times = []
    metric_rows = []
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for d_path, g_path in report.valid_pairs:
        degraded_img, native_max_in = read_image_grayscale_meta(d_path)
        h_in, w_in = degraded_img.shape

        degraded_norm, _stats = normalize_degraded(
            degraded_img, norm_method, lower_pct, upper_pct, fixed_range, eps
        )
        x = torch.from_numpy(degraded_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        t_start = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                if args.use_tta:
                    pred = tta_forward(model, x)
                else:
                    pred = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        inference_times.append(elapsed)

        pred_np = pred.squeeze().float().cpu().numpy()
        out_h, out_w = pred_np.shape

        # Save in the SAME format as the input (.npy stays .npy, images stay
        # images), preserving the filename exactly. For .npy (pre-normalized
        # float data, e.g. the KLA dataset) native_max_in is 1.0, so this is
        # a clip to [0,1] -- the only clipping point in the whole pipeline.
        if d_path.suffix.lower() == ".npy":
            native_max_out = 1.0
        elif args.bit_depth_out is not None:
            native_max_out = 65535.0 if args.bit_depth_out == 16 else 255.0
        else:
            native_max_out = 65535.0 if native_max_in > 255.0 else 255.0
        out_path = output_dir / d_path.name
        save_restored(out_path, pred_np, native_max_out, reference_ext=d_path.suffix)

        row = {
            "filename": d_path.name, "input_h": h_in, "input_w": w_in,
            "output_h": out_h, "output_w": out_w, "inference_time_s": elapsed,
        }

        if g_path is not None:
            gt_img, _ = read_image_grayscale_meta(g_path)
            gt_norm, _gt_max = normalize_gt(gt_img)
            pred_clamped = np.clip(pred_np, 0.0, 1.0)
            psnr = compute_psnr(pred_clamped, gt_norm)
            ssim = compute_ssim(pred_clamped, gt_norm)
            psnr_vals.append(psnr)
            ssim_vals.append(ssim)
            row["psnr"] = psnr
            row["ssim"] = ssim
            if lpips_metric is not None:
                lp = lpips_metric(torch.from_numpy(pred_clamped), torch.from_numpy(gt_norm))
                lpips_vals.append(lp)
                row["lpips"] = lp

        metric_rows.append(row)
        logger.info("Restored %s: %dx%d -> %dx%d (%.1f ms)", d_path.name, h_in, w_in, out_h, out_w, elapsed * 1000)

    # ---- Report ----
    avg_time = float(np.mean(inference_times))
    total_time = float(np.sum(inference_times))
    throughput = 1.0 / avg_time if avg_time > 0 else float("inf")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test images processed:   {len(metric_rows)}")
    print(f"Device:                  {device}")
    print(f"Model parameters:        {n_params / 1e6:.2f} M")
    print(f"Model size:              {size_mb:.2f} MB")
    print(f"Model load time:         {model_load_time:.3f} s")
    print(f"Warm-up time (3 iters):  {warmup_time:.3f} s")
    print(f"Average inference time:  {avg_time * 1000:.2f} ms/image")
    print(f"Total inference time:    {total_time:.3f} s")
    print(f"Throughput:              {throughput:.2f} images/sec")
    if psnr_vals:
        print(f"Mean PSNR:               {np.mean(psnr_vals):.3f} dB")
        print(f"Mean SSIM:               {np.mean(ssim_vals):.4f}")
    if lpips_vals:
        print(f"Mean LPIPS:              {np.mean(lpips_vals):.4f}")
    print(f"Restored images saved to: {output_dir}")
    print("=" * 60)

    if metric_rows:
        csv_path = output_dir / "metrics.csv"
        fieldnames = list(metric_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)
        logger.info("Per-image metrics written to %s", csv_path)


if __name__ == "__main__":
    main()
