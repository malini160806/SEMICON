#!/usr/bin/env python
"""Lightweight production inference script (no metrics/validation overhead).

Use this for quick single-image or batch restoration / latency benchmarking.
For dataset evaluation with PSNR/SSIM/LPIPS reporting, use evaluate.py instead.

Examples:
    python inference.py --input path/to/image.npy --output_dir ./out --weights weights/best_model.pth
    python inference.py --input ./datasets/Test_NoisyLR --output_dir ./outputs/test_restored --weights weights/best_model.pth
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluate import load_model_and_config, resolve_device
from utils.image_utils import (
    SUPPORTED_EXTENSIONS,
    normalize_degraded,
    read_image_grayscale_meta,
    save_restored,
)
from utils.logger import get_logger

logger = get_logger(name="inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast production inference (single image or directory)")
    p.add_argument("--input", type=str, required=True, help="Image file or directory of images")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--use_amp", action="store_true", default=None)
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    t0 = time.perf_counter()
    model, cfg, checkpoint = load_model_and_config(Path(args.weights), args.config, device)
    if isinstance(checkpoint, dict) and checkpoint.get("ema_state_dict"):
        model.load_state_dict(checkpoint["ema_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.3fs on %s", load_time, device)

    norm_cfg = cfg["normalization"] if isinstance(cfg, dict) else cfg.normalization
    norm_method = norm_cfg.get("method", "percentile") if isinstance(norm_cfg, dict) else norm_cfg.method
    lower_pct = norm_cfg.get("lower_percentile", 0.5) if isinstance(norm_cfg, dict) else norm_cfg.lower_percentile
    upper_pct = norm_cfg.get("upper_percentile", 99.5) if isinstance(norm_cfg, dict) else norm_cfg.upper_percentile
    fixed_range = tuple(norm_cfg.get("fixed_range", (0.0, 255.0)) if isinstance(norm_cfg, dict) else norm_cfg.fixed_range)
    eps = norm_cfg.get("eps", 1e-6) if isinstance(norm_cfg, dict) else norm_cfg.eps

    inf_cfg = cfg["inference"] if isinstance(cfg, dict) else cfg.inference
    use_amp = args.use_amp if args.use_amp is not None else (inf_cfg.get("use_amp", True) if isinstance(inf_cfg, dict) else inf_cfg.use_amp)
    use_amp = use_amp and device == "cuda"

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"No images found at {input_path}")

    times = []
    for img_path in image_paths:
        img, native_max_in = read_image_grayscale_meta(img_path)
        norm, _ = normalize_degraded(img, norm_method, lower_pct, upper_pct, fixed_range, eps)
        x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float().to(device)
        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        t_start = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        times.append(elapsed)

        pred_np = pred.squeeze().float().cpu().numpy()
        # Save in the same format as the input (.npy stays .npy, images stay images).
        native_max_out = 1.0 if img_path.suffix.lower() == ".npy" else (
            65535.0 if native_max_in > 255.0 else 255.0
        )
        save_restored(output_dir / img_path.name, pred_np, native_max_out, reference_ext=img_path.suffix)
        logger.info("%s -> %s (%.1f ms)", img_path.name, pred_np.shape, elapsed * 1000)

    avg_ms = (sum(times) / len(times)) * 1000
    print(f"\nProcessed {len(image_paths)} image(s) | avg {avg_ms:.2f} ms/image | "
          f"{1000.0 / avg_ms:.2f} images/sec | model load {load_time:.3f}s")


if __name__ == "__main__":
    main()
