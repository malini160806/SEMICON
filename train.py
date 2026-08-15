#!/usr/bin/env python
"""Training entry point for the KLA/i4C image restoration model.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --set training.epochs=50 training.batch_size=8

Loads paired data from `data.train_dir` (config/config.yaml), auto-detecting
its degraded/ground-truth subfolder layout (e.g. NoisyLR/GT), and splits it
into train/validation with a seeded, leak-free 80/20 split -- `data.test_dir`
(e.g. Test_NoisyLR) is never touched here; it has no ground truth and is only
used by evaluate.py/inference.py for final inference.

Implements: AdamW, cosine LR schedule with warmup, mixed precision (torch.amp),
gradient clipping, gradient accumulation, EMA, early stopping,
validation-based checkpointing (best_psnr.pth / best_ssim.pth / last_model.pth /
best_model.pth), automatic resume, TensorBoard + CSV logging.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.augmentations import AugmentConfig
from datasets.paired_dataset import PairedRestorationDataset, build_train_val_split
from losses.restoration_loss import build_loss
from metrics.metrics import LPIPSMetric, compute_psnr, compute_ssim, count_parameters
from models.restoration_model import build_model
from utils.checkpoint import ModelEma, load_checkpoint, save_checkpoint
from utils.config import apply_overrides, load_config, save_config, to_plain_dict
from utils.logger import CSVLogger, TBLogger, get_logger, set_seed

CSV_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "val_lpips",
    "learning_rate", "epoch_time_s", "gpu_mem_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KLA image restoration model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--set", nargs="*", default=[], help="Override config keys, e.g. training.epochs=10")
    parser.add_argument("--resume_path", type=str, default=None, help="Explicit checkpoint to resume from")
    parser.add_argument("--no_resume", action="store_true", help="Ignore existing checkpoints, train from scratch")
    return parser.parse_args()


def build_dataloaders(cfg, logger) -> tuple[DataLoader, DataLoader]:
    """Builds train/val loaders from `data.train_dir` (auto-detects its
    degraded/ground-truth subfolders, e.g. NoisyLR/GT) with a seeded 80/20
    split. `data.test_dir` (e.g. Test_NoisyLR) is NEVER touched here --
    it has no ground truth and is only ever used by evaluate.py/inference.py.
    """
    aug_cfg = AugmentConfig.from_dict(dict(cfg.augmentation)) if cfg.augmentation.enabled else AugmentConfig(enabled=False)

    split = build_train_val_split(
        train_dir=cfg.data.train_dir,
        val_ratio=cfg.data.validation_split,
        seed=cfg.data.seed,
        scale_factor=cfg.data.scale_factor,
        max_pairs=cfg.data.get("max_pairs"),
    )
    logger.info(
        "Dataset layout: %s | detected scale factors: %s | %d train / %d val pairs",
        split.layout, sorted(split.report.detected_scales), len(split.train_pairs), len(split.val_pairs),
    )
    if split.report.resolution_mismatches:
        logger.warning("%d pairs had unexpected resolution ratios (see validate_dataset.py for detail)",
                        len(split.report.resolution_mismatches))

    common_norm_kwargs = dict(
        norm_method=cfg.normalization.method,
        lower_percentile=cfg.normalization.lower_percentile,
        upper_percentile=cfg.normalization.upper_percentile,
        fixed_range=tuple(cfg.normalization.fixed_range),
        eps=cfg.normalization.eps,
    )

    train_ds = PairedRestorationDataset(
        pairs=split.train_pairs,
        scale_factor=cfg.data.scale_factor,
        patch_size=cfg.data.patch_size,
        augment=True,
        augment_cfg=aug_cfg,
        seed=cfg.experiment.seed,
        split_name="train",
        **common_norm_kwargs,
    )
    val_ds = PairedRestorationDataset(
        pairs=split.val_pairs,
        scale_factor=cfg.data.scale_factor,
        patch_size=cfg.data.patch_size,
        augment=False,  # validation gets deterministic preprocessing only, no augmentation
        seed=cfg.experiment.seed,
        split_name="val",
        **common_norm_kwargs,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_epochs = cfg.training.epochs
    warmup_epochs = cfg.training.warmup_epochs
    min_lr_ratio = cfg.training.min_lr / cfg.training.learning_rate

    def lr_lambda(step: int) -> float:
        epoch = step / max(steps_per_epoch, 1)
        if epoch < warmup_epochs:
            return max(epoch / max(warmup_epochs, 1e-8), 1e-3)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1e-8)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, loss_fn, device: str, use_amp: bool,
             lpips_metric: LPIPSMetric | None) -> dict:
    model.eval()
    total_loss, total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0, 0.0
    n = 0
    for batch in val_loader:
        degraded = batch["degraded"].to(device)
        gt = batch["gt"].to(device)
        with torch.autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
            pred = model(degraded)
            loss, _ = loss_fn(pred, gt)

        pred_np = pred.clamp(0, 1).squeeze().float().cpu().numpy()
        gt_np = gt.squeeze().float().cpu().numpy()

        total_loss += loss.item()
        total_psnr += compute_psnr(pred_np, gt_np)
        total_ssim += compute_ssim(pred_np, gt_np)
        if lpips_metric is not None:
            total_lpips += lpips_metric(pred.clamp(0, 1), gt)
        n += 1

    n = max(n, 1)
    return {
        "val_loss": total_loss / n,
        "val_psnr": total_psnr / n,
        "val_ssim": total_ssim / n,
        "val_lpips": (total_lpips / n) if lpips_metric is not None else float("nan"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    log_dir = Path(cfg.training.log_dir) / cfg.experiment.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_file=log_dir / "train.log")
    set_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    save_config(cfg, log_dir / "resolved_config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(cfg, logger)
    logger.info("Train batches/epoch: %d | Val images: %d", len(train_loader), len(val_loader))

    model = build_model(cfg.model).to(device)
    logger.info("Model parameters: %.2fM", count_parameters(model) / 1e6)

    loss_fn = build_loss(cfg.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(enabled=(cfg.training.use_amp and device == "cuda"))

    ema = ModelEma(model, decay=cfg.training.ema.decay, device=device) if cfg.training.ema.enabled else None

    lpips_metric = None
    try:
        lpips_metric = LPIPSMetric(device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LPIPS unavailable (%s); validation LPIPS will be skipped.", exc)

    csv_logger = CSVLogger(cfg.training.csv_log_path, CSV_FIELDS)
    tb_logger = TBLogger(log_dir)

    start_epoch = 0
    best_metrics = {"psnr": -1.0, "ssim": -1.0, "lpips": float("inf")}
    patience_counter = 0

    last_ckpt = Path(args.resume_path) if args.resume_path else ckpt_dir / "last_model.pth"
    if not args.no_resume and cfg.training.resume and last_ckpt.exists():
        checkpoint = load_checkpoint(last_ckpt, model, optimizer, scheduler, scaler, ema, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metrics.update(checkpoint.get("best_metrics", {}))
        logger.info("Resumed from %s at epoch %d", last_ckpt, start_epoch)

    plain_cfg = to_plain_dict(cfg)
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            degraded = batch["degraded"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(cfg.training.use_amp and device == "cuda")):
                pred = model(degraded)
                loss, loss_parts = loss_fn(pred, gt)
                loss = loss / cfg.training.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.training.grad_accum_steps == 0:
                if cfg.training.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            scheduler.step()
            global_step += 1
            running_loss += loss_parts["total"]
            pbar.set_postfix(loss=loss_parts["total"], lr=optimizer.param_groups[0]["lr"])

        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

        val_metrics = {"val_loss": float("nan"), "val_psnr": float("nan"),
                        "val_ssim": float("nan"), "val_lpips": float("nan")}
        if (epoch + 1) % cfg.training.val_every_n_epochs == 0:
            eval_model = ema.ema_model if ema is not None else model
            val_metrics = validate(
                eval_model, val_loader, loss_fn, device, cfg.training.use_amp, lpips_metric
            )
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f PSNR=%.3f SSIM=%.4f LPIPS=%.4f | %.1fs",
                epoch + 1, train_loss, val_metrics["val_loss"], val_metrics["val_psnr"],
                val_metrics["val_ssim"], val_metrics["val_lpips"], epoch_time,
            )

        csv_logger.log({
            "epoch": epoch + 1, "train_loss": train_loss, **val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"], "epoch_time_s": epoch_time,
            "gpu_mem_mb": gpu_mem_mb,
        })
        tb_logger.scalar("train/loss", train_loss, epoch)
        tb_logger.scalar("val/loss", val_metrics["val_loss"], epoch)
        tb_logger.scalar("val/psnr", val_metrics["val_psnr"], epoch)
        tb_logger.scalar("val/ssim", val_metrics["val_ssim"], epoch)
        tb_logger.scalar("val/lpips", val_metrics["val_lpips"], epoch)
        tb_logger.scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        improved = False
        if val_metrics["val_psnr"] > best_metrics["psnr"]:
            best_metrics["psnr"] = val_metrics["val_psnr"]
            save_checkpoint(ckpt_dir / "best_psnr.pth", model, optimizer, scheduler, scaler, ema,
                             epoch, best_metrics, plain_cfg)
            if cfg.training.select_best_metric == "psnr":
                improved = True
        if val_metrics["val_ssim"] > best_metrics["ssim"]:
            best_metrics["ssim"] = val_metrics["val_ssim"]
            save_checkpoint(ckpt_dir / "best_ssim.pth", model, optimizer, scheduler, scaler, ema,
                             epoch, best_metrics, plain_cfg)
            if cfg.training.select_best_metric == "ssim":
                improved = True
        if val_metrics["val_lpips"] < best_metrics["lpips"]:
            best_metrics["lpips"] = val_metrics["val_lpips"]
            if cfg.training.select_best_metric == "lpips":
                improved = True

        if improved:
            save_checkpoint(ckpt_dir / "best_model.pth", model, optimizer, scheduler, scaler, ema,
                             epoch, best_metrics, plain_cfg)
            patience_counter = 0
            logger.info("New best model (%s) saved to %s", cfg.training.select_best_metric, ckpt_dir / "best_model.pth")
        else:
            patience_counter += 1

        save_checkpoint(ckpt_dir / "last_model.pth", model, optimizer, scheduler, scaler, ema,
                         epoch, best_metrics, plain_cfg)
        if (epoch + 1) % cfg.training.checkpoint_every_n_epochs == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch + 1}.pth", model, optimizer, scheduler, scaler, ema,
                             epoch, best_metrics, plain_cfg)

        if cfg.training.early_stopping.enabled and patience_counter >= cfg.training.early_stopping.patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch + 1, patience_counter)
            break

    tb_logger.close()
    logger.info("Training complete. Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
