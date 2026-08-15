"""Checkpoint save/load, EMA, and best-model selection utilities."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ModelEma:
    """Exponential Moving Average of model weights.

    Maintaining an EMA copy generally improves OOD generalization and reduces
    variance from mini-batch noise, at negligible extra cost (one buffer copy
    of the model + a lerp per step).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.ema_model.named_parameters())
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, buf in model.named_buffers():
            ema_buffers[name].copy_(buf)

    def state_dict(self) -> Dict[str, Any]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state_dict)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[ModelEma] = None,
    epoch: int = 0,
    best_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metrics": best_metrics or {},
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[ModelEma] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    return checkpoint


def load_weights_only(path: str | Path, model: nn.Module, map_location: str = "cpu", prefer_ema: bool = True) -> nn.Module:
    """Load only model weights for inference/evaluation (no optimizer state).

    If the checkpoint contains an EMA copy and `prefer_ema` is True, the EMA
    weights are loaded instead of the raw model weights (EMA typically
    generalizes slightly better).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if prefer_ema and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
            model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        # Raw state_dict saved directly.
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unrecognized checkpoint format in {path}")
    return model
