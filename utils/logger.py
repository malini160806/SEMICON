"""Experiment logging: TensorBoard + CSV, plus reproducibility helpers."""
from __future__ import annotations

import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs used across the pipeline for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_logger(name: str = "kla_restoration", log_file: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


class CSVLogger:
    """Append-only CSV logger for epoch-level training metrics."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class TBLogger:
    """Thin wrapper around torch.utils.tensorboard, no-op if unavailable."""

    def __init__(self, log_dir: str | Path):
        self.enabled = _HAS_TB
        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def image(self, tag: str, img_tensor, step: int) -> None:
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        if self.enabled:
            self.writer.close()
