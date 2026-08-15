"""Configuration loading utilities.

Loads the YAML config into a nested, attribute-accessible object and supports
CLI overrides (`--set model.base_channels=64`) so experiments do not require
editing the YAML file by hand.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigDict(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_config_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config_dict(v) for v in obj]
    return obj


def load_config(path: str | Path) -> ConfigDict:
    """Load a YAML configuration file into a ConfigDict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    return _to_config_dict(raw)


def apply_overrides(cfg: ConfigDict, overrides: list[str]) -> ConfigDict:
    """Apply dotted-key overrides like ["training.batch_size=8", "model.scale_factor=4"]."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key.path=value")
        key_path, value_str = item.split("=", 1)
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = yaml.safe_load(value_str)
    return cfg


def to_plain_dict(obj: Any) -> Any:
    """Recursively convert a ConfigDict (or nested dict/list) into plain
    built-in dict/list types, so it can be safely pickled into a checkpoint
    without requiring the loader to import ConfigDict."""
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    return obj


def save_config(cfg: ConfigDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain_dict(cfg), f, sort_keys=False)
