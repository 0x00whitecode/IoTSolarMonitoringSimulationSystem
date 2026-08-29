"""Configuration loader and shared utilities."""
import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).parent.resolve()
_CONFIG_CACHE: Dict[str, Any] | None = None


def get_project_root() -> Path:
    return _PROJECT_ROOT


def load_config(config_path: str | None = None) -> dict:
    """Load the central YAML configuration. Cached after first load."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    _CONFIG_CACHE = cfg
    return cfg


def setup_logging(cfg: dict | None = None) -> logging.Logger:
    """Configure project-wide logging."""
    if cfg is None:
        cfg = load_config()
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = _PROJECT_ROOT / log_cfg.get("file", "results/logs/system.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=fmt,
                         handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger("smart_solar_iot")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_path(path: str | Path) -> Path:
    """Return absolute path relative to project root."""
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p
