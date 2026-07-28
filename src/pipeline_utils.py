"""
Utilities for Rashomon×SHAP pipeline.

Includes configuration management, reproducibility helpers, and I/O utilities.
"""
from __future__ import annotations

import json
import logging
import os
import random
import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")


# =============================================================================
# Hardware Detection
# =============================================================================

def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware (CPU, GPU, MPS for Apple Silicon).
    
    Returns:
        Dictionary with hardware info and recommended settings.
    """
    # On SLURM, os.cpu_count() returns the full node count. It does not
    # return the allocated cores. Respect the scheduler's allocation. This
    # avoids oversubscription.
    _slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    _cpu_count = int(_slurm_cpus) if _slurm_cpus else (os.cpu_count() or 1)

    info = {
        "cpu_count": _cpu_count,
        "has_cuda": torch.cuda.is_available(),
        "cuda_devices": 0,
        "has_mps": False,
        "recommended_device": "cpu",
        "recommended_num_gpus": 0,
        "recommended_num_cpus": max(1, (os.cpu_count() or 1) - 1),
    }
    
    # Check CUDA
    if info["has_cuda"]:
        info["cuda_devices"] = torch.cuda.device_count()
        info["cuda_names"] = [torch.cuda.get_device_name(i) for i in range(info["cuda_devices"])]
        info["recommended_device"] = "cuda"
        info["recommended_num_gpus"] = info["cuda_devices"]
    
    # Check MPS (Apple Silicon)
    try:
        info["has_mps"] = torch.backends.mps.is_available()
        if info["has_mps"] and not info["has_cuda"]:
            info["recommended_device"] = "mps"
            # MPS does not use num_gpus in the same way. Keep it at 0 for AutoGluon.
            info["recommended_num_gpus"] = 0
    except AttributeError:
        pass  # Older PyTorch without MPS support
    
    return info


def get_optimal_resources(
    cfg_num_gpus: Union[int, str],
    cfg_num_cpus: Union[int, str],
) -> Dict[str, Union[int, str]]:
    """
    Resolve 'auto' settings to optimal values based on hardware.
    
    Args:
        cfg_num_gpus: Config value for GPUs (int or 'auto')
        cfg_num_cpus: Config value for CPUs (int or 'auto')
    
    Returns:
        Dictionary with resolved num_gpus and num_cpus values.
    """
    hw = detect_hardware()
    
    # Resolve GPUs
    if cfg_num_gpus == "auto":
        num_gpus = hw["recommended_num_gpus"]
    else:
        num_gpus = int(cfg_num_gpus)
        # Do not request more GPUs than available
        if hw["has_cuda"]:
            num_gpus = min(num_gpus, hw["cuda_devices"])
        elif not hw["has_mps"]:
            num_gpus = 0
    
    # Resolve CPUs
    if cfg_num_cpus == "auto":
        num_cpus = hw["recommended_num_cpus"]
    else:
        num_cpus = int(cfg_num_cpus)
        num_cpus = min(num_cpus, hw["cpu_count"])
    
    return {
        "num_gpus": num_gpus,
        "num_cpus": num_cpus,
        "device": hw["recommended_device"],
    }


# =============================================================================
# Generic Helpers
# =============================================================================

def _from_dict_dataclass(cls: Type[T], raw: Dict[str, Any]) -> T:
    """
    Create a dataclass instance from a dict. Ignore unknown keys.
    The function stores unknown keys in `obj.extra` if the dataclass defines it.
    """
    init_field_names = {f.name for f in fields(cls) if f.init}
    init_kwargs = {k: raw[k] for k in init_field_names if k in raw}
    obj = cls(**init_kwargs)

    extra_keys = {k: v for k, v in raw.items() if k not in init_field_names}
    if hasattr(obj, "extra"):
        setattr(obj, "extra", extra_keys)

    return obj


def _require_positive_int(name: str, v: int) -> None:
    if not isinstance(v, int) or v <= 0:
        raise ValueError(f"{name} must be a positive int; got {v!r}")


def _require_nonneg_int(name: str, v: int) -> None:
    if not isinstance(v, int) or v < 0:
        raise ValueError(f"{name} must be a non-negative int; got {v!r}")


def _require_nonneg_float(name: str, v: float) -> None:
    try:
        fv = float(v)
    except Exception as e:
        raise ValueError(f"{name} must be a float; got {v!r}") from e
    if fv < 0.0:
        raise ValueError(f"{name} must be non-negative; got {v!r}")


def _require_in(name: str, v: Any, allowed: List[Any]) -> None:
    if v not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {v!r}")


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ProjectConfig:
    outdir: str
    run_name: Optional[str]
    random_seed: int
    
    extra: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outdir, str) or not self.outdir.strip():
            raise ValueError("outdir must be a non-empty string")
        _require_positive_int("random_seed", int(self.random_seed))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProjectConfig":
        return _from_dict_dataclass(cls, raw)


@dataclass
class DataConfig:
    """General-purpose data configuration. Unknown YAML keys land in ``extra``."""
    freq: str
    horizon: int
    target_lags: int
    cov_lag: int

    extra: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.freq, str) or not self.freq.strip():
            raise ValueError("freq must be a non-empty string (e.g. 'M')")
        _require_positive_int("horizon", self.horizon)
        _require_nonneg_int("target_lags", self.target_lags)
        _require_nonneg_int("cov_lag", self.cov_lag)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DataConfig":
        return _from_dict_dataclass(cls, raw)


@dataclass
class SplitsConfig:
    n_splits: int
    min_train_time: int
    val_size: int
    test_size: int

    extra: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_positive_int("n_splits", self.n_splits)
        _require_positive_int("min_train_time", self.min_train_time)
        _require_positive_int("val_size", self.val_size)
        _require_positive_int("test_size", self.test_size)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SplitsConfig":
        return _from_dict_dataclass(cls, raw)


@dataclass
class AutoMLConfig:
    time_limit_s: int
    presets: str
    seeds: List[int]
    eval_metric: str
    max_models_per_rashomon: int
    num_gpus: Union[int, str]
    num_cpus: Union[int, str]
    fit_strategy: str
    early_stop_patience: int = 10
    use_bag_holdout: bool = True
    verbosity: int = 2

    extra: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_positive_int("time_limit_s", self.time_limit_s)
        if not isinstance(self.presets, str) or not self.presets.strip():
            raise ValueError("presets must be a non-empty string")

        if not isinstance(self.seeds, list) or not self.seeds:
            raise ValueError("seeds must be a non-empty list of ints")
        for i, s in enumerate(self.seeds):
            if not isinstance(s, int):
                raise ValueError(f"seeds[{i}] must be int; got {s!r}")

        if not isinstance(self.eval_metric, str) or not self.eval_metric.strip():
            raise ValueError("eval_metric must be a non-empty string")

        _require_positive_int("max_models_per_rashomon", self.max_models_per_rashomon)

        if isinstance(self.num_gpus, int):
            _require_nonneg_int("num_gpus", self.num_gpus)
        elif isinstance(self.num_gpus, str):
            _require_in("num_gpus", self.num_gpus, ["auto"])
        else:
            raise ValueError("num_gpus must be int or 'auto'")

        if isinstance(self.num_cpus, int):
            _require_positive_int("num_cpus", self.num_cpus)
        elif isinstance(self.num_cpus, str):
            _require_in("num_cpus", self.num_cpus, ["auto"])
        else:
            raise ValueError("num_cpus must be int or 'auto'")

        _require_in("fit_strategy", self.fit_strategy, ["sequential", "parallel"])

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AutoMLConfig":
        return _from_dict_dataclass(cls, raw)


@dataclass
class RashomonConfig:
    eps_list: List[float]

    extra: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.eps_list, list) or not self.eps_list:
            raise ValueError("eps_list must be a non-empty list of floats")
        for i, e in enumerate(self.eps_list):
            try:
                fe = float(e)
            except Exception as ex:
                raise ValueError(f"eps_list[{i}] must be float; got {e!r}") from ex
            if fe <= 0.0:
                raise ValueError(f"eps_list[{i}] must be > 0; got {e!r}")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RashomonConfig":
        return _from_dict_dataclass(cls, raw)


@dataclass
class ReportConfig:
    topk_features: int
    generate_plots: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["csv", "parquet"])

    extra: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_positive_int("topk_features", self.topk_features)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ReportConfig":
        return _from_dict_dataclass(cls, raw)


# =============================================================================
# Reproducibility + IO Utilities
# =============================================================================

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Environment variable for some libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_run_dir(outdir: str, run_name: Optional[str]) -> Path:
    """Create and return the run directory."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Save dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    run_dir: Path,
    level: int = logging.INFO,
    name: str = "rashomon_shap"
) -> logging.Logger:
    """
    Setup structured logging with file and console handlers.
    
    Args:
        run_dir: Directory to save log file
        level: Logging level
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# Memory Utilities
# =============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def log_memory_usage(logger: logging.Logger, prefix: str = "") -> None:
    """Log current memory usage."""
    mem = get_memory_usage_mb()
    if mem > 0:
        logger.debug(f"{prefix}Memory usage: {mem:.1f} MB")
