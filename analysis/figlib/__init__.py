"""
figlib: shared infrastructure for thesis figure generation.

This package holds the shared parts exactly once: the colour palette, the
matplotlib style presets, the dataset registry, the result-CSV loaders, and
the save helper. The figure scripts used to copy-paste these parts.

Figure modules in ``analysis/figures/`` import from here. They contain only
their own plotting logic. The CLI entry point is ``analysis/make_figures.py``.
"""
from __future__ import annotations

from . import style, datasets, data

__all__ = ["style", "datasets", "data"]
