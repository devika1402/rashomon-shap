#!/usr/bin/env python3
"""
make_figures.py: single entry point for all thesis figures.

Usage (from the project root):
    python analysis/make_figures.py                 # build every figure
    python analysis/make_figures.py all             # same
    python analysis/make_figures.py rashomon_size   # build one
    python analysis/make_figures.py shap_cv_groups robustness_bars
    python analysis/make_figures.py --list          # list figure names

Each figure is one module in analysis/figures/ exposing build() -> Path.
Shared palette, style, dataset registry, and CSV loaders live in
analysis/figlib/. This script writes outputs to figures/ at the project root.

Data-driven figures read results/<run>/... Run from the project root. The
relative paths then resolve. Without results/ present, those modules skip and
leave the committed figures untouched. robustness_bars reads the committed
analysis/robustness_tree_only/ CSVs. It builds from a clean clone.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make analysis/ importable. `figlib` and `figures` then resolve as packages,
# whatever the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import (
    epsilon_sensitivity,
    rashomon_size,
    robustness_bars,
    shap_cv_groups,
    stability_heatmap,
    stability_vs_size,
)

# Registry: ordered so `all` builds overview figures before close-read ones.
REGISTRY = {
    "rashomon_size":       rashomon_size.build,
    "epsilon_sensitivity": epsilon_sensitivity.build,
    "stability_heatmap":   stability_heatmap.build,
    "stability_vs_size":   stability_vs_size.build,
    "shap_cv_groups":      shap_cv_groups.build,
    "robustness_bars":     robustness_bars.build,
}


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("--list", "-l"):
        print("Available figures:")
        for name in REGISTRY:
            print(f"  {name}")
        return 0

    if not argv or argv == ["all"]:
        selected = list(REGISTRY)
    else:
        unknown = [a for a in argv if a not in REGISTRY]
        if unknown:
            print(f"Unknown figure(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(REGISTRY)}  (or 'all')")
            return 2
        selected = argv

    print(f"Building {len(selected)} figure(s) -> figures/\n")
    for name in selected:
        REGISTRY[name]()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
