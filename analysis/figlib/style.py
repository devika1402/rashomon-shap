"""
figlib.style: colour palette, fonts, and matplotlib style presets.

There are two style presets. The thesis figures were authored in two visual
families. This file keeps both families verbatim.

  * ``apply_base_style()``: the bold-title academic style. The four
    data-overview figures use it (epsilon sensitivity, set size, stability
    heatmap, stability-vs-size scatter).
  * ``apply_ink_style()``: the high-contrast INK / MUTED / HAIR ink scheme.
    The two close-read figures use it (SHAP-CV groups, robustness bars).

Each preset is self-contained. It sets every key that the other preset touches
back to the matplotlib default. Either preset therefore fully determines the
style state. The render order of figures in one process does not matter. This
lets ``make_figures.py`` render all figures in one run. The output still
matches the original one-script-per-style layout byte for byte.
"""
from __future__ import annotations

import glob as _glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ROOT / "figures"

# ── Framework palette (the UL logo hues) ──────────────────────────────────────
C_AG = "#3c7fd0"     # AutoGluon blue
C_H2O = "#f8c925"    # H2O gold
C_LIGHT = "#BBBBBB"   # error bands / minor elements (base style)

# ── Ink scheme (close-read figures) ───────────────────────────────────────────
INK = "#1a1a1a"      # primary text / data labels
MUTED = "#666666"    # secondary: ticks, rank numbers
HAIR = "#cccccc"     # spines, dividers, reference lines

# ── Feature-group palette ─────────────────────────────────────────────────────
FEATURE_COLORS = {
    "target_lag": "#0072B2",   # blue
    "cov":        "#009E73",   # teal-green
    "calendar":   "#E69F00",   # amber
    "other":      "#888888",   # grey
}

# The text width of a standard thesis page is about 6.3 in. Author the figures
# at 6.5 and include them at width=\textwidth. The on-page font size is then
# identical across figures.
WIDTH = 6.5

# Shared serif stack. It resolves to Latin Modern (the thesis body font) when
# TeX Live is installed. If not, it uses CMU Serif, then DejaVu Serif.
SERIF_STACK = ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"]


def _register_latin_modern() -> None:
    """Register Latin Modern with matplotlib if TeX Live is present.

    This fails soft on a machine without TeX Live, such as the cluster. There
    the serif stack falls back to DejaVu Serif and Computer Modern math.
    """
    for lm in _glob.glob(
        "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf"
    ):
        try:
            _fm.fontManager.addfont(lm)
        except Exception:
            pass


_register_latin_modern()


# ── Style presets ─────────────────────────────────────────────────────────────
# Each dict is the full union of keys that either preset touches. A key that a
# preset does not "own" is set to the matplotlib default. Neither preset can
# then leak into the other across figures rendered in the same process.

BASE_RC = {
    "font.family":       "serif",
    "font.serif":        SERIF_STACK,
    "mathtext.fontset":  "cm",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.frameon":    False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    # ink-only keys reset to matplotlib defaults:
    "axes.edgecolor":    "black",
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
}

INK_RC = {
    "font.family":       "serif",
    "font.serif":        SERIF_STACK,
    "mathtext.fontset":  "cm",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    HAIR,
    "axes.linewidth":    0.8,
    "text.color":        INK,
    "axes.labelcolor":   INK,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "legend.frameon":    False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    # base-only keys reset to matplotlib defaults:
    "axes.labelsize":    "medium",
    "axes.titlesize":    "large",
    "axes.titleweight":  "normal",
    "xtick.labelsize":   "medium",
    "ytick.labelsize":   "medium",
    "legend.fontsize":   "medium",
}


def apply_base_style() -> None:
    """Bold-title academic style (data-overview figures)."""
    plt.rcParams.update(BASE_RC)


def apply_ink_style() -> None:
    """High-contrast ink scheme (close-read figures)."""
    plt.rcParams.update(INK_RC)


def alpha_fill(hexcolor: str, a: float) -> tuple[float, float, float, float]:
    """Return an RGBA tuple for a hex colour at alpha ``a``."""
    r = int(hexcolor[1:3], 16) / 255
    g = int(hexcolor[3:5], 16) / 255
    b = int(hexcolor[5:7], 16) / 255
    return (r, g, b, a)


def save_fig(fig: "plt.Figure", name: str) -> Path:
    """Save ``fig`` to ``figures/<name>`` plus a PNG twin, and close it.

    The thesis includes the PDF. GitHub does not render a PDF inline. The
    README therefore needs a raster copy of the same figure, which is the PNG.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  saved -> {path} (+ .png)")
    return path


def abort_empty(fig: "plt.Figure", name: str, n_series: int) -> None:
    """Discard ``fig`` and skip if nothing was plotted into it.

    A figure module loops over runs and skips each missing one. Without this
    guard it reaches save_fig with empty axes. It then overwrites a good PDF
    with a blank chart. This fails without any sign, because the file still
    looks like a figure. The guard turns a missing ``results/`` into a visible
    skip.
    """
    if n_series == 0:
        plt.close(fig)
        print(f"  [skip] {name}: no results/ data found. "
              "Run the pipeline first, and run from the project root.")
