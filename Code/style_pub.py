"""
style_pub.py — Shared publication style for PRE/Soft Matter figures.
Usage:
    from style_pub import set_style, fig_double, fig_single, save, add_panel_label, \
                          kymo_show, COLORS, PRE_SINGLE, PRE_DOUBLE

All figures: PDF vector output, dpi=600, serif fonts (TrueType embedded).
Kymographs: rasterized=True for imshow layer, vector axes.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Layout constants ──────────────────────────────────────────────
PRE_SINGLE = 3.375   # inches  (single-column PRE/Soft Matter)
PRE_DOUBLE = 6.875   # inches  (double-column)

# ── Color palette (colorblind-tolerant) ───────────────────────────
COLORS = {
    "blue":   "#1f77b4",
    "red":    "#d62728",
    "green":  "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "brown":  "#8c564b",
    "gray":   "#7f7f7f",
    "cyan":   "#17becf",
}
C = list(COLORS.values())   # indexed access: C[0]=blue, C[1]=red, ...

# ── Output directory ──────────────────────────────────────────────
OUT = Path(__file__).parent.parent / "Figure" / "pub"

# ── rcParams ──────────────────────────────────────────────────────
RC = {
    "font.family":       "serif",
    "mathtext.fontset":  "stix",       # STIX = Times-compatible math
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth":   1.0,
    "axes.linewidth":    0.6,
    "grid.linewidth":    0.4,
    "legend.fontsize":   7,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.7",
    "legend.handlelength": 1.5,
    "figure.dpi":        150,          # screen preview
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":      42,           # TrueType fonts → editable in Illustrator
    "ps.fonttype":       42,
}


def set_style():
    """Apply PRE rcParams globally."""
    plt.rcParams.update(RC)


# ── Figure factories ──────────────────────────────────────────────
def fig_single(h=3.0, **kw):
    """Single-column figure (3.375 in wide)."""
    return plt.subplots(figsize=(PRE_SINGLE, h), **kw)


def fig_double(h=3.5, **kw):
    """Double-column figure (6.875 in wide)."""
    return plt.subplots(figsize=(PRE_DOUBLE, h), **kw)


def fig_panels(nrow, ncol, w=None, h=None, **kw):
    """Custom multi-panel figure."""
    w = w or PRE_DOUBLE
    h = h or (w * nrow / ncol * 0.8)
    return plt.subplots(nrow, ncol, figsize=(w, h), **kw)


# ── Panel labels ──────────────────────────────────────────────────
def add_panel_label(ax, label, x=-0.18, y=1.04):
    """Add bold (a), (b), ... label to upper-left of axes."""
    ax.text(x, y, f"({label})",
            transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


# ── Kymograph helper ──────────────────────────────────────────────
def kymo_show(ax, data, x_arr, t_arr, cmap="RdBu_r", label="",
              vmin=None, vmax=None, colorbar=True):
    """
    imshow kymograph with rasterized=True (hybrid PDF: vector axes, raster image).
    data shape: (N_space, N_time), x_arr: spatial, t_arr: temporal.
    """
    extent = [t_arr[0], t_arr[-1], x_arr[0], x_arr[-1]]
    im = ax.imshow(data, origin="lower", aspect="auto", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel(r"Time $\tau$")
    ax.set_ylabel(r"$x/H_0$")
    if colorbar:
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label(label, fontsize=7)
    return im


# ── Save ──────────────────────────────────────────────────────────
def save(fig, name, tight=True):
    """Save to figures_pub/{name}.pdf at 600 dpi."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    if tight:
        fig.savefig(path)         # bbox_inches="tight" already in RC
    else:
        fig.savefig(path, bbox_inches=None)
    print(f"  → saved: {path}")
    plt.close(fig)
