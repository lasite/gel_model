#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig4_phasemap.py — Fig.4: 1D slab S_χ × Da phase map + amplitude curves.

Layout: double-column (6.875 in), mosaic layout.
  (a) [left, tall] S_χ × Da 2D phase map from 1D slab scan (16×16)
  (b) [top-right]  J_amp and θ_amp vs Da for both regimes (from scan CSV)
  (c) [bot-right]  Penetration depth ℓ_u vs Bi_c (new short scan)

Requires: scan_results_schi_da/scan_results.csv
Run the scan first with:
  python Code/scan_optimized.py --x-param S_chi --x-min 0.4 --x-max 1.4 --nx 16 \
    --y-param Da --y-min 3 --y-max 20 --ny 16 --N 40 --t-end 400 \
    --outdir scan_results_schi_da
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

from style_pub import (set_style, PRE_DOUBLE, C, COLORS,
                       add_panel_label, save, kymo_show)
from scan_optimized import Params, simulate, finalize_params

set_style()

SCAN_CSV = Path(__file__).parent.parent / "scan_results_schi_da" / "scan_results.csv"

# ── Label → color mapping ─────────────────────────────────────────
LABEL_COLORS = {
    "oscillatory_nonuniform": "#1f77b4",
    "oscillatory_uniform":    "#aec6e8",
    "steady_cold_uniform":    "#d0e8ff",
    "steady_cold_nonuniform": "#9ecae1",
    "steady_warm_uniform":    "#fdae6b",
    "steady_warm_nonuniform": "#f5922a",
    "steady_hot_uniform":     "#d62728",
    "steady_hot_nonuniform":  "#843c0c",
    "solve_failed":           "#333333",
}
LABEL_DISPLAY = {
    "oscillatory_nonuniform": "Oscillatory (nonunif.)",
    "oscillatory_uniform":    "Oscillatory (uniform)",
    "steady_cold_uniform":    "Steady cold",
    "steady_cold_nonuniform": "Steady cold (nonunif.)",
    "steady_warm_uniform":    "Steady warm",
    "steady_warm_nonuniform": "Steady warm (nonunif.)",
    "steady_hot_uniform":     "Steady hot",
    "steady_hot_nonuniform":  "Steady hot (nonunif.)",
    "solve_failed":           "Failed",
}


# ══════════════════════════════════════════════════════════════════
# Panel (a): 2D phase map
# ══════════════════════════════════════════════════════════════════

def panel_phase_map(ax, df):
    Da_vals   = sorted(df["Da"].unique())
    Schi_vals = sorted(df["S_chi"].unique())
    nDa   = len(Da_vals)
    nSchi = len(Schi_vals)

    # Build color image
    img = np.zeros((nSchi, nDa, 3))
    J_amp_grid = np.full((nSchi, nDa), np.nan)

    for i, sc in enumerate(Schi_vals):
        for j, da in enumerate(Da_vals):
            row = df[(df["S_chi"] == sc) & (df["Da"] == da)]
            if len(row) == 0:
                continue
            label = row.iloc[0].get("label", "solve_failed")
            hex_c = LABEL_COLORS.get(label, "#888888")
            r, g, b = mcolors.to_rgb(hex_c)
            img[i, j] = [r, g, b]
            if "J_amp" in row.columns:
                J_amp_grid[i, j] = row.iloc[0]["J_amp"]

    extent = [Da_vals[0], Da_vals[-1], Schi_vals[0], Schi_vals[-1]]
    ax.imshow(img, origin="lower", aspect="auto", extent=extent, rasterized=True)

    # J_amp contours
    try:
        X, Y = np.meshgrid(Da_vals, Schi_vals)
        cs = ax.contour(X, Y, J_amp_grid, levels=[0.1, 0.5, 1.0],
                        colors="white", linewidths=0.7, linestyles="--", alpha=0.8)
        ax.clabel(cs, fmt=r"$\Delta J=%.1f$", fontsize=5.5, inline=True)
    except Exception:
        pass

    # Regime labels
    ax.text(5.0,  0.65, "Regime I\n(volume pulse)", ha="center", fontsize=6.5,
            color="white", fontweight="bold", multialignment="center")
    ax.text(12.0, 1.15, "Regime II\n(thermal breathing)", ha="center", fontsize=6.5,
            color="white", fontweight="bold", multialignment="center")

    ax.set_xlabel(r"Damköhler number $\mathrm{Da}$")
    ax.set_ylabel(r"LCST sensitivity $S_\chi$")
    ax.set_title("1D slab phase diagram", fontsize=8)

    # Legend
    seen = set()
    patches = []
    for _, row in df.iterrows():
        lbl = row.get("label", "solve_failed")
        if lbl not in seen and lbl in LABEL_DISPLAY:
            seen.add(lbl)
            patches.append(mpatches.Patch(color=LABEL_COLORS.get(lbl, "#888"),
                                           label=LABEL_DISPLAY[lbl]))
    ax.legend(handles=patches, fontsize=5.5, loc="upper left",
              framealpha=0.85, ncol=1)


# ══════════════════════════════════════════════════════════════════
# Panel (b): Amplitude vs Da (from scan CSV slices)
# ══════════════════════════════════════════════════════════════════

def panel_amp_da(ax, df):
    for sc, col, lab in [(0.7, C[1], r"$S_\chi=0.7$ (Reg. I)"),
                          (1.0, C[0], r"$S_\chi=1.0$ (Reg. II)")]:
        sub = df[np.abs(df["S_chi"] - sc) < 0.05].sort_values("Da")
        if len(sub) == 0:
            continue
        Da_  = sub["Da"].values
        Jamp = sub["J_amp"].values if "J_amp" in sub else np.zeros(len(Da_))
        Tamp = sub["theta_amp"].values if "theta_amp" in sub else np.zeros(len(Da_))
        ax.plot(Da_, Jamp, color=col, lw=1.0, label=lab + r" ($\Delta J$)")

    ax2 = ax.twinx()
    for sc, col in [(0.7, C[1]), (1.0, C[0])]:
        sub = df[np.abs(df["S_chi"] - sc) < 0.05].sort_values("Da")
        if len(sub) == 0 or "theta_amp" not in sub.columns:
            continue
        ax2.plot(sub["Da"].values, sub["theta_amp"].values,
                 color=col, lw=0.9, ls=":", label=r"$\Delta\theta$")

    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"$\Delta J$", fontsize=8)
    ax2.set_ylabel(r"$\Delta \theta$", fontsize=8)
    ax.set_title(r"Oscillation amplitude", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")


# ══════════════════════════════════════════════════════════════════
# Panel (c): Penetration depth ℓ_u vs Bi_c
# ══════════════════════════════════════════════════════════════════

def penetration_depth(data):
    """Fraction of slab with time-mean u > 0.5."""
    x   = data["x"]
    u   = data["u"]
    i0  = int(0.60 * u.shape[1])
    u_mean = np.mean(u[:, i0:], axis=1)
    depth  = float(np.sum(u_mean > 0.5) / len(x))
    return depth


def panel_penetration(ax):
    print("  Computing penetration depth vs Bi_c ...")
    Bic_arr  = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    depths   = []

    for Bic in Bic_arr:
        p = Params(Da=9.5, S_chi=1.0, Bi_c=Bic, Bi_T=0.10, Gamma_A=1.5,
                   N=40, t_end=300, n_save=3000)
        try:
            d = simulate(p)
            depth = penetration_depth(d)
        except Exception as e:
            print(f"    Bi_c={Bic:.2f} failed: {e}")
            depth = np.nan
        depths.append(depth)
        print(f"    Bi_c={Bic:.2f}  ℓ_u/H₀={depth:.3f}")

    depths = np.array(depths)
    ax.plot(Bic_arr, depths, color=C[3], lw=1.1, marker="o", ms=3.5)
    ax.set_xlabel(r"Reactant Biot number $\mathrm{Bi}_c$")
    ax.set_ylabel(r"Penetration depth $\ell_u / H_0$")
    ax.set_title("Reactant penetration", fontsize=8)
    ax.annotate(r"stirring $\uparrow$", xy=(1.8, depths[-1]+0.02),
                fontsize=6.5, color=C[3], ha="right")
    print("  Penetration done.")


# ══════════════════════════════════════════════════════════════════
# Assemble Fig.4
# ══════════════════════════════════════════════════════════════════

def main():
    set_style()

    if not SCAN_CSV.exists():
        print(f"ERROR: {SCAN_CSV} not found.")
        print("Run the scan first:")
        print("  python code/scan_optimized_slab.py \\")
        print("    --x-param S_chi --x-min 0.4 --x-max 1.4 --nx 16 \\")
        print("    --y-param Da --y-min 3 --y-max 20 --ny 16 \\")
        print("    --N 40 --t-end 400 --outdir scan_results_schi_da")
        return

    df = pd.read_csv(SCAN_CSV)
    print(f"  Loaded {len(df)} scan rows. Columns: {list(df.columns)}")

    # mosaic: left panel tall (spans 2 rows), right two small panels
    fig = plt.figure(figsize=(PRE_DOUBLE, 5.5))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.42,
                           left=0.10, right=0.97, top=0.96, bottom=0.10,
                           width_ratios=[1.3, 1.0])

    ax_a  = fig.add_subplot(gs[:, 0])    # left tall panel
    ax_b  = fig.add_subplot(gs[0, 1])    # top right
    ax_c  = fig.add_subplot(gs[1, 1])    # bottom right

    panel_phase_map(ax_a, df)
    add_panel_label(ax_a, "a")

    panel_amp_da(ax_b, df)
    add_panel_label(ax_b, "b")

    panel_penetration(ax_c)
    add_panel_label(ax_c, "c")

    save(fig, "fig4_phasemap")
    print("Fig.4 done.")


if __name__ == "__main__":
    main()
