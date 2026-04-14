#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig1_mechanism.py — Fig.2: System schematic + LCST constitutive response
                          + annotated oscillation cycle + phase portrait.

Layout: double-column (6.875 in), 2×2 panels.
  (a) [top-left]  Slab-in-bath schematic (matplotlib patches, no data)
  (b) [top-right] LCST constitutive response J_eq(θ): swollen ↔ collapsed branches
  (c) [bottom-left]  Annotated time series ⟨J⟩(τ) and ⟨θ⟩(τ) with phase labels
  (d) [bottom-right] J–θ phase portrait showing limit cycle

Data source: 1D slab simulate() at Da=9.5, S_chi=1.0 (Regime II).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from scipy.optimize import brentq
from dataclasses import replace

from style_pub import (set_style, PRE_DOUBLE, C, COLORS,
                       add_panel_label, save, OUT)

# ─── Import slab model ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from scan_optimized import Params, simulate, finalize_params, local_chem_pot

set_style()

_FIG2_CACHE = os.path.join(os.path.dirname(__file__), "../figures_pub/fig2_mechanism_cache.npz")
_FIG2_CACHE_VERSION = 3

# ══════════════════════════════════════════════════════════════════
# Panel (a): System schematic
# ══════════════════════════════════════════════════════════════════

def draw_schematic(ax):
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # gel slab body
    gel = FancyBboxPatch((0, 0), 1.4, 1.8,
                         boxstyle="round,pad=0.04",
                         facecolor="#c8e6fa", edgecolor="#1f77b4", linewidth=1.2)
    ax.add_patch(gel)

    # reaction zone shading (near x=L, right side)
    rxn = FancyBboxPatch((0.95, 0), 0.45, 1.8,
                         boxstyle="square,pad=0",
                         facecolor="#ffd699", edgecolor="none", alpha=0.7)
    ax.add_patch(rxn)
    ax.text(1.18, 0.9, "reaction\nzone", ha="center", va="center",
            fontsize=6.5, color="#8B4513", style="italic",
            rotation=90)

    # bath (right of gel)
    bath = FancyBboxPatch((1.52, 0), 0.58, 1.8,
                          boxstyle="round,pad=0.04",
                          facecolor="#eaf5e9", edgecolor="#2ca02c", linewidth=1.0,
                          linestyle="--")
    ax.add_patch(bath)
    ax.text(1.81, 0.9, r"H$_2$O$_2$ bath", ha="center", va="center",
            fontsize=6.5, color="#2ca02c", rotation=90)

    # symmetry line at x=0
    ax.plot([0, 0], [0, 1.8], color="#555", lw=1.0, ls="--")
    ax.text(0, -0.15, r"$x=0$" + "\n symmetry", ha="center", va="top", fontsize=7)

    # boundary at x=L
    ax.text(1.4, -0.15, r"$x=H_0$" + "\n exchange", ha="center", va="top", fontsize=7)

    # arrows: H2O2 diffusing in (right→left)
    for y in [0.45, 0.9, 1.35]:
        ax.annotate("", xy=(1.0, y), xytext=(1.55, y),
                    arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                    lw=0.8, mutation_scale=7))

    # arrows: heat escaping (up)
    for x in [0.4, 0.9, 1.2]:
        ax.annotate("", xy=(x, 2.05), xytext=(x, 1.82),
                    arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                    lw=0.8, mutation_scale=7))
    ax.text(0.7, 2.1, "heat loss", ha="center", va="bottom",
            fontsize=6.5, color="#d62728")

    # solvent exchange arrows (both directions at gel surface)
    ax.annotate("", xy=(1.55, 1.3), xytext=(1.4, 1.3),
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                lw=0.8, mutation_scale=7))
    ax.annotate("", xy=(1.4, 1.05), xytext=(1.55, 1.05),
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                lw=0.8, mutation_scale=7))
    ax.text(0.68, 0.9, "PNIPAM\ngel", ha="center", va="center",
            fontsize=7, color="#1f77b4", fontweight="bold")

    # catalyst label
    ax.text(1.18, 0.2, r"Pt NPs", ha="center", va="bottom",
            fontsize=6, color="#8B4513")

    # x-axis arrow
    ax.annotate("", xy=(1.45, -0.22), xytext=(0.05, -0.22),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.7, mutation_scale=7))
    ax.text(0.72, -0.28, r"$x$", ha="center", va="top", fontsize=8)


# ══════════════════════════════════════════════════════════════════
# Cached data builders
# ══════════════════════════════════════════════════════════════════

def compute_constitutive_data():
    p0 = finalize_params(Params(S_chi=1.0, Da=9.5))
    m_b = float(local_chem_pot(np.array([p0.J_init]), np.array([0.0]), p0)[0])

    theta_range = np.linspace(-0.5, 3.0, 800)
    Jmin, Jmax = p0.phi_p0 * 1.02, 6.0

    def root_fn(J, th):
        return float(local_chem_pot(np.array([J]), np.array([th]), p0)[0]) - m_b

    def all_roots(th):
        grid = np.linspace(Jmin, Jmax, 800)
        vals = np.array([root_fn(j, th) for j in grid])
        roots = []
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if np.isnan(fa) or np.isnan(fb):
                continue
            if fa * fb < 0:
                try:
                    r = brentq(lambda J: root_fn(J, th), a, b, xtol=1e-10, maxiter=200)
                    if not any(abs(r - r0) < 1e-5 for r0 in roots):
                        roots.append(r)
                except Exception:
                    pass
        return sorted(roots)

    upper = np.full_like(theta_range, np.nan)
    lower = np.full_like(theta_range, np.nan)
    middle = np.full_like(theta_range, np.nan)

    for i, th in enumerate(theta_range):
        roots = all_roots(float(th))
        if not roots:
            continue
        if len(roots) == 1:
            upper[i] = roots[0]
            continue

        # classify by slope d(mu-m_b)/dJ: positive = outer stable branch
        slopes = []
        for r in roots:
            h = 1e-5
            slopes.append((root_fn(r + h, th) - root_fn(r - h, th)) / (2 * h))

        stable_roots = [r for r, s in zip(roots, slopes) if s > 0]
        unstable_roots = [r for r, s in zip(roots, slopes) if s < 0]

        if stable_roots:
            upper[i] = max(stable_roots)
            lower[i] = min(stable_roots)
        if unstable_roots:
            middle[i] = unstable_roots[0]

    return {
        "theta_const": theta_range,
        "upper_const": upper,
        "lower_const": lower,
        "middle_const": middle,
    }


def run_regime_ii():
    print("  Running Regime II simulation (Da=9.5, S_chi=1.0, N=40, t_end=350) ...")
    p = Params(Da=9.5, S_chi=1.0, N=40, t_end=350, n_save=5000,
               Bi_c=0.70, Bi_T=0.10, Gamma_A=1.5)
    data = simulate(p)
    print(f"  Done. nfev={data['nfev']}")
    return data


def extract_cycle_data(data):
    t = np.asarray(data["t"])
    J_mean = np.mean(data["J"], axis=0)
    theta_mean = np.mean(data["theta"], axis=0)
    J_probe = np.asarray(data["J"][-1])

    # Use the x=H0 boundary signal to identify macro-cycles cleanly:
    # the boundary node is free of the MMO-like interior sub-peaks.
    i0 = int(0.55 * len(t))
    t_tail = t[i0:] - t[i0]
    J_tail = J_mean[i0:]
    th_tail = theta_mean[i0:]
    probe_tail = J_probe[i0:]

    from scipy.signal import find_peaks
    probe_prom = max(0.015, 0.12 * np.ptp(probe_tail))
    peaks, _ = find_peaks(probe_tail, prominence=probe_prom, distance=25)

    if len(peaks) >= 4:
        start = peaks[-4]
        stop = peaks[-1]
        t_tail = t_tail[start:stop] - t_tail[start]
        J_tail = J_tail[start:stop]
        th_tail = th_tail[start:stop]
        peaks = peaks[-4:] - start

        ref_start = peaks[1]
        ref_stop = peaks[2]
        phase_frac = np.array([0.18, 0.42, 0.68, 0.90])
        phase_idx = ref_start + np.rint(phase_frac * (ref_stop - ref_start)).astype(int)
    else:
        phase_idx = np.linspace(0, len(t_tail) - 1, 4).astype(int)

    phase_idx = np.clip(phase_idx, 0, len(t_tail) - 1)
    return {
        "cache_version": np.array([_FIG2_CACHE_VERSION], dtype=int),
        "t_full": t,
        "J_mean_full": J_mean,
        "theta_mean_full": theta_mean,
        "t_cycle": t_tail,
        "J_cycle": J_tail,
        "theta_cycle": th_tail,
        "phase_idx": phase_idx,
    }


def load_or_build_fig2_data():
    if os.path.exists(_FIG2_CACHE):
        print("  Loading Fig.2 cache ...")
        d = np.load(_FIG2_CACHE)
        required = {"theta_const", "upper_const", "lower_const", "middle_const",
                    "t_full", "J_mean_full", "theta_mean_full",
                    "t_cycle", "J_cycle", "theta_cycle", "phase_idx", "cache_version"}
        if required.issubset(set(d.files)) and int(np.atleast_1d(d["cache_version"])[0]) == _FIG2_CACHE_VERSION:
            return {k: d[k] for k in d.files}
        print("  Cache missing required fields; rebuilding ...")

    const = compute_constitutive_data()
    sim = run_regime_ii()
    cyc = extract_cycle_data(sim)
    cache = {**const, **cyc}
    np.savez_compressed(_FIG2_CACHE, **cache)
    return cache


# ══════════════════════════════════════════════════════════════════
# Panel (b): LCST constitutive response J_eq(θ)
# ══════════════════════════════════════════════════════════════════

def constitutive_curve(ax, cache):
    theta_range = cache["theta_const"]
    upper = cache["upper_const"]
    lower = cache["lower_const"]
    middle = cache["middle_const"]

    # Plot equilibrium branches
    mask_u = ~np.isnan(upper)
    if np.any(mask_u):
        ax.plot(theta_range[mask_u], upper[mask_u], color=C[0], lw=1.2,
                label="swollen branch")

    mask_l = ~np.isnan(lower) & (lower < upper - 1e-6)
    if np.any(mask_l):
        ax.plot(theta_range[mask_l], lower[mask_l], color=C[1], lw=1.2,
                ls="--", label="collapsed branch")

    mask_m = ~np.isnan(middle)
    if np.any(mask_m):
        ax.plot(theta_range[mask_m], middle[mask_m], color="0.45", lw=1.0,
                ls=":", label="unstable branch")

    # LCST crossover band
    ax.axvspan(0.88, 1.12, color="0.85", alpha=0.6, zorder=0)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.8)

    ymax = np.nanmax(np.r_[upper[mask_u], lower[mask_l] if np.any(mask_l) else [np.nan]])
    ymin = np.nanmin(np.r_[upper[mask_u], lower[mask_l] if np.any(mask_l) else [np.nan]])
    ax.text(1.0, ymax * 0.98, "LCST\ncrossover", fontsize=6.2, color="0.35",
            va="top", ha="center")
    ax.text(0.10, 1.20, "cold swollen", fontsize=6.2, color=C[0],
            ha="left", va="center")
    ax.text(1.95, 0.16, "hot collapsed", fontsize=6.2, color=C[1],
            ha="left", va="center")

    # Heating / cooling trajectories (content reference only; same layout)
    if np.any(mask_u):
        iu0 = np.where(mask_u)[0][int(0.30 * np.sum(mask_u))]
        iu1 = np.where(mask_u)[0][int(0.70 * np.sum(mask_u))]
        ax.annotate("", xy=(theta_range[iu1], upper[iu1]), xytext=(theta_range[iu0], upper[iu0]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C[0]))
        ax.text(theta_range[iu0] + 0.10, upper[iu0] + 0.05, "heating", color=C[0], fontsize=6.0)

    if np.any(mask_l):
        il0 = np.where(mask_l)[0][int(0.75 * np.sum(mask_l))]
        il1 = np.where(mask_l)[0][int(0.25 * np.sum(mask_l))]
        ax.annotate("", xy=(theta_range[il1], lower[il1]), xytext=(theta_range[il0], lower[il0]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C[1], ls="--"))
        ax.text(theta_range[il0] - 0.45, lower[il0] + 0.05, "cooling", color=C[1], fontsize=6.0)

    ax.set_xlabel(r"Temperature $\theta = \Delta T / \Delta T^*$")
    ax.set_ylabel(r"Equilibrium swelling ratio $J_\mathrm{eq}$")
    ax.set_title("LCST constitutive response", fontsize=8)
    ax.legend(loc="center right", fontsize=6.0, framealpha=0.9)
    ax.set_xlim(-0.2, 2.3)
    ax.set_ylim(0.12, 1.36)


# ══════════════════════════════════════════════════════════════════
# Panel (c): Annotated time series
# ══════════════════════════════════════════════════════════════════

def plot_timeseries(ax, cache):
    t_tail = cache["t_cycle"]
    J_tail = cache["J_cycle"]
    th_tail = cache["theta_cycle"]
    phase_idx = cache["phase_idx"].astype(int)

    ax2 = ax.twinx()
    lJ, = ax.plot(t_tail, J_tail, color=C[0], lw=1.2, label=r"$\langle J \rangle$")
    lT, = ax2.plot(t_tail, th_tail, color=C[1], lw=1.0, ls="--",
                   label=r"$\langle \theta \rangle$")

    # Phase annotations
    phase_x = t_tail[phase_idx]
    y_annot = J_tail.max() - 0.04 * np.ptp(J_tail)
    for n, px in enumerate(phase_x, start=1):
        ax.axvline(px, color="gray", ls=":", lw=0.45, alpha=0.45)
        ax.text(px, y_annot, str(n), ha="center", va="top", fontsize=5.6, color="0.25",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.75", lw=0.5))

    ax.set_xlabel(r"Time $\tau$")
    ax.set_ylabel(r"$\langle J \rangle$", color=C[0])
    ax2.set_ylabel(r"$\langle \theta \rangle$", color=C[1])
    ax.tick_params(axis="y", colors=C[0])
    ax2.tick_params(axis="y", colors=C[1])
    ax.set_title("Several representative cycles", fontsize=8)

    lines = [lJ, lT]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=6.5)


# ══════════════════════════════════════════════════════════════════
# Panel (d): Phase portrait
# ══════════════════════════════════════════════════════════════════

def plot_phase_portrait(ax, cache):
    J_tail = cache["J_cycle"]
    th_tail = cache["theta_cycle"]
    phase_idx = cache["phase_idx"].astype(int)

    ax.plot(J_tail, th_tail, color=C[0], lw=0.9, alpha=0.8)

    # Arrow showing direction of traversal
    mid = len(J_tail) // 4
    ax.annotate("", xy=(J_tail[mid+3], th_tail[mid+3]),
                xytext=(J_tail[mid], th_tail[mid]),
                arrowprops=dict(arrowstyle="-|>", color=C[0],
                                lw=0.8, mutation_scale=8))

    # Mark representative phase locations consistent with panel (c)
    for n, idx in enumerate(phase_idx, start=1):
        ax.plot(J_tail[idx], th_tail[idx], 'o', ms=2.8, color='k', zorder=5)
        ax.text(J_tail[idx] + 0.015, th_tail[idx] + 0.01, str(n), fontsize=5.8, color='k')

    ax.set_xlabel(r"$\langle J \rangle$ (swelling ratio)")
    ax.set_ylabel(r"$\langle \theta \rangle$ (temperature)")
    ax.set_title("Limit cycle", fontsize=8)


# ══════════════════════════════════════════════════════════════════
# Assemble Fig.2
# ══════════════════════════════════════════════════════════════════

def main():
    set_style()
    cache = load_or_build_fig2_data()

    fig = plt.figure(figsize=(PRE_DOUBLE, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38,
                          left=0.09, right=0.97, top=0.97, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Schematic
    draw_schematic(ax_a)
    add_panel_label(ax_a, "a", x=-0.05, y=1.04)

    # (b) Constitutive curve
    constitutive_curve(ax_b, cache)
    add_panel_label(ax_b, "b")

    # (c) Time series
    plot_timeseries(ax_c, cache)
    add_panel_label(ax_c, "c")

    # (d) Phase portrait
    plot_phase_portrait(ax_d, cache)
    add_panel_label(ax_d, "d")

    save(fig, "fig2_mechanism")
    print("Fig.2 done.")


if __name__ == "__main__":
    main()
