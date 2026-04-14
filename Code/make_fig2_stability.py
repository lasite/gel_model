#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
make_fig2_stability.py — Fig.1: 0D linear stability & bifurcation — 2×3 panels.

Layout: double-column (6.875 in), 2 rows × 3 columns.
  Row 1 (bifurcation structure at fixed S_χ, Bi_T, Γ_A):
    (a) J vs Da: three steady-state branches + limit-cycle envelope
    (b) Δθ vs Da: oscillation amplitude (supercritical Hopf)
    (c) max Re(λ) vs Da: eigenvalue crossing zero
  Row 2 (Hopf boundary in 2D parameter planes):
    (d) Da–Bi_T: max Re(λ) colormap + Re(λ)=0 contour
    (e) Da–S_χ: same
    (f) Da–Γ_A: same

Data source: sphere_0d_analysis.py (0D lumped model).
Parameters: χ∞=0.40, S_χ=0.5, Bi_T=0.8 (validated against existing fig2/fig5).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

from style_pub import (set_style, PRE_DOUBLE, C, COLORS,
                       add_panel_label, save)
from sphere_0d_analysis import (SphereParams, find_steady_state,
                                  find_multiple_steady_states,
                                  stability_analysis, auto_m_b,
                                  integrate_0d, detect_oscillation)

set_style()

# ─── 0D baseline (validated: produces clean Hopf at Da ≈ 1–3.5) ──
BASE = SphereParams(
    phi_p0=0.15, chi_inf=0.40, S_chi=0.5, chi1=0.80,
    Omega_e=0.10, Da=3.0, Gamma_A=2.0, eps_T=0.03,
    m_act=4.0, use_cat_density=False,
    Bi_u=1.0, Bi_T=0.8, Bi_mu=2.0, J_init=1.5, m_b=None,
)


def make_params(**kw):
    d = BASE.__dict__.copy()
    d.update(kw)
    p = SphereParams(**d)
    p.m_b = auto_m_b(p)
    return p


def find_all_branches(Da_arr):
    """For each Da, find all steady states and classify stability."""
    branches = []  # list of dicts
    for Da in Da_arr:
        p = make_params(Da=Da)
        all_ss = find_multiple_steady_states(p, n_tries=15)
        for ss in all_ss:
            info = stability_analysis(p, ss)
            branches.append({
                "Da": Da, "J": ss[0], "u": ss[1], "theta": ss[2],
                "max_real": info["max_real"],
                "hopf": info["oscillatory_unstable"],
                "stable": info["stable"],
            })
    return branches


def compute_limit_cycles(Da_arr, t_end=800.0):
    """Integrate 0D at each Da starting near the collapsed branch."""
    results = []
    for Da in Da_arr:
        p = make_params(Da=Da)
        # Find collapsed branch as starting point
        all_ss = find_multiple_steady_states(p, n_tries=15)
        if not all_ss:
            results.append({"Da": Da, "osc": False})
            continue
        collapsed = min(all_ss, key=lambda s: s[0])
        # Perturb away from the SS to trigger limit cycle
        y0 = np.array([collapsed[0]*1.1, collapsed[1]*1.1, collapsed[2]*0.9])
        sol = integrate_0d(p, y0=y0, t_end=t_end, n_pts=10000)
        if not sol.success:
            results.append({"Da": Da, "osc": False})
            continue
        osc = detect_oscillation(sol.t, sol.y[2])  # theta
        osc_J = detect_oscillation(sol.t, sol.y[0])  # J
        i0 = int(0.5 * len(sol.t))
        J_tail = sol.y[0, i0:]
        th_tail = sol.y[2, i0:]
        results.append({
            "Da": Da, "osc": osc["oscillatory"],
            "J_min": np.min(J_tail), "J_max": np.max(J_tail),
            "th_min": np.min(th_tail), "th_max": np.max(th_tail),
            "dtheta": osc["amp"], "dJ": osc_J["amp"],
            "period": osc["period"],
        })
    return results


# ══════════════════════════════════════════════════════════════════
# Row 1: Bifurcation panels
# ══════════════════════════════════════════════════════════════════

def panel_bif_J(ax, branches, lc_results):
    """(a) J vs Da: steady-state branches + limit-cycle envelope."""
    # Separate branches by J value (swollen > 1, collapsed < 0.4, saddle in between)
    for b in branches:
        if b["J"] > 0.8:
            mk, col = "o", C[0]    # swollen branch (blue)
        elif b["J"] < 0.4:
            mk, col = "s", C[1]    # collapsed branch (orange)
        else:
            mk, col = "^", C[2]    # saddle (green)
        edge = "k" if b["hopf"] else (col if b["stable"] else "red")
        ax.plot(b["Da"], b["J"], mk, ms=3, color=col, mec=edge, mew=0.4,
                alpha=0.8, zorder=3)

    # Limit-cycle envelope (shaded)
    osc_Da = [r["Da"] for r in lc_results if r["osc"]]
    J_lo   = [r["J_min"] for r in lc_results if r["osc"]]
    J_hi   = [r["J_max"] for r in lc_results if r["osc"]]
    if osc_Da:
        ax.fill_between(osc_Da, J_lo, J_hi, alpha=0.25, color=C[2],
                        label="Limit cycle", zorder=2)

    # Legend for branches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color=C[0], ls="", ms=4, label="Swollen (stable)"),
        Line2D([0],[0], marker="s", color=C[1], ls="", ms=4, label="Collapsed"),
        Line2D([0],[0], marker="^", color=C[2], ls="", ms=4, label="Saddle"),
        Line2D([0],[0], color=C[2], lw=6, alpha=0.25, label="Limit cycle"),
    ]
    ax.legend(handles=handles, fontsize=5, loc="upper right", framealpha=0.8)
    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"$J$")


def panel_bif_dtheta(ax, lc_results):
    """(b) Δθ vs Da: oscillation amplitude."""
    Da_arr = [r["Da"] for r in lc_results]
    dth = [r["dtheta"] if r["osc"] else 0.0 for r in lc_results]
    ax.plot(Da_arr, dth, "o-", color=C[1], ms=3, lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"$\Delta\theta$")


def panel_bif_eigenvalue(ax, Da_arr_fine):
    """(c) max Re(λ) vs Da for collapsed branch."""
    re_arr = []
    im_arr = []
    prev_ss = None
    for Da in Da_arr_fine:
        p = make_params(Da=Da)
        # Track collapsed branch (lowest J)
        all_ss = find_multiple_steady_states(p, n_tries=10)
        if not all_ss:
            re_arr.append(np.nan); im_arr.append(np.nan)
            continue
        # Pick the collapsed branch (smallest J)
        ss = min(all_ss, key=lambda s: s[0])
        info = stability_analysis(p, ss)
        idx = np.argmax(np.real(info["eigvals"]))
        re_arr.append(np.real(info["eigvals"][idx]))
        im_arr.append(np.abs(np.imag(info["eigvals"][idx])))

    re_arr = np.array(re_arr)
    im_arr = np.array(im_arr)

    ax.plot(Da_arr_fine, re_arr, "-", color=C[0], lw=1.2,
            label=r"max Re$(\lambda)$")
    ax.axhline(0, color="k", lw=0.5)

    # Mark Hopf crossings
    for i in range(len(re_arr) - 1):
        if np.isfinite(re_arr[i]) and np.isfinite(re_arr[i+1]):
            if re_arr[i] * re_arr[i+1] < 0:
                Da_h = Da_arr_fine[i] + (Da_arr_fine[i+1]-Da_arr_fine[i]) * \
                       (-re_arr[i]) / (re_arr[i+1]-re_arr[i])
                ax.axvline(Da_h, color=C[1], ls=":", lw=0.7, alpha=0.7)
                ax.plot(Da_h, 0, "o", mfc="white", mec=C[1], ms=4, zorder=5)
                ax.annotate(f"$\\mathrm{{Da}}_c={Da_h:.1f}$",
                            (Da_h, 0.02), fontsize=5.5, color=C[1],
                            ha="center", va="bottom")

    # Frequency on twin axis
    ax2 = ax.twinx()
    ax2.plot(Da_arr_fine, im_arr, "--", color=C[4], lw=0.8,
             label=r"$|\mathrm{Im}(\lambda)|$")
    ax2.set_ylabel(r"$|\mathrm{Im}(\lambda)|$", fontsize=7, color=C[4])
    ax2.tick_params(axis="y", colors=C[4], labelsize=6)

    ax.set_xlabel(r"$\mathrm{Da}$")
    ax.set_ylabel(r"max Re$(\lambda)$")

    lines1 = ax.get_lines()[:1]
    lines2 = ax2.get_lines()[:1]
    ax.legend(lines1 + lines2,
              [l.get_label() for l in lines1 + lines2],
              fontsize=5.5, loc="upper right")


# ══════════════════════════════════════════════════════════════════
# Row 2: 2D Hopf boundary maps
# ══════════════════════════════════════════════════════════════════

def hopf_2d_scan(x_param, x_range, y_param, y_range):
    """Compute max Re(λ) on a 2D grid; return (Re_grid, Hopf_grid)."""
    nx, ny = len(x_range), len(y_range)
    Re_grid = np.full((ny, nx), np.nan)

    for j, yv in enumerate(y_range):
        for i, xv in enumerate(x_range):
            p = make_params(**{x_param: xv, y_param: yv})
            # Multi-guess search
            ss = find_steady_state(p, p.J_init, 0.5, 0.1)
            if ss is None:
                ss = find_steady_state(p, p.J_init * 0.6, 0.3, 0.5)
            if ss is None:
                all_ss = find_multiple_steady_states(p, n_tries=8)
                # Pick collapsed branch if available
                if all_ss:
                    ss = min(all_ss, key=lambda s: s[0])
            if ss is None:
                continue
            info = stability_analysis(p, ss)
            Re_grid[j, i] = info["max_real"]

    return Re_grid


def panel_hopf_map(ax, x_range, y_range, Re_grid, xlabel, ylabel):
    """Plot a 2D Hopf boundary colormap with Re(λ)=0 contour."""
    ext = [x_range[0], x_range[-1], y_range[0], y_range[-1]]
    vmax = np.nanmax(np.abs(Re_grid))
    vmax = min(vmax, 0.5)

    im = ax.imshow(Re_grid, origin="lower", aspect="auto", extent=ext,
                   cmap="coolwarm", vmin=-vmax, vmax=vmax, rasterized=True)

    # Re(λ)=0 contour
    X, Y = np.meshgrid(x_range, y_range)
    try:
        # Mask NaN for contour
        Re_masked = np.where(np.isfinite(Re_grid), Re_grid, 0)
        cs = ax.contour(X, Y, Re_masked, levels=[0], colors="black",
                        linewidths=1.5)
    except Exception:
        pass

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ══════════════════════════════════════════════════════════════════
# Assemble Fig.1
# ══════════════════════════════════════════════════════════════════

def main():
    set_style()

    # ── Row 1 data: bifurcation structure ──
    Da_bif = np.linspace(0.3, 7.0, 30)
    print("  Finding all steady-state branches ...")
    branches = find_all_branches(Da_bif)
    print(f"    Found {len(branches)} branch points.")

    Da_lc = np.linspace(0.3, 7.0, 25)
    print("  Computing limit cycles (time integration) ...")
    lc_results = compute_limit_cycles(Da_lc, t_end=600.0)
    n_osc = sum(1 for r in lc_results if r["osc"])
    print(f"    {n_osc}/{len(lc_results)} oscillating.")

    Da_eig = np.linspace(0.3, 7.0, 80)

    # ── Row 2 data: 2D Hopf maps ──
    N2d = 30
    print(f"  Computing Da–Bi_T map ({N2d}×{N2d}) ...")
    Da_2d  = np.linspace(0.5, 6.0, N2d)
    BiT_2d = np.linspace(0.1, 2.0, N2d)
    Re_BiT = hopf_2d_scan("Da", Da_2d, "Bi_T", BiT_2d)
    print("    Done.")

    print(f"  Computing Da–S_χ map ({N2d}×{N2d}) ...")
    Schi_2d = np.linspace(0.1, 2.0, N2d)
    Re_Schi = hopf_2d_scan("Da", Da_2d, "S_chi", Schi_2d)
    print("    Done.")

    print(f"  Computing Da–Γ_A map ({N2d}×{N2d}) ...")
    GA_2d = np.linspace(0.5, 4.0, N2d)
    Re_GA = hopf_2d_scan("Da", Da_2d, "Gamma_A", GA_2d)
    print("    Done.")

    # ── Assemble figure ──
    fig = plt.figure(figsize=(PRE_DOUBLE, 4.8))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.45,
                          left=0.07, right=0.95, top=0.96, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    print("  Plotting panels ...")
    panel_bif_J(ax_a, branches, lc_results)
    add_panel_label(ax_a, "a")

    panel_bif_dtheta(ax_b, lc_results)
    add_panel_label(ax_b, "b")

    panel_bif_eigenvalue(ax_c, Da_eig)
    add_panel_label(ax_c, "c")

    panel_hopf_map(ax_d, Da_2d, BiT_2d, Re_BiT,
                   r"$\mathrm{Da}$", r"$\mathrm{Bi}_T$")
    add_panel_label(ax_d, "d")

    panel_hopf_map(ax_e, Da_2d, Schi_2d, Re_Schi,
                   r"$\mathrm{Da}$", r"$S_\chi$")
    add_panel_label(ax_e, "e")

    panel_hopf_map(ax_f, Da_2d, GA_2d, Re_GA,
                   r"$\mathrm{Da}$", r"$\Gamma_A$")
    add_panel_label(ax_f, "f")

    # Output as fig1_stability (paper's Fig.1)
    save(fig, "fig1_stability")
    print("Fig.1 (stability) done.")


if __name__ == "__main__":
    main()
