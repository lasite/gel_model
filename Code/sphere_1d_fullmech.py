#!/usr/bin/env python3
"""
sphere_1d_fullmech.py — 1D 球坐标 + 完整球力学

关键改进 (相比 sphere_1d.py):
  1. 给定 J(R), 重建 r(R) → λ_r = JR²/r², λ_θ = r/R (一般 λ_r ≠ λ_θ)
  2. 弹性化学势用真实 (λ_r, λ_θ): m_el = Ω_e[(I₁/(3J) - 1/J]
  3. 应力平衡 dσ_r/dr + (2/r)(σ_r-σ_θ)=0 → 非局域修正 Σ_r(R)
  4. m_el_full = m_el_local + Ω_e·Σ_r  (各向异性应力对化学势的修正)

可开关:  elastic_model='full' (完整力学) / 'isotropic' (各向同性近似)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import lil_matrix, csc_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time, os, shutil


# ═══════════════════════════════════════════════════════════════
@dataclass
class P:
    N: int = 51
    t_end: float = 600.0
    n_save: int = 6000
    method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float = 0.3

    phi_p0: float = 0.15
    chi_inf: float = 0.40
    S_chi: float = 0.50
    chi1: float = 0.80
    Omega_e: float = 0.10
    ell: float = 0.008

    Da: float = 2.5
    delta: float = 0.10
    alpha: float = 0.30
    Gamma_A: float = 1.0
    eps_T: float = 0.03
    arrh_cap: float = 50.0

    m_act: float = 4.0
    m_diff: float = 2.0
    m_mob: float = 1.0
    use_cat_density: bool = False

    Bi_mu: float = 1.0
    Bi_c: float = 1.0
    Bi_T: float = 0.80

    m_b: float = 0.0
    auto_m_b: bool = True

    M0: float = 1.0
    D0: float = 1.0
    C0: float = 1.0
    K0: float = 1.0

    # 'full' = 完整力学, 'isotropic' = 各向同性近似
    elastic_model: str = 'full'

    J_init: float = 1.50       # 用于 m_b
    J_start: float = 0.35      # 实际初始
    u_init: float = 0.50
    theta_init: float = 1.70
    eps_pert: float = 0.02

    J_min_factor: float = 1.02
    J_max: float = 8.0
    u_floor: float = 1e-12
    theta_clip: float = 25.0

_LOG_J_MAX = np.log(8.0)

# ═══════════════════════════════════════════════════════════════
# Spherical grid
# ═══════════════════════════════════════════════════════════════

class Grid:
    def __init__(self, N):
        self.N = N
        self.dR = 1.0 / N
        self.R_f = np.linspace(0, 1, N + 1)
        self.R_c = 0.5 * (self.R_f[:-1] + self.R_f[1:])
        self.R2_f = self.R_f ** 2
        self.V = (self.R_f[1:] ** 3 - self.R_f[:-1] ** 3) / 3.0
        self.V_safe = np.maximum(self.V, 1e-30)

    def sph_div(self, flux):
        return (self.R2_f[1:] * flux[1:] - self.R2_f[:-1] * flux[:-1]) / self.V_safe

    def sph_laplacian(self, a):
        dR = self.dR
        grad = np.zeros(self.N + 1)
        grad[1:self.N] = (a[1:] - a[:-1]) / dR
        return self.sph_div(grad)


# ═══════════════════════════════════════════════════════════════
# Full spherical kinematics + stress equilibrium
# ═══════════════════════════════════════════════════════════════

def reconstruct_kinematics(J, grid):
    """
    Given J(R), compute r(R), λ_r, λ_θ.
    r³(R) = 3∫₀ᴿ J(R')R'² dR'
    λ_θ = r/R,  λ_r = J/λ_θ² = JR²/r²
    """
    N = grid.N
    r3_f = np.zeros(N + 1)
    for i in range(N):
        r3_f[i + 1] = r3_f[i] + 3.0 * J[i] * grid.V[i]
    r_f = np.cbrt(np.maximum(r3_f, 1e-30))
    r_c = 0.5 * (r_f[:-1] + r_f[1:])

    lam_theta = np.empty(N)
    # R=0: L'Hôpital lim_{R→0} r/R = (dr/dR)|_0, at center λ_r=λ_θ=J^{1/3}
    lam_theta[0] = J[0] ** (1.0 / 3.0)
    lam_theta[1:] = r_c[1:] / grid.R_c[1:]
    lam_r = J / np.maximum(lam_theta ** 2, 1e-12)

    return r_c, lam_r, lam_theta


def stress_equilibrium(J, lam_r, lam_theta, r_c, grid):
    """
    Solve radial stress equilibrium from free surface inward.

    Cauchy stress: σ_r - σ_θ = (G/J)(λ_r² - λ_θ²)
    dσ_r/dr + (2/r)(σ_r - σ_θ) = 0
    In material coords: dσ_r/dR = -(2λ_r/r)(G/J)(λ_r² - λ_θ²)
    BC: σ_r(surface) = 0

    Returns Σ_r = σ_r/G  (dimensionless)
    """
    N = grid.N
    dR = grid.dR
    Sigma_r = np.zeros(N)

    for i in range(N - 2, -1, -1):
        # Integrand at i and i+1
        r_ip1 = max(r_c[i + 1], 1e-12)
        f_ip1 = (2 * lam_r[i + 1] / r_ip1) * (lam_r[i + 1] ** 2 - lam_theta[i + 1] ** 2) / J[i + 1]

        r_i = max(r_c[i], 1e-12)
        f_i = (2 * lam_r[i] / r_i) * (lam_r[i] ** 2 - lam_theta[i] ** 2) / J[i]

        # Trapezoidal: Σ_r(i) = Σ_r(i+1) + ∫ dR  (integrating inward from surface)
        Sigma_r[i] = Sigma_r[i + 1] + 0.5 * (f_ip1 + f_i) * dR

    return Sigma_r


# ═══════════════════════════════════════════════════════════════
# Constitutive laws
# ═══════════════════════════════════════════════════════════════

def phi_from_J(J, p):
    return np.clip(p.phi_p0 / np.maximum(J, p.phi_p0 * p.J_min_factor), 0, 0.999)

def harmonic_mean(a, b):
    return 2.0 * a * b / np.maximum(a + b, 1e-30)

def mixing_chem_pot(J, theta, p):
    phi = phi_from_J(J, p)
    chi = p.chi_inf + p.S_chi * theta + p.chi1 * phi
    return np.log(np.maximum(1 - phi, 1e-15)) + phi + chi * phi ** 2

def elastic_chem_pot_iso(J, p):
    """Isotropic: m_el = Ω_e(J^{-1/3} - 1/J)"""
    return p.Omega_e * (J ** (-1.0 / 3.0) - 1.0 / J)

def elastic_chem_pot_full(J, lam_r, lam_theta, Sigma_r, p):
    """
    Full mechanics elastic chemical potential.

    Local part: Ω_e · [(I₁/(3J) - 1/J]  where I₁ = λ_r² + 2λ_θ²
      → Reduces to Ω_e(J^{-1/3} - 1/J) for isotropic (✓)

    Nonlocal part: Ω_e · Σ_r  (stress equilibrium correction)
      → Σ_r = 0 for uniform J (✓)

    m_el_full = m_el_local + Ω_e · Σ_r
    """
    I1 = lam_r ** 2 + 2 * lam_theta ** 2
    m_el_local = p.Omega_e * (I1 / (3.0 * J) - 1.0 / J)
    m_el_nonlocal = p.Omega_e * Sigma_r
    return m_el_local + m_el_nonlocal

def thermal_factor(theta, p):
    denom = 1.0 + p.eps_T * np.maximum(theta, -0.95 / p.eps_T)
    exp_arg = np.clip(p.Gamma_A * theta / denom, -p.arrh_cap, p.arrh_cap)
    return np.exp(exp_arg)

def reaction_rate(u, theta, J, p):
    phi = phi_from_J(J, p)
    act = np.maximum(1 - phi, 1e-12) ** p.m_act
    th = thermal_factor(theta, p)
    u_s = np.maximum(u, p.u_floor)
    cat = phi if p.use_cat_density else 1.0
    return u_s * cat * act * th

def finalize_params(p):
    if p.auto_m_b:
        J0 = np.array([p.J_init])
        m_mix = mixing_chem_pot(J0, np.array([0.0]), p)
        m_el = elastic_chem_pot_iso(J0, p)
        p = replace(p, m_b=float(m_mix[0] + m_el[0]))
    return p


# ═══════════════════════════════════════════════════════════════
# RHS
# ═══════════════════════════════════════════════════════════════

def rhs(t, y, p, grid):
    N = p.N
    dR = grid.dR
    log_J_min = np.log(p.phi_p0 * p.J_min_factor)

    logJ = np.clip(y[:N], log_J_min, _LOG_J_MAX)
    W = y[N:2 * N]
    theta = np.clip(y[2 * N:], -10, p.theta_clip)

    J = np.exp(logJ)
    u = np.maximum(W / J, p.u_floor)
    phi = phi_from_J(J, p)

    # ─── Chemical potential ───
    m_mix = mixing_chem_pot(J, theta, p)

    if p.elastic_model == 'full':
        r_c, lam_r, lam_theta = reconstruct_kinematics(J, grid)
        Sigma_r = stress_equilibrium(J, lam_r, lam_theta, r_c, grid)
        m_el = elastic_chem_pot_full(J, lam_r, lam_theta, Sigma_r, p)
    else:
        m_el = elastic_chem_pot_iso(J, p)

    m_local = m_mix + m_el
    J_lap = grid.sph_laplacian(J)
    m = m_local - p.ell ** 2 * J_lap

    # ─── Solvent flux ───
    q = np.zeros(N + 1)
    M_cell = p.M0 * np.maximum(1 - phi, 1e-12) ** p.m_mob
    M_face = harmonic_mean(M_cell[:-1], M_cell[1:])
    q[1:N] = -M_face * (m[1:] - m[:-1]) / dR
    q[N] = p.Bi_mu * (m[-1] - p.m_b)

    div_q = grid.sph_div(q)
    logJ_t = -div_q / J

    # ─── Reaction ───
    R_rate = reaction_rate(u, theta, J, p)

    # ─── Substrate flux ───
    nflux = np.zeros(N + 1)
    D_cell = p.D0 * np.maximum(1 - phi, 1e-12) ** p.m_diff
    D_face = harmonic_mean(D_cell[:-1], D_cell[1:])
    q_int = q[1:N]
    u_up = np.where(q_int >= 0, u[:-1], u[1:])
    nflux[1:N] = q_int * u_up - p.delta * D_face * (u[1:] - u[:-1]) / dR
    nflux[N] = p.Bi_c * (u[-1] - 1.0)
    W_t = -grid.sph_div(nflux) - p.Da * J * R_rate

    # ─── Heat ───
    h = np.zeros(N + 1)
    h[1:N] = -p.alpha * p.K0 * (theta[1:] - theta[:-1]) / dR
    h[N] = p.Bi_T * theta[-1]
    theta_t = (-grid.sph_div(h) + p.Da * J * R_rate) / p.C0

    return np.concatenate([logJ_t, W_t, theta_t])


# ═══════════════════════════════════════════════════════════════
# Initial conditions + Solver
# ═══════════════════════════════════════════════════════════════

def make_y0(p, grid, mode='from_0d'):
    R = grid.R_c
    log_J_min = np.log(p.phi_p0 * p.J_min_factor)
    if mode == 'from_0d':
        J0 = p.J_start + p.eps_pert * np.cos(np.pi * R)
        u0 = p.u_init + 0.05 * (1 - R)
        th0 = p.theta_init + 0.05 * (1 - R ** 2)
    elif mode == 'collapsed_core':
        sm = 0.5 * (1 + np.tanh((R - 0.5) / 0.08))
        J0 = 0.25 + (1.5 - 0.25) * sm
        u0 = 0.1 + 0.6 * sm
        th0 = 1.5 * (1 - sm) + 0.1 * sm
    else:
        J0 = p.J_start * np.ones(p.N) + p.eps_pert * np.cos(np.pi * R)
        u0 = p.u_init + 0.05 * (1 - R)
        th0 = p.theta_init + 0.01 * (1 - R ** 2)
    J0 = np.maximum(J0, np.exp(log_J_min) + 1e-6)
    u0 = np.maximum(u0, p.u_floor)
    return np.concatenate([np.log(J0), J0 * u0, th0])


def make_sparsity(N):
    sz = 3 * N
    S = lil_matrix((sz, sz), dtype=np.float64)
    bw = {(0,0):2,(0,1):-1,(0,2):1,(1,0):2,(1,1):1,(1,2):1,(2,0):-1,(2,1):-1,(2,2):1}
    for (kr,kc),w in bw.items():
        if w < 0: continue
        for i in range(N):
            for dj in range(-w, w+1):
                j = i + dj
                if 0 <= j < N:
                    S[kr*N+i, kc*N+j] = 1.0
    return csc_matrix(S)


def simulate(p, mode='from_0d'):
    p = finalize_params(p)
    grid = Grid(p.N)
    y0 = make_y0(p, grid, mode)
    n3 = 3 * p.N

    rhs_fn = lambda t, y: rhs(t, y, p, grid)
    S = make_sparsity(p.N)

    t0 = time.perf_counter()
    sol = solve_ivp(rhs_fn, (0, p.t_end), y0, jac_sparsity=S,
                    method=p.method,
                    t_eval=np.linspace(0, p.t_end, p.n_save),
                    rtol=p.rtol, atol=p.atol, max_step=p.max_step)
    elapsed = time.perf_counter() - t0
    if not sol.success:
        raise RuntimeError(sol.message)

    N = p.N
    lgJmin = np.log(p.phi_p0 * p.J_min_factor)
    Jsol = np.exp(np.clip(sol.y[:N], lgJmin, _LOG_J_MAX))
    Wsol = sol.y[N:2*N]
    thsol = sol.y[2*N:]
    usol = np.maximum(Wsol / Jsol, p.u_floor)

    # Surface radius
    r_surf = np.zeros(len(sol.t))
    for ti in range(len(sol.t)):
        r3 = np.sum(3 * Jsol[:, ti] * grid.V)
        r_surf[ti] = r3 ** (1./3.)

    # Kinematics at final time
    r_c_f, lr_f, lt_f = reconstruct_kinematics(Jsol[:, -1], grid)
    Sr_f = stress_equilibrium(Jsol[:, -1], lr_f, lt_f, r_c_f, grid)

    return {
        "R": grid.R_c, "t": sol.t, "J": Jsol, "u": usol, "theta": thsol,
        "phi": phi_from_J(Jsol, p), "r_surface": r_surf, "grid": grid,
        "nfev": sol.nfev, "elapsed": elapsed,
        "lam_r_final": lr_f, "lam_theta_final": lt_f,
        "Sigma_r_final": Sr_f, "r_c_final": r_c_f,
    }


# ═══════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════

def osc_metrics(t, y, frac=0.5, floor=1e-3):
    i0 = int(len(t) * frac)
    tt, yy = t[i0:], y[i0:]
    if len(tt) < 30:
        return {"osc": False, "amp": 0, "T": np.nan, "np": 0}
    yd = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
    amp = float(np.max(yd) - np.min(yd))
    pks, _ = find_peaks(yd, prominence=max(0.15*amp, floor), distance=max(3, len(yd)//20))
    T = float(np.mean(np.diff(tt[pks]))) if len(pks) >= 2 else np.nan
    return {"osc": amp > floor and len(pks) >= 2, "amp": amp, "T": T, "np": len(pks)}


def vol_avg(field, grid):
    wt = grid.V / np.sum(grid.V)
    return np.sum(field * wt[:, None], axis=0)


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def plot_all(data, p, outdir, label=""):
    R, t = data["R"], data["t"]
    J, u, theta, phi = data["J"], data["u"], data["theta"], data["phi"]
    r_surf = data["r_surface"]
    grid = data["grid"]
    sfx = f"_{label}" if label else ""

    wt = grid.V / np.sum(grid.V)
    Jm = vol_avg(J, grid); um = vol_avg(u, grid); thm = vol_avg(theta, grid)
    Jstd = np.sqrt(vol_avg((J - Jm[None,:])**2, grid))
    thstd = np.sqrt(vol_avg((theta - thm[None,:])**2, grid))
    oJ = osc_metrics(t, Jm); oth = osc_metrics(t, thm)

    # Kymographs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, fld, ttl, cm in [
        (axes[0,0], J, "J", "viridis"), (axes[0,1], theta, r"$\theta$", "hot"),
        (axes[1,0], u, "u", "YlOrRd"), (axes[1,1], phi, r"$\phi$", "Greens"),
    ]:
        im = ax.pcolormesh(t, R, fld, shading="auto", cmap=cm)
        ax.set_xlabel("t"); ax.set_ylabel("R"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax)
    title = (f"elastic={p.elastic_model} Da={p.Da} BiT={p.Bi_T} BiM={p.Bi_mu} "
             f"$\\Omega_e$={p.Omega_e}")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{outdir}/kymo{sfx}.png", dpi=150); plt.close(fig)

    # Time series
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    axes[0].plot(t, Jm, 'b', lw=1)
    axes[0].fill_between(t, Jm-Jstd, Jm+Jstd, alpha=0.2, color='blue')
    axes[0].set_ylabel("<J>")
    osc_lbl = f"OSC T={oJ['T']:.1f}" if oJ["osc"] else "stable"
    axes[0].set_title(f"{title}  |  {osc_lbl}")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, r_surf, 'purple', lw=1); axes[1].set_ylabel("r_surf/R₀")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, um, 'orange', lw=1); axes[2].set_ylabel("<u>")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, thm, 'r', lw=1)
    axes[3].fill_between(t, thm-thstd, thm+thstd, alpha=0.2, color='red')
    axes[3].set_ylabel(r"<$\theta$>"); axes[3].grid(True, alpha=0.3)
    axes[4].plot(t, Jstd, 'b--', lw=1, label="std(J)")
    axes[4].plot(t, thstd, 'r--', lw=1, label=r"std($\theta$)")
    axes[4].set_xlabel("t"); axes[4].set_ylabel("spatial std")
    axes[4].legend(fontsize=9); axes[4].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/ts{sfx}.png", dpi=150); plt.close(fig)

    # Anisotropy at final time
    lr = data.get("lam_r_final")
    lt = data.get("lam_theta_final")
    Sr = data.get("Sigma_r_final")
    rc = data.get("r_c_final")
    if lr is not None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes[0,0]
        ax.plot(R, lr, 'b-', lw=2, label=r'$\lambda_r$')
        ax.plot(R, lt, 'r-', lw=2, label=r'$\lambda_\theta$')
        ax.plot(R, J[:,-1]**(1./3.), 'k--', lw=1, label=r'$J^{1/3}$ (iso)')
        ax.set_xlabel("R"); ax.set_ylabel("stretch"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title(r"$\lambda_r$ vs $\lambda_\theta$ (final)")

        ax = axes[0,1]
        aniso = (lr - lt) / np.maximum(0.5*(lr+lt), 1e-6)
        ax.plot(R, aniso, 'g-', lw=2)
        ax.axhline(0, color='k', ls='-', lw=0.5)
        ax.set_xlabel("R"); ax.set_ylabel(r"$(\lambda_r-\lambda_\theta)/\bar\lambda$")
        ax.set_title("Anisotropy ratio"); ax.grid(True, alpha=0.3)

        ax = axes[0,2]
        ax.plot(R, Sr, 'm-', lw=2)
        ax.axhline(0, color='k', ls='-', lw=0.5)
        ax.set_xlabel("R"); ax.set_ylabel(r"$\Sigma_r = \sigma_r/G$")
        ax.set_title("Radial stress (from equilibrium)"); ax.grid(True, alpha=0.3)

        # Deviatoric stress
        ax = axes[1,0]
        dev = (1./J[:,-1]) * (lr**2 - lt**2)
        ax.plot(R, dev, 'c-', lw=2)
        ax.set_xlabel("R"); ax.set_ylabel(r"$(\sigma_r-\sigma_\theta)/G$")
        ax.set_title("Deviatoric stress"); ax.grid(True, alpha=0.3)

        # Chemical potential components
        ax = axes[1,1]
        m_mix = mixing_chem_pot(J[:,-1], theta[:,-1], p)
        I1 = lr**2 + 2*lt**2
        m_el_loc = p.Omega_e * (I1/(3*J[:,-1]) - 1./J[:,-1])
        m_el_nonloc = p.Omega_e * Sr
        m_el_iso = elastic_chem_pot_iso(J[:,-1], p)
        ax.plot(R, m_mix, 'b-', lw=1.5, label='m_mix')
        ax.plot(R, m_el_loc, 'r-', lw=1.5, label='m_el (local)')
        ax.plot(R, m_el_nonloc, 'g-', lw=1.5, label='m_el (stress eq)')
        ax.plot(R, m_el_iso, 'k--', lw=1, label='m_el (isotropic)')
        ax.axhline(p.m_b, color='gray', ls=':', label='m_b')
        ax.set_xlabel("R"); ax.set_ylabel("m")
        ax.set_title("Chemical potential components"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # r(R) mapping
        ax = axes[1,2]
        if rc is not None:
            ax.plot(R, rc, 'b-', lw=2, label='r(R) actual')
            ax.plot(R, R * J[:,-1]**(1./3.), 'k--', lw=1, label='r(R) isotropic')
            ax.set_xlabel("R (material)"); ax.set_ylabel("r (current)")
            ax.set_title("Deformation map"); ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Full kinematics analysis (final state)", fontsize=12, fontweight="bold")
        fig.tight_layout(); fig.savefig(f"{outdir}/mech{sfx}.png", dpi=150); plt.close(fig)

    # Radial snapshots
    n_t = len(t)
    snaps = [int(f*n_t) for f in [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]]
    snaps = [min(s, n_t-1) for s in snaps]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(snaps)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for si, c in zip(snaps, colors):
        for ax, fld, yl in [(axes[0,0], J[:,si], "J"), (axes[0,1], theta[:,si], r"$\theta$"),
                            (axes[1,0], u[:,si], "u"), (axes[1,1], phi[:,si], r"$\phi$")]:
            ax.plot(R, fld, color=c, lw=1.2,
                    label=f"t={t[si]:.0f}" if ax is axes[0,0] else None)
            ax.set_ylabel(yl); ax.set_xlabel("R"); ax.grid(True, alpha=0.3)
    axes[0,0].legend(fontsize=7, ncol=2)
    fig.suptitle("Radial profiles", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{outdir}/prof{sfx}.png", dpi=150); plt.close(fig)

    return {"osc_J": oJ, "osc_theta": oth}


# ═══════════════════════════════════════════════════════════════
# Main: systematic comparison full vs isotropic
# ═══════════════════════════════════════════════════════════════

def main():
    outdir = Path("/home/claude/sphere_fullmech")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1D 球坐标: 完整力学 vs 各向同性 比较")
    print("=" * 60)

    base = dict(
        N=31, t_end=400, n_save=4000,
        chi_inf=0.40, S_chi=0.50, Da=2.5,
        Bi_T=0.80, Bi_mu=1.0, Bi_c=1.0,
        Gamma_A=1.0, m_act=4.0,
        alpha=0.30, delta=0.10,
        Omega_e=0.10, phi_p0=0.15, chi1=0.80,
        ell=0.008, J_init=1.5, J_start=0.35,
        u_init=0.50, theta_init=1.70,
        max_step=0.5, rtol=1e-5, atol=1e-7,
    )

    runs = [
        ("full_base",       "full",      {}),
        ("iso_base",        "isotropic", {}),
        ("full_Oe03",       "full",      {"Omega_e": 0.30}),
        ("iso_Oe03",        "isotropic", {"Omega_e": 0.30}),
        ("full_Oe05",       "full",      {"Omega_e": 0.50}),
        ("iso_Oe05",        "isotropic", {"Omega_e": 0.50}),
        ("full_cs_Oe03",    "full",      {"Omega_e": 0.30, "J_start": 1.5}),
        ("iso_cs_Oe03",     "isotropic", {"Omega_e": 0.30, "J_start": 1.5}),
    ]

    for label, emodel, overrides in runs:
        params = {**base, **overrides, "elastic_model": emodel}
        pp = finalize_params(P(**params))
        mode = 'collapsed_core' if 'coreshell' in label else 'from_0d'
        try:
            d = simulate(pp, mode=mode)
            info = plot_all(d, pp, outdir, label)
            Jm = vol_avg(d["J"], d["grid"])
            thm = vol_avg(d["theta"], d["grid"])
            Jf = np.mean(d["J"][:,-1]); thf = np.mean(d["theta"][:,-1])
            Jr = d["J"][:,-1]
            lr = d.get("lam_r_final", np.zeros(1))
            lt = d.get("lam_theta_final", np.zeros(1))
            aniso_max = np.max(np.abs(lr - lt) / np.maximum(0.5*(lr+lt), 1e-6)) if len(lr)>1 else 0
            oJ = info["osc_J"]; oth = info["osc_theta"]
            status = "OSC" if oth["osc"] or oJ["osc"] else "stab"
            print(f"  {label:20s}: {status} J={Jf:.3f} θ={thf:.3f} "
                  f"aniso={aniso_max:.3f} amp_θ={oth['amp']:.4f} [{d['elapsed']:.1f}s]")
        except Exception as e:
            print(f"  {label:20s}: FAIL {str(e)[:60]}")

    # Copy to outputs
    out_final = Path("/mnt/user-data/outputs/sphere_fullmech")
    if out_final.exists():
        shutil.rmtree(out_final)
    shutil.copytree(outdir, out_final)
    shutil.copy(__file__, out_final / "sphere_1d_fullmech.py")
    print(f"\n→ {out_final}")


if __name__ == "__main__":
    main()
