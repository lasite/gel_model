#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sphere_1d.py — 1D 球对称 LCST 凝胶球空间分辨模型

模型变量（球对称, 参考坐标 s ∈ [0,1]）
---------------------------------------
状态向量采用
    y = [logJ_0..N-1, W_0..N-1, theta_0..N-1]
其中
    J(s,t)     : 局部溶胀比 / 体积比
    u(s,t)     : 底物浓度
    W = J*u    : 守恒变量（避免体积变化项单独处理）
    theta(s,t) : 无量纲温度

控制方程（reduced spherical J-u-theta model）
---------------------------------------------
    Q_s = - M(J,theta) * d_s mu
    logJ_t = -(1/J) * [ (1/s^2) d_s (s^2 Q_s) ]

    N_u = u Q_s - delta * D(J) * d_s u
    W_t = - [ (1/s^2) d_s (s^2 N_u) ] - Da * J * R(u,theta,J)

    h = - alpha * K0 * d_s theta
    theta_t = ( - [ (1/s^2) d_s (s^2 h) ] + Da * J * R(u,theta,J) ) / C0

    mu = m_local(J,theta) - ell^2 * Delta_s J

其中 Delta_s J = (1/s^2) d_s (s^2 d_s J)

边界条件
--------
中心 s=0:
    Q_s = 0, N_u = 0, h = 0, d_s J = 0
外表面 s=1:
    Q_s = Bi_mu * (mu - m_b)
    N_u = Bi_c  * (u - 1)
    h   = Bi_T  * theta

说明
----
1. 这是从 slab 代码平滑过渡到球坐标的 reduced spherical PDE，保留球坐标几何、
   表面供料/散热、LCST 塌缩抑制输运等核心机制。
2. 第一阶段不显式求解 r(R,t), lambda_r, lambda_theta 的严格球力学平衡；
   采用以 J 为主变量的 reduced poro-thermo-chemical closure。
3. 支持两种弹性化学势：
     - sphere elastic : m_el = Omega_e * (J^(-1/3) - 1/J)
     - slab   elastic : m_el = Omega_e * (J - 1/J)
4. 支持是否乘入催化剂密度因子 phi = phi_p0/J。
5. 提供单次模拟 + Da 扫描两种模式。

推荐开发顺序
------------
默认先用：
    use_sphere_elastic = False
    use_cat_density    = False
先验证仅仅换成球坐标几何后，是否仍能得到振荡/壳层门控；
然后再打开更“球珠化”的本构：
    use_sphere_elastic = True
    use_cat_density    = True
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import csc_matrix, lil_matrix


# ============================================================================
# Parameters
# ============================================================================

@dataclass
class Params:
    # ---------- numerics ----------
    N: int = 121
    t_end: float = 500.0
    n_save: int = 5000
    method: str = "BDF"
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    max_step: float = 0.5

    # ---------- constitutive / working point ----------
    phi_p0: float = 0.15
    chi_inf: float = 0.60
    S_chi: float = 1.00
    chi1: float = 1.10
    Omega_e: float = 0.12
    ell: float = 0.01

    use_sphere_elastic: bool = False   # False: slab-like closure, True: isotropic-sphere closure
    use_cat_density: bool = False      # False: no phi factor, True: reaction rate multiplied by phi

    # ---------- transport / reaction ----------
    Da: float = 4.0
    delta: float = 0.08
    alpha: float = 0.20
    Gamma_A: float = 1.5
    eps_T: float = 0.03
    arrh_exp_cap: float = 60.0
    n_react: int = 1

    m_act: float = 6.0
    m_diff: float = 2.0
    m_mob: float = 1.0
    M0: float = 1.0
    D0: float = 2.0
    C0: float = 1.0
    K0: float = 1.0

    # ---------- boundary exchange ----------
    Bi_mu: float = 1.00
    Bi_c: float = 0.70
    Bi_T: float = 0.10
    m_b: float = 0.0
    auto_set_m_b: bool = True

    # ---------- optional thermal trigger ----------
    use_hill: bool = False
    theta_c: float = 0.50
    n_hill: float = 4.0
    hill_eps: float = 0.05

    # ---------- initial condition ----------
    J_init: float = 1.30
    u_init: float = 0.02
    theta_init: float = 0.0
    eps_J: float = 5.0e-3
    eps_u: float = 5.0e-3
    eps_theta: float = 1.0e-4

    # ---------- safeguards ----------
    J_min: float = 0.18
    J_max: float = 8.0
    u_floor: float = 1.0e-12
    phi_floor: float = 1.0e-10
    phi_ceiling: float = 1.0 - 1.0e-10
    theta_clip: float = 25.0


# ============================================================================
# Grid geometry (spherical shells)
# ============================================================================

def make_grid(N: int) -> Dict[str, np.ndarray]:
    ds = 1.0 / N
    s_face = np.linspace(0.0, 1.0, N + 1)
    s_cell = 0.5 * (s_face[:-1] + s_face[1:])
    area_face = s_face**2                    # 4π omitted (cancels everywhere)
    vol_cell = (s_face[1:]**3 - s_face[:-1]**3) / 3.0
    return {
        "ds": ds,
        "s_face": s_face,
        "s_cell": s_cell,
        "A": area_face,
        "V": vol_cell,
    }


# ============================================================================
# Constitutive laws
# ============================================================================

def phi_from_J(J: np.ndarray, p: Params, phi_hard_ceil: float = 0.995) -> np.ndarray:
    J_safe = np.maximum(J, p.phi_p0 / phi_hard_ceil)
    phi = p.phi_p0 / J_safe
    return np.clip(phi, p.phi_floor, p.phi_ceiling)


def harmonic_mean(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30) -> np.ndarray:
    return 2.0 * a * b / np.maximum(a + b, eps)


def local_chem_pot(J: np.ndarray, theta: np.ndarray, p: Params) -> np.ndarray:
    """
    无量纲局域化学势 m_local = m_mix + m_el

    mixing:
        m_mix = ln(1-phi) + phi + chi(phi,theta) * phi^2
        chi   = chi_inf + S_chi * theta + chi1 * phi

    elastic:
        sphere: m_el = Omega_e * (J^(-1/3) - 1/J)
        slab  : m_el = Omega_e * (J - 1/J)
    """
    phi = phi_from_J(J, p)
    chi = p.chi_inf + p.S_chi * theta + p.chi1 * phi
    m_mix = np.log(np.maximum(1.0 - phi, 1.0e-15)) + phi + chi * phi**2

    if p.use_sphere_elastic:
        m_el = p.Omega_e * (J**(-1.0 / 3.0) - 1.0 / J)
    else:
        m_el = p.Omega_e * (J - 1.0 / J)
    return m_mix + m_el


def finalize_params(p: Params) -> Params:
    if p.auto_set_m_b:
        J0 = np.array([p.J_init], dtype=float)
        th0 = np.array([p.theta_init], dtype=float)
        p = replace(p, m_b=float(local_chem_pot(J0, th0, p)[0]))
    return p


def thermal_factor(theta: np.ndarray, p: Params) -> np.ndarray:
    """
    默认使用 Arrhenius-like thermal factor.
    也可切换到 Hill gate，用于更温和的点火测试。
    """
    theta = np.clip(theta, -p.theta_clip, p.theta_clip)

    if p.use_hill:
        th = np.maximum(theta, 0.0)
        hill = (th**p.n_hill) / np.maximum(th**p.n_hill + p.theta_c**p.n_hill, 1.0e-15)
        return p.hill_eps + hill

    denom = 1.0 + p.eps_T * np.maximum(theta, -1.0 / p.eps_T * 0.95)
    expo = np.clip(p.Gamma_A * theta / denom, -p.arrh_exp_cap, p.arrh_exp_cap)
    return np.exp(expo)


def reaction_rate(u: np.ndarray, theta: np.ndarray, J: np.ndarray, p: Params) -> np.ndarray:
    """
    R = u^n * act(J) * thermal(theta) * cat_factor

    act(J)       = (1-phi)^m_act
    cat_factor   = phi  (可选，表示催化剂密度 ~ 聚合物浓度)
    """
    phi = phi_from_J(J, p)
    accessibility = np.maximum(1.0 - phi, 1.0e-12)**p.m_act
    cat_factor = phi if p.use_cat_density else 1.0
    return np.maximum(u, p.u_floor)**p.n_react * accessibility * cat_factor * thermal_factor(theta, p)


# ============================================================================
# Spherical differential operators (finite-volume / flux form)
# ============================================================================

def spherical_laplacian_neumann(a: np.ndarray, grid: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Delta_s a = (1/s^2) d_s (s^2 d_s a)
    离散时用有限体积形式：div(A * grad)/V
    Neumann at center and outer boundary: d_s a = 0
    """
    ds = grid["ds"]
    A = grid["A"]
    V = grid["V"]
    n = len(a)

    grad = np.zeros(n + 1)
    grad[1:n] = (a[1:] - a[:-1]) / ds
    grad[0] = 0.0
    grad[n] = 0.0

    flux = A * grad
    lap = (flux[1:] - flux[:-1]) / V
    return lap


def face_gradient(a: np.ndarray, ds: float) -> np.ndarray:
    n = len(a)
    g = np.zeros(n + 1)
    g[1:n] = (a[1:] - a[:-1]) / ds
    return g


def divergence_from_face_flux(F: np.ndarray, grid: Dict[str, np.ndarray]) -> np.ndarray:
    """div_s(F) = (1/s^2)d_s(s^2 F) -> (A_{i+1/2}F_{i+1/2} - A_{i-1/2}F_{i-1/2}) / V_i"""
    A = grid["A"]
    V = grid["V"]
    return (A[1:] * F[1:] - A[:-1] * F[:-1]) / V


# ============================================================================
# Semi-discrete RHS (state = [logJ, W=J*u, theta])
# ============================================================================

_LOG_J_MAX = np.log(6.0)


def rhs_sphere_1d(t: float, y: np.ndarray, p: Params, grid: Dict[str, np.ndarray]) -> np.ndarray:
    n = p.N
    ds = grid["ds"]
    log_J_min = np.log(max(p.J_min, p.phi_p0 * 1.02))

    logJ = np.clip(y[:n], log_J_min, _LOG_J_MAX)
    W = y[n:2 * n]
    theta = np.clip(y[2 * n:], -p.theta_clip, p.theta_clip)

    J = np.exp(logJ)
    u = np.maximum(W / J, p.u_floor)
    phi = phi_from_J(J, p)

    # ---- chemical potential ----
    m_local = local_chem_pot(J, theta, p)
    lapJ = spherical_laplacian_neumann(J, grid)
    mu = m_local - p.ell**2 * lapJ

    # ---- solvent flux q ----
    q = np.zeros(n + 1)
    M_cell = p.M0 * np.maximum(1.0 - phi, 1.0e-12)**p.m_mob
    M_face = harmonic_mean(M_cell[:-1], M_cell[1:])
    dmu = face_gradient(mu, ds)
    q[1:n] = -M_face * dmu[1:n]
    q[0] = 0.0
    q[n] = p.Bi_mu * (mu[-1] - p.m_b)        # outward positive

    div_q = divergence_from_face_flux(q, grid)
    logJ_t = -div_q / J

    # ---- reaction rate ----
    R = reaction_rate(u, theta, J, p)

    # ---- substrate flux N_u ----
    nflux = np.zeros(n + 1)
    D_cell = p.D0 * np.maximum(1.0 - phi, 1.0e-12)**p.m_diff
    D_face = harmonic_mean(D_cell[:-1], D_cell[1:])

    q_int = q[1:n]
    u_up = np.where(q_int >= 0.0, u[:-1], u[1:])
    du = face_gradient(u, ds)
    nflux[1:n] = q_int * u_up - p.delta * D_face * du[1:n]
    nflux[0] = 0.0
    nflux[n] = p.Bi_c * (u[-1] - 1.0)        # outward positive

    div_n = divergence_from_face_flux(nflux, grid)
    W_t = -div_n - p.Da * J * R

    # ---- heat flux h ----
    h = np.zeros(n + 1)
    dth = face_gradient(theta, ds)
    h[1:n] = -p.alpha * p.K0 * dth[1:n]
    h[0] = 0.0
    h[n] = p.Bi_T * theta[-1]                # outward positive

    div_h = divergence_from_face_flux(h, grid)
    theta_t = (-div_h + p.Da * J * R) / p.C0

    return np.concatenate([logJ_t, W_t, theta_t])


# ============================================================================
# Jacobian sparsity / finite-difference Jacobian
# ============================================================================

def make_jac_sparsity(N: int) -> csc_matrix:
    """
    3N x 3N Jacobian sparsity pattern (approximate).

    state = [logJ_0.., W_0.., theta_0..]

    logJ eq depends on:
        logJ neighbors up to ±2 (through mu and ell^2 Delta J)
        theta local / ±1     (through chi(theta))
    W eq depends on:
        logJ neighbors up to ±2
        W neighbors up to ±1
        theta local / ±1
    theta eq depends on:
        logJ local / ±1
        W local
        theta neighbors up to ±1
    """
    size = 3 * N
    S = lil_matrix((size, size), dtype=np.float64)

    # block-row / block-col bandwidths
    bw = {
        (0, 0): 2, (0, 1): 1, (0, 2): 1,
        (1, 0): 2, (1, 1): 1, (1, 2): 1,
        (2, 0): 1, (2, 1): 1, (2, 2): 1,
    }

    for (kr, kc), w in bw.items():
        for i in range(N):
            row = kr * N + i
            for dj in range(-w, w + 1):
                j = i + dj
                if 0 <= j < N:
                    col = kc * N + j
                    S[row, col] = 1.0

    return csc_matrix(S)


def make_sparse_fd_jac(rhs_fn, active_cols: List[int]):
    def jac(t, y):
        f0 = rhs_fn(t, y)
        n = len(y)
        Jmat = np.zeros((n, n))
        for i in active_cols:
            hi = max(1.0e-8, 1.0e-6 * abs(y[i]))
            yp = y.copy()
            yp[i] += hi
            Jmat[:, i] = (rhs_fn(t, yp) - f0) / hi
        return Jmat
    return jac


# ============================================================================
# Initialization / simulation
# ============================================================================

def initial_state(p: Params, grid: Dict[str, np.ndarray]) -> np.ndarray:
    s = grid["s_cell"]
    log_J_min = np.log(max(p.J_min, p.phi_p0 * 1.02))

    J0 = np.maximum(p.J_init + p.eps_J * np.cos(np.pi * s), np.exp(log_J_min) + 1.0e-6)
    u0 = np.maximum(p.u_init + p.eps_u * np.cos(np.pi * s), p.u_floor)
    th0 = p.theta_init + p.eps_theta * s

    logJ0 = np.log(J0)
    W0 = J0 * u0
    return np.concatenate([logJ0, W0, th0])


def simulate(p: Params) -> Dict:
    p = finalize_params(p)
    grid = make_grid(p.N)
    y0 = initial_state(p, grid)

    rhs = lambda t, y: rhs_sphere_1d(t, y, p, grid)
    sparsity = make_jac_sparsity(p.N)
    active_cols = np.unique(sparsity.tocoo().col).tolist()
    jac = make_sparse_fd_jac(rhs, active_cols)

    sol = solve_ivp(
        fun=rhs,
        jac=jac,
        t_span=(0.0, p.t_end),
        y0=y0,
        t_eval=np.linspace(0.0, p.t_end, p.n_save),
        method=p.method,
        rtol=p.rtol,
        atol=p.atol,
        max_step=p.max_step,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    n = p.N
    log_J_min = np.log(max(p.J_min, p.phi_p0 * 1.02))
    logJ = np.clip(sol.y[:n], log_J_min, _LOG_J_MAX)
    J = np.exp(logJ)
    W = sol.y[n:2 * n]
    theta = np.clip(sol.y[2 * n:], -p.theta_clip, p.theta_clip)
    u = np.maximum(W / J, p.u_floor)
    phi = phi_from_J(J, p)

    return {
        "success": True,
        "params": p,
        "grid": grid,
        "t": sol.t,
        "J": J,
        "u": u,
        "theta": theta,
        "phi": phi,
        "nfev": sol.nfev,
    }


# ============================================================================
# Diagnostics
# ============================================================================

def volume_average(field_xt: np.ndarray, grid: Dict[str, np.ndarray]) -> np.ndarray:
    V = grid["V"][:, None]
    return np.sum(field_xt * V, axis=0) / np.sum(V)


def volume_std(field_xt: np.ndarray, grid: Dict[str, np.ndarray]) -> np.ndarray:
    mean = volume_average(field_xt, grid)[None, :]
    V = grid["V"][:, None]
    var = np.sum(V * (field_xt - mean)**2, axis=0) / np.sum(V)
    return np.sqrt(var)


def tail_slice(n_t: int, frac0: float = 0.60) -> slice:
    return slice(max(0, int(frac0 * n_t)), n_t)


def detrend(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = np.polyfit(t, y, 1)
    return y - np.polyval(c, t)


def oscillation_metrics(t: np.ndarray, y: np.ndarray, frac0: float = 0.60, amp_floor: float = 1.0e-3) -> Dict:
    sl = tail_slice(len(t), frac0)
    tt, yy = t[sl], y[sl]
    if len(tt) < 20:
        return {"amp": 0.0, "period": np.nan, "n_peaks": 0, "oscillatory": False}

    yd = detrend(tt, yy)
    amp = float(np.max(yd) - np.min(yd))
    prom = max(0.15 * amp, amp_floor)
    mdist = max(3, len(yd) // 20)
    peaks, _ = find_peaks(yd, prominence=prom, distance=mdist)
    troughs, _ = find_peaks(-yd, prominence=prom, distance=mdist)

    period = np.nan
    peak_cv = np.nan
    if len(peaks) >= 2:
        dT = np.diff(tt[peaks])
        period = float(np.mean(dT))
        peak_cv = float(np.std(dT) / np.mean(dT)) if len(dT) >= 2 else 0.0

    osc = amp > amp_floor and len(peaks) >= 2 and len(troughs) >= 2 and (np.isnan(peak_cv) or peak_cv < 0.40)
    return {"amp": amp, "period": period, "n_peaks": int(len(peaks)), "oscillatory": bool(osc)}


def classify_run(data: Dict) -> Dict:
    grid = data["grid"]
    t = data["t"]
    J = data["J"]
    u = data["u"]
    theta = data["theta"]

    J_mean = volume_average(J, grid)
    u_mean = volume_average(u, grid)
    theta_mean = volume_average(theta, grid)

    J_std = volume_std(J, grid)
    theta_std = volume_std(theta, grid)

    th_m = oscillation_metrics(t, theta_mean, amp_floor=2.0e-3)
    J_m = oscillation_metrics(t, J_mean, amp_floor=2.0e-4)

    osc = bool(th_m["oscillatory"] or J_m["oscillatory"])
    sl = tail_slice(len(t))
    non_uni = max(np.mean(J_std[sl]), np.mean(theta_std[sl])) > 2.0e-3

    theta_f = float(theta_mean[-1])
    thermal_state = "hot" if theta_f > 0.25 else ("cold" if theta_f < 0.05 else "warm")

    if osc and non_uni:
        label = "oscillatory_nonuniform"
    elif osc:
        label = "oscillatory_uniform"
    elif non_uni:
        label = f"steady_{thermal_state}_nonuniform"
    else:
        label = f"steady_{thermal_state}_uniform"

    return {
        "label": label,
        "is_oscillatory": int(osc),
        "is_nonuniform": int(non_uni),
        "thermal_state": thermal_state,
        "theta_amp": float(th_m["amp"]),
        "J_amp": float(J_m["amp"]),
        "theta_period": float(th_m["period"]) if np.isfinite(th_m["period"]) else np.nan,
        "J_period": float(J_m["period"]) if np.isfinite(J_m["period"]) else np.nan,
        "theta_peaks": int(th_m["n_peaks"]),
        "J_peaks": int(J_m["n_peaks"]),
        "J_mean_final": float(J_mean[-1]),
        "u_mean_final": float(u_mean[-1]),
        "theta_mean_final": float(theta_mean[-1]),
        "nfev": int(data.get("nfev", -1)),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_timeseries(data: Dict, outdir: Path, suffix: str = "") -> None:
    grid = data["grid"]
    t = data["t"]
    J_mean = volume_average(data["J"], grid)
    u_mean = volume_average(data["u"], grid)
    theta_mean = volume_average(data["theta"], grid)
    phi_mean = volume_average(data["phi"], grid)

    info = classify_run(data)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t, J_mean, "b-", lw=1.2)
    axes[0].set_ylabel(r"$\langle J \rangle$")
    axes[0].set_title(
        f"sphere_1d | {info['label']} | Da={data['params'].Da:.3g}, Bi_T={data['params'].Bi_T:.3g}, "
        f"Bi_c={data['params'].Bi_c:.3g}, S_chi={data['params'].S_chi:.3g}"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, phi_mean, color="purple", lw=1.2)
    axes[1].set_ylabel(r"$\langle \phi \rangle$")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, u_mean, color="orange", lw=1.2)
    axes[2].set_ylabel(r"$\langle u \rangle$")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, theta_mean, color="red", lw=1.2)
    axes[3].set_ylabel(r"$\langle \theta \rangle$")
    axes[3].set_xlabel("t")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"timeseries{suffix}.png", dpi=160)
    plt.close(fig)


def plot_spacetime(data: Dict, outdir: Path, suffix: str = "") -> None:
    s = data["grid"]["s_cell"]
    t = data["t"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fields = [(data["J"], "J(s,t)", "viridis"),
              (data["u"], "u(s,t)", "plasma"),
              (data["theta"], r"$\theta$(s,t)", "coolwarm")]

    for ax, (F, title, cmap) in zip(axes, fields):
        im = ax.imshow(F, origin="lower", aspect="auto",
                       extent=[t[0], t[-1], s[0], s[-1]], cmap=cmap)
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.savefig(outdir / f"spacetime{suffix}.png", dpi=160)
    plt.close(fig)


def plot_snapshots(data: Dict, outdir: Path, suffix: str = "") -> None:
    s = data["grid"]["s_cell"]
    t = data["t"]

    idxs = np.linspace(0, len(t) - 1, 6, dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    for k in idxs:
        axes[0].plot(s, data["J"][:, k], lw=1.0, label=f"t={t[k]:.0f}")
        axes[1].plot(s, data["u"][:, k], lw=1.0)
        axes[2].plot(s, data["theta"][:, k], lw=1.0)

    axes[0].set_title("J(s) snapshots")
    axes[1].set_title("u(s) snapshots")
    axes[2].set_title(r"$\theta$(s) snapshots")
    for ax in axes:
        ax.set_xlabel("s")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    fig.savefig(outdir / f"snapshots{suffix}.png", dpi=160)
    plt.close(fig)


def plot_phase_means(data: Dict, outdir: Path, suffix: str = "") -> None:
    grid = data["grid"]
    Jm = volume_average(data["J"], grid)
    um = volume_average(data["u"], grid)
    thm = volume_average(data["theta"], grid)

    i0 = int(0.5 * len(Jm))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    axes[0].plot(Jm[i0:], thm[i0:], "b-", lw=1.0)
    axes[0].set_xlabel(r"$\langle J \rangle$")
    axes[0].set_ylabel(r"$\langle \theta \rangle$")
    axes[0].set_title(r"$\langle J\rangle$-$\langle\theta\rangle$")

    axes[1].plot(um[i0:], thm[i0:], "r-", lw=1.0)
    axes[1].set_xlabel(r"$\langle u \rangle$")
    axes[1].set_ylabel(r"$\langle \theta \rangle$")
    axes[1].set_title(r"$\langle u\rangle$-$\langle\theta\rangle$")

    axes[2].plot(Jm[i0:], um[i0:], "g-", lw=1.0)
    axes[2].set_xlabel(r"$\langle J \rangle$")
    axes[2].set_ylabel(r"$\langle u \rangle$")
    axes[2].set_title(r"$\langle J\rangle$-$\langle u\rangle$")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.savefig(outdir / f"phase_means{suffix}.png", dpi=160)
    plt.close(fig)


def save_summary(data: Dict, outdir: Path, suffix: str = "") -> None:
    info = classify_run(data)
    path = outdir / f"summary{suffix}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("sphere_1d summary\n")
        f.write("=" * 60 + "\n")
        for k, v in data["params"].__dict__.items():
            f.write(f"{k:20s} = {v}\n")
        f.write("\nDiagnostics\n")
        f.write("-" * 60 + "\n")
        for k, v in info.items():
            f.write(f"{k:20s} = {v}\n")


# ============================================================================
# Da scan
# ============================================================================

def scan_Da(base: Params, Da_vals: np.ndarray, outdir: Path) -> List[Dict]:
    rows = []
    outdir.mkdir(parents=True, exist_ok=True)

    for i, Da in enumerate(Da_vals, 1):
        p = replace(base, Da=float(Da))
        try:
            data = simulate(p)
            info = classify_run(data)
            row = {"Da": float(Da), **info, "status": "ok"}
        except Exception as e:
            row = {
                "Da": float(Da),
                "label": "solve_failed",
                "is_oscillatory": 0,
                "is_nonuniform": 0,
                "thermal_state": "failed",
                "theta_amp": np.nan,
                "J_amp": np.nan,
                "theta_period": np.nan,
                "J_period": np.nan,
                "theta_peaks": 0,
                "J_peaks": 0,
                "J_mean_final": np.nan,
                "u_mean_final": np.nan,
                "theta_mean_final": np.nan,
                "nfev": -1,
                "status": f"failed: {e}",
            }
        rows.append(row)
        print(f"[{i:>3d}/{len(Da_vals)}] Da={Da:.4g} -> {row['label']}")

    csv_path = outdir / "scan_Da.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot_scan_Da(rows, outdir)
    return rows


def plot_scan_Da(rows: List[Dict], outdir: Path) -> None:
    Da = np.array([r["Da"] for r in rows], dtype=float)
    osc = np.array([r["is_oscillatory"] for r in rows], dtype=float)
    nonuni = np.array([r["is_nonuniform"] for r in rows], dtype=float)
    amp = np.array([r["theta_amp"] for r in rows], dtype=float)
    period = np.array([r["theta_period"] for r in rows], dtype=float)
    Jf = np.array([r["J_mean_final"] for r in rows], dtype=float)
    thf = np.array([r["theta_mean_final"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    axes[0, 0].plot(Da, Jf, "b-o", ms=3)
    axes[0, 0].set_xlabel("Da")
    axes[0, 0].set_ylabel(r"final $\langle J \rangle$")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(Da, thf, "r-o", ms=3)
    axes[0, 1].set_xlabel("Da")
    axes[0, 1].set_ylabel(r"final $\langle \theta \rangle$")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(Da, amp, "m-o", ms=3, label=r"$\Delta\langle\theta\rangle$")
    axes[1, 0].plot(Da, osc, "k--", lw=1.0, label="oscillatory")
    axes[1, 0].plot(Da, nonuni, "g--", lw=1.0, label="nonuniform")
    axes[1, 0].set_xlabel("Da")
    axes[1, 0].set_ylabel("amp / indicator")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(Da, period, "c-o", ms=3)
    axes[1, 1].set_xlabel("Da")
    axes[1, 1].set_ylabel("period")
    axes[1, 1].grid(True, alpha=0.3)

    fig.savefig(outdir / "scan_Da.png", dpi=160)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="1D spherical LCST gel-bead model (finite-volume / BDF)")

    ap.add_argument("--mode", choices=["single", "scan-da"], default="single")
    ap.add_argument("--outdir", default="sphere_1d_results")

    # numerics
    ap.add_argument("--N", type=int, default=121)
    ap.add_argument("--t-end", type=float, default=500.0)
    ap.add_argument("--n-save", type=int, default=5000)
    ap.add_argument("--max-step", type=float, default=0.5)

    # working point / constitutive
    ap.add_argument("--phi-p0", type=float, default=0.15)
    ap.add_argument("--chi-inf", type=float, default=0.60)
    ap.add_argument("--S-chi", type=float, default=1.00)
    ap.add_argument("--chi1", type=float, default=1.10)
    ap.add_argument("--Omega-e", type=float, default=0.12)
    ap.add_argument("--ell", type=float, default=0.01)
    ap.add_argument("--sphere-elastic", action="store_true")
    ap.add_argument("--cat-density", action="store_true")

    # transport / reaction
    ap.add_argument("--Da", type=float, default=4.0)
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--Gamma-A", type=float, default=1.5)
    ap.add_argument("--eps-T", type=float, default=0.03)
    ap.add_argument("--n-react", type=int, default=1)
    ap.add_argument("--m-act", type=float, default=6.0)
    ap.add_argument("--m-diff", type=float, default=2.0)
    ap.add_argument("--m-mob", type=float, default=1.0)
    ap.add_argument("--M0", type=float, default=1.0)
    ap.add_argument("--D0", type=float, default=2.0)
    ap.add_argument("--C0", type=float, default=1.0)
    ap.add_argument("--K0", type=float, default=1.0)

    # boundaries
    ap.add_argument("--Bi-mu", type=float, default=1.00)
    ap.add_argument("--Bi-c", type=float, default=0.70)
    ap.add_argument("--Bi-T", type=float, default=0.10)
    ap.add_argument("--m-b", type=float, default=0.0)
    ap.add_argument("--no-auto-mb", action="store_true")

    # initial condition
    ap.add_argument("--J-init", type=float, default=1.30)
    ap.add_argument("--u-init", type=float, default=0.02)
    ap.add_argument("--theta-init", type=float, default=0.0)

    # optional Hill trigger
    ap.add_argument("--use-hill", action="store_true")
    ap.add_argument("--theta-c", type=float, default=0.50)
    ap.add_argument("--n-hill", type=float, default=4.0)
    ap.add_argument("--hill-eps", type=float, default=0.05)

    # Da scan
    ap.add_argument("--Da-min", type=float, default=2.0)
    ap.add_argument("--Da-max", type=float, default=8.0)
    ap.add_argument("--n-Da", type=int, default=13)

    return ap


def params_from_args(args) -> Params:
    return Params(
        N=args.N,
        t_end=args.t_end,
        n_save=args.n_save,
        max_step=args.max_step,

        phi_p0=args.phi_p0,
        chi_inf=args.chi_inf,
        S_chi=args.S_chi,
        chi1=args.chi1,
        Omega_e=args.Omega_e,
        ell=args.ell,
        use_sphere_elastic=args.sphere_elastic,
        use_cat_density=args.cat_density,

        Da=args.Da,
        delta=args.delta,
        alpha=args.alpha,
        Gamma_A=args.Gamma_A,
        eps_T=args.eps_T,
        n_react=args.n_react,
        m_act=args.m_act,
        m_diff=args.m_diff,
        m_mob=args.m_mob,
        M0=args.M0,
        D0=args.D0,
        C0=args.C0,
        K0=args.K0,

        Bi_mu=args.Bi_mu,
        Bi_c=args.Bi_c,
        Bi_T=args.Bi_T,
        m_b=args.m_b,
        auto_set_m_b=(not args.no_auto_mb),

        J_init=args.J_init,
        u_init=args.u_init,
        theta_init=args.theta_init,

        use_hill=args.use_hill,
        theta_c=args.theta_c,
        n_hill=args.n_hill,
        hill_eps=args.hill_eps,
    )


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = finalize_params(params_from_args(args))

    print("=" * 72)
    print("sphere_1d.py — spherical 1D LCST gel-bead model")
    print("=" * 72)
    print(f"mode              = {args.mode}")
    print(f"N                 = {base.N}")
    print(f"t_end             = {base.t_end}")
    print(f"Da                = {base.Da}")
    print(f"Bi_mu, Bi_c, Bi_T = {base.Bi_mu}, {base.Bi_c}, {base.Bi_T}")
    print(f"S_chi, Gamma_A    = {base.S_chi}, {base.Gamma_A}")
    print(f"sphere_elastic    = {base.use_sphere_elastic}")
    print(f"cat_density       = {base.use_cat_density}")
    print(f"auto m_b          = {base.auto_set_m_b}  ->  m_b = {base.m_b:.6g}")
    print(f"outdir            = {outdir.resolve()}")
    print()

    if args.mode == "single":
        data = simulate(base)
        info = classify_run(data)
        print("Diagnostics:")
        for k, v in info.items():
            print(f"  {k:18s} = {v}")

        plot_timeseries(data, outdir)
        plot_spacetime(data, outdir)
        plot_snapshots(data, outdir)
        plot_phase_means(data, outdir)
        save_summary(data, outdir)
        print("\nSaved:")
        print(f"  {outdir / 'timeseries.png'}")
        print(f"  {outdir / 'spacetime.png'}")
        print(f"  {outdir / 'snapshots.png'}")
        print(f"  {outdir / 'phase_means.png'}")
        print(f"  {outdir / 'summary.txt'}")
        return

    if args.mode == "scan-da":
        Da_vals = np.linspace(args.Da_min, args.Da_max, args.n_Da)
        rows = scan_Da(base, Da_vals, outdir)
        n_ok = sum(r["status"] == "ok" for r in rows)
        n_osc = sum(int(r["is_oscillatory"]) for r in rows if r["status"] == "ok")
        print(f"\nScan finished: ok = {n_ok}/{len(rows)}, oscillatory = {n_osc}/{max(n_ok,1)}")
        print(f"Saved: {outdir / 'scan_Da.csv'}")
        print(f"       {outdir / 'scan_Da.png'}")


if __name__ == "__main__":
    main()
