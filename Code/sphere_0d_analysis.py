#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sphere_0d_analysis.py — 催化剂嫁接型 LCST 凝胶球 0D 集总模型分析

模型：3-ODE 系统 (J, u, θ)
  dJ/dt   = J^{2/3} [m_b - m(J,θ)]
  d(Ju)/dt = Bi_u J^{2/3}(1-u) - Da J R̂(u,θ,J)
  C(J) dθ/dt = Da R̂(u,θ,J) - Bi_T J^{2/3} θ

包含：
  1. 稳态求解
  2. Jacobian 特征值分析
  3. Hopf 分岔扫描
  4. 时间积分演示
  5. 参数相图
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SphereParams:
    # 工作点
    phi_p0: float = 0.15        # 参考态聚合物体积分数
    chi_inf: float = 0.55       # 基准 Flory-Huggins 参数 (T_∞ 下)
    S_chi: float = 1.2          # 温度敏感度: dχ/dθ
    chi1: float = 0.80          # 浓度依赖: χ = χ_∞ + S_χ θ + χ₁ φ
    Omega_e: float = 0.10       # 弹性模量 (无量纲 Gv_s/R_gT_∞)

    # 反应/输运
    Da: float = 8.0             # Damköhler 数
    Gamma_A: float = 2.0        # Arrhenius 热敏指数
    eps_T: float = 0.03         # 温度非线性修正
    n_react: int = 1            # 反应级数
    m_act: float = 4.0          # 可达性指数 a(J) = (1-φ)^m_act
    use_cat_density: bool = True  # 是否包含催化剂密度因子 φ/J

    # 边界交换
    Bi_u: float = 1.0           # 底物交换 Biot 数
    Bi_T: float = 0.15          # 热交换 Biot 数
    Bi_mu: float = 2.0          # 溶胀交换 Biot 数 (吸收进 dJ/dt 系数)

    # 其它
    C0: float = 1.0             # 参考热容
    m_b: float = None           # 外浴化学势 (auto)
    J_init: float = 1.5         # 初始溶胀比

    # 数值
    arrh_cap: float = 50.0


# ═══════════════════════════════════════════════════════════════════
# 本构关系
# ═══════════════════════════════════════════════════════════════════

def phi_of_J(J, p):
    """φ = φ_{p0}/J"""
    return np.clip(p.phi_p0 / np.maximum(J, p.phi_p0 * 1.01), 0, 0.999)

def chem_pot(J, theta, p):
    """
    无量纲化学势 m(J,θ) = m_mix + m_el
    球坐标各向同性溶胀弹性: m_el = Ω_e (J^{-1/3} - J^{-1})
    """
    phi = phi_of_J(J, p)
    chi = p.chi_inf + p.S_chi * theta + p.chi1 * phi
    m_mix = np.log(np.maximum(1 - phi, 1e-15)) + phi + chi * phi**2
    # 球各向同性弹性 (neo-Hookean)
    m_el = p.Omega_e * (J**(-1.0/3.0) - 1.0/J)
    return m_mix + m_el

def reaction_rate(u, theta, J, p):
    """
    R̂(u,θ,J) = u^n · f(J) · exp(Γ_A θ / (1+ε_T θ))
    f(J) = [φ_{p0}/J]^{cat} · (1-φ)^{m_act}
    """
    phi = phi_of_J(J, p)
    # 热因子
    denom = 1.0 + p.eps_T * np.maximum(theta, -0.95/p.eps_T)
    exp_arg = np.clip(p.Gamma_A * theta / denom, -p.arrh_cap, p.arrh_cap)
    thermal = np.exp(exp_arg)
    # 可达性 × 催化剂密度
    accessibility = np.maximum(1.0 - phi, 1e-15)**p.m_act
    if p.use_cat_density:
        cat_factor = phi  # ∝ φ = φ_{p0}/J (催化剂密度与聚合物成正比)
    else:
        cat_factor = 1.0
    return np.maximum(u, 1e-15)**p.n_react * cat_factor * accessibility * thermal

def C_eff(J, p):
    """有效热容 (可取为 J 的函数，这里简化)"""
    return p.C0  # 可改为 p.C0 * J 等

def auto_m_b(p):
    """自动设定外浴化学势 = 初始平衡态化学势"""
    return float(chem_pot(p.J_init, 0.0, p))


# ═══════════════════════════════════════════════════════════════════
# 0D ODE 系统
# ═══════════════════════════════════════════════════════════════════

def rhs_0d(t, y, p):
    """
    y = [J, u, θ]
    dJ/dt   = Bi_μ J^{2/3} [m_b - m(J,θ)]
    d(Ju)/dt = Bi_u J^{2/3}(1-u) - Da J R̂
      → J du/dt = -u dJ/dt + Bi_u J^{2/3}(1-u) - Da J R̂
    C dθ/dt = Da R̂ - Bi_T J^{2/3} θ
    """
    J = np.clip(y[0], p.phi_p0 * 1.05, 15.0)
    u = np.clip(y[1], 1e-15, 50.0)
    theta = np.clip(y[2], -10.0, 30.0)

    m = chem_pot(J, theta, p)
    R = reaction_rate(u, theta, J, p)
    J23 = J**(2.0/3.0)

    dJdt = p.Bi_mu * J23 * (p.m_b - m)
    dJu_dt = p.Bi_u * J23 * (1.0 - u) - p.Da * J * R
    dudt = (dJu_dt - u * dJdt) / J
    dthetadt = (p.Da * R - p.Bi_T * J23 * theta) / C_eff(J, p)

    return [dJdt, dudt, dthetadt]


def rhs_vec(y, p):
    """用于 fsolve 的向量版"""
    return rhs_0d(0, y, p)


# ═══════════════════════════════════════════════════════════════════
# 稳态求解
# ═══════════════════════════════════════════════════════════════════

def find_steady_state(p, J_guess=None, u_guess=None, theta_guess=None):
    """用 fsolve 找稳态"""
    if J_guess is None:
        J_guess = p.J_init
    if u_guess is None:
        u_guess = 0.5
    if theta_guess is None:
        theta_guess = 0.1

    def F(y):
        return rhs_0d(0, y, p)

    sol = fsolve(F, [J_guess, u_guess, theta_guess], full_output=True)
    x, info, ier, msg = sol
    if ier == 1 and x[0] > p.phi_p0 * 1.05 and x[1] > 0:
        return x
    return None


def find_multiple_steady_states(p, n_tries=50):
    """多初值搜索所有稳态"""
    found = []
    for J0 in np.linspace(p.phi_p0 * 1.1, 5.0, n_tries):
        for u0 in [0.01, 0.3, 0.7, 0.99]:
            for th0 in [0.0, 0.2, 0.5, 1.0, 2.0]:
                ss = find_steady_state(p, J0, u0, th0)
                if ss is not None:
                    is_new = True
                    for s in found:
                        if np.linalg.norm(ss - s) < 1e-4:
                            is_new = False
                            break
                    if is_new:
                        found.append(ss)
    return found


# ═══════════════════════════════════════════════════════════════════
# Jacobian 与线性稳定性
# ═══════════════════════════════════════════════════════════════════

def numerical_jacobian(p, ss, eps=1e-7):
    """数值 Jacobian"""
    n = len(ss)
    J = np.zeros((n, n))
    f0 = np.array(rhs_0d(0, ss, p))
    for i in range(n):
        yp = ss.copy()
        yp[i] += eps
        fp = np.array(rhs_0d(0, yp, p))
        J[:, i] = (fp - f0) / eps
    return J


def stability_analysis(p, ss):
    """返回特征值和稳定性信息"""
    Jac = numerical_jacobian(p, ss)
    eigvals = np.linalg.eigvals(Jac)
    max_real = np.max(np.real(eigvals))
    has_imag = np.any(np.abs(np.imag(eigvals)) > 1e-8)
    stable = max_real < 0
    return {
        "eigvals": eigvals,
        "jacobian": Jac,
        "max_real": max_real,
        "stable": stable,
        "oscillatory_unstable": (not stable) and has_imag,
        "has_imaginary": has_imag,
    }


# ═══════════════════════════════════════════════════════════════════
# 时间积分
# ═══════════════════════════════════════════════════════════════════

def integrate_0d(p, y0=None, t_end=500.0, n_pts=5000):
    """积分 0D 模型，带回退策略"""
    if y0 is None:
        y0 = [p.J_init, 0.05, 0.01]
    if p.m_b is None:
        p.m_b = auto_m_b(p)

    for method, rtol, atol, ms in [
        ("BDF", 1e-8, 1e-10, 0.5),
        ("BDF", 1e-6, 1e-8, 0.2),
        ("Radau", 1e-6, 1e-8, 0.2),
        ("LSODA", 1e-6, 1e-8, 0.5),
    ]:
        try:
            sol = solve_ivp(
                lambda t, y: rhs_0d(t, y, p),
                (0, t_end), y0,
                method=method, rtol=rtol, atol=atol,
                t_eval=np.linspace(0, t_end, n_pts),
                max_step=ms,
            )
            if sol.success:
                return sol
        except:
            continue
    # 返回失败的 sol 对象
    from types import SimpleNamespace
    return SimpleNamespace(success=False, message="All methods failed",
                          t=np.array([0]), y=np.array([[y0[0]], [y0[1]], [y0[2]]]))


# ═══════════════════════════════════════════════════════════════════
# 振荡检测
# ═══════════════════════════════════════════════════════════════════

def detect_oscillation(t, y, frac=0.5):
    """检测后半段信号的振荡"""
    i0 = int(len(t) * frac)
    tt, yy = t[i0:], y[i0:]
    if len(tt) < 30:
        return {"oscillatory": False, "amp": 0, "period": np.nan}
    yy_d = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
    amp = float(np.max(yy_d) - np.min(yy_d))
    prom = max(0.1 * amp, 1e-4)
    peaks, _ = find_peaks(yy_d, prominence=prom, distance=max(3, len(yy_d)//30))
    period = np.nan
    if len(peaks) >= 2:
        dT = np.diff(tt[peaks])
        period = float(np.mean(dT))
    osc = amp > 1e-3 and len(peaks) >= 2
    return {"oscillatory": osc, "amp": amp, "period": period, "n_peaks": len(peaks)}


# ═══════════════════════════════════════════════════════════════════
# 分析 1: 反应因子 f(J) 的行为
# ═══════════════════════════════════════════════════════════════════

def plot_reaction_factor(p, outdir="."):
    """画出 f(J) = cat_density × accessibility 随 J 的变化"""
    J_arr = np.linspace(p.phi_p0 * 1.05, 5.0, 500)
    phi_arr = p.phi_p0 / J_arr
    cat_density = phi_arr
    accessibility = (1.0 - phi_arr)**p.m_act
    f_J = cat_density * accessibility

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(J_arr, cat_density, 'b-', lw=2, label=r'$\phi_{p0}/J$ (cat. density)')
    ax.plot(J_arr, accessibility, 'r-', lw=2, label=r'$(1-\phi)^{m_{act}}$ (access.)')
    ax.set_xlabel('J'); ax.set_ylabel('Factor')
    ax.set_title('催化剂密度 vs 可达性')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(J_arr, f_J, 'k-', lw=2.5)
    J_peak = J_arr[np.argmax(f_J)]
    ax.axvline(J_peak, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('J'); ax.set_ylabel('f(J)')
    ax.set_title(f'净反应因子 f(J)，峰值在 J={J_peak:.2f}')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # 化学势 vs J at different θ
    for th, c in [(0, 'blue'), (0.5, 'orange'), (1.0, 'red'), (2.0, 'darkred')]:
        m_arr = np.array([chem_pot(J, th, p) for J in J_arr])
        ax.plot(J_arr, m_arr, color=c, lw=1.5, label=f'θ={th}')
    if p.m_b is not None:
        ax.axhline(p.m_b, color='green', ls='--', lw=1, label='$m_b$ (bath)')
    ax.set_xlabel('J'); ax.set_ylabel('m(J,θ)')
    ax.set_title('化学势 vs J (不同温度)')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{outdir}/01_constitutive.png", dpi=150)
    plt.close(fig)
    print("  [1] 本构关系图 → 01_constitutive.png")


# ═══════════════════════════════════════════════════════════════════
# 分析 2: 稳态与线性稳定性
# ═══════════════════════════════════════════════════════════════════

def analyze_steady_states(p, outdir="."):
    """找稳态并分析稳定性"""
    if p.m_b is None:
        p.m_b = auto_m_b(p)

    ss_list = find_multiple_steady_states(p)
    print(f"  [2] 找到 {len(ss_list)} 个稳态:")
    results = []
    for i, ss in enumerate(ss_list):
        info = stability_analysis(p, ss)
        eigvals = info["eigvals"]
        phi_ss = p.phi_p0 / ss[0]
        status = "稳定" if info["stable"] else "不稳定"
        if info["oscillatory_unstable"]:
            status = "不稳定(振荡型)"
        print(f"      SS{i}: J={ss[0]:.4f}, u={ss[1]:.4f}, θ={ss[2]:.4f}, "
              f"φ={phi_ss:.4f} → {status}")
        print(f"           λ = {eigvals}")
        results.append((ss, info))
    return results


# ═══════════════════════════════════════════════════════════════════
# 分析 3: Hopf 分岔 — Da 扫描
# ═══════════════════════════════════════════════════════════════════

def hopf_scan_Da(p_base, Da_range=None, outdir="."):
    """扫描 Da，找 Hopf 分岔"""
    if Da_range is None:
        Da_range = np.linspace(1.0, 30.0, 120)

    max_reals = []
    max_imags = []
    J_ss_arr = []
    theta_ss_arr = []
    u_ss_arr = []

    prev_ss = None  # 跟踪上一个稳态作为下一个初始猜测

    for Da in Da_range:
        p = SphereParams(**{**p_base.__dict__, "Da": Da})
        if p.m_b is None:
            p.m_b = auto_m_b(p)

        # 多初值搜索
        best_ss = None
        best_J = -1
        guesses = [
            (p.J_init, 0.5, 0.1),
            (p.J_init * 0.8, 0.3, 0.05),
            (p.J_init * 1.2, 0.7, 0.2),
            (p.J_init * 0.5, 0.3, 0.5),
        ]
        if prev_ss is not None:
            guesses.insert(0, tuple(prev_ss))
        for g in guesses:
            ss = find_steady_state(p, *g)
            if ss is not None and ss[0] > best_J:
                best_ss = ss
                best_J = ss[0]

        if best_ss is None:
            max_reals.append(np.nan)
            max_imags.append(np.nan)
            J_ss_arr.append(np.nan)
            theta_ss_arr.append(np.nan)
            u_ss_arr.append(np.nan)
            continue

        prev_ss = best_ss
        info = stability_analysis(p, best_ss)
        idx = np.argmax(np.real(info["eigvals"]))
        max_reals.append(np.real(info["eigvals"][idx]))
        max_imags.append(np.abs(np.imag(info["eigvals"][idx])))
        J_ss_arr.append(best_ss[0])
        theta_ss_arr.append(best_ss[2])
        u_ss_arr.append(best_ss[1])

    max_reals = np.array(max_reals)
    max_imags = np.array(max_imags)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(Da_range, max_reals, 'b-', lw=2)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    # 标记过零点
    for i in range(len(max_reals)-1):
        if np.isfinite(max_reals[i]) and np.isfinite(max_reals[i+1]):
            if max_reals[i] * max_reals[i+1] < 0:
                Da_hopf = Da_range[i] + (Da_range[i+1]-Da_range[i]) * \
                          (-max_reals[i]) / (max_reals[i+1] - max_reals[i])
                ax.axvline(Da_hopf, color='red', ls='--', alpha=0.7)
                ax.annotate(f'Hopf ≈ {Da_hopf:.1f}', (Da_hopf, 0),
                           fontsize=9, color='red')
    ax.set_xlabel('Da'); ax.set_ylabel(r'max Re($\lambda$)')
    ax.set_title('最大特征值实部 vs Da')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(Da_range, max_imags, 'r-', lw=2)
    ax.set_xlabel('Da'); ax.set_ylabel(r'|Im($\lambda$)|')
    ax.set_title('对应特征值虚部 (振荡频率)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(Da_range, J_ss_arr, 'g-', lw=2, label='$J_{ss}$')
    ax.set_xlabel('Da'); ax.set_ylabel('$J_{ss}$')
    ax.set_title('稳态溶胀比')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(Da_range, theta_ss_arr, 'r-', lw=2, label=r'$\theta_{ss}$')
    ax2 = ax.twinx()
    ax2.plot(Da_range, u_ss_arr, 'b--', lw=1.5, label='$u_{ss}$')
    ax.set_xlabel('Da')
    ax.set_ylabel(r'$\theta_{ss}$', color='red')
    ax2.set_ylabel('$u_{ss}$', color='blue')
    ax.set_title('稳态温度与底物')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Hopf 分岔扫描 (S_χ={p_base.S_chi}, Bi_T={p_base.Bi_T})',
                fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{outdir}/02_hopf_Da_scan.png", dpi=150)
    plt.close(fig)
    print("  [3] Hopf 分岔(Da) → 02_hopf_Da_scan.png")


# ═══════════════════════════════════════════════════════════════════
# 分析 4: 时间积分演示
# ═══════════════════════════════════════════════════════════════════

def time_integration_demo(p, outdir=".", t_end=600.0, label=""):
    """积分并画时间序列"""
    if p.m_b is None:
        p.m_b = auto_m_b(p)

    # 多组初始条件尝试
    ics = [
        [p.J_init, 0.02, 0.01],
        [p.J_init, 0.5, 0.05],
        [p.J_init * 0.8, 0.1, 0.02],
    ]
    sol = None
    for y0 in ics:
        sol = integrate_0d(p, y0=y0, t_end=t_end, n_pts=8000)
        if sol.success and len(sol.t) > 100:
            break

    if sol is None or not sol.success:
        print(f"  积分失败 ({label}): {getattr(sol, 'message', 'unknown')}")
        return None

    t = sol.t
    J, u, theta = sol.y[0], sol.y[1], sol.y[2]
    phi = p.phi_p0 / np.maximum(J, p.phi_p0 * 1.01)

    # 振荡检测
    osc_info = detect_oscillation(t, theta)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, J, 'b-', lw=1.2)
    ax.set_ylabel('J (溶胀比)')
    ax.set_title(f'Da={p.Da}, S_χ={p.S_chi}, Bi_T={p.Bi_T}, Γ_A={p.Gamma_A} '
                f'| {"振荡" if osc_info["oscillatory"] else "非振荡"}'
                f'{" T="+str(round(osc_info["period"],1)) if osc_info["oscillatory"] else ""}')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, phi, 'g-', lw=1.2)
    ax.set_ylabel('φ (聚合物分数)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, u, 'purple', lw=1.2)
    ax.set_ylabel('u (底物浓度)')
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, theta, 'r-', lw=1.2)
    ax.set_ylabel('θ (温度)')
    ax.set_xlabel('t (无量纲时间)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    fig.savefig(f"{outdir}/03_timeseries{suffix}.png", dpi=150)
    plt.close(fig)
    print(f"  [4] 时间序列{suffix} → 03_timeseries{suffix}.png")

    # 相图
    if osc_info["oscillatory"]:
        i0 = int(len(t) * 0.5)
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

        ax = axes2[0]
        ax.plot(J[i0:], theta[i0:], 'b-', lw=0.6, alpha=0.8)
        ax.set_xlabel('J'); ax.set_ylabel('θ')
        ax.set_title('J-θ 相图 (极限环)')
        ax.grid(True, alpha=0.3)

        ax = axes2[1]
        ax.plot(u[i0:], theta[i0:], 'r-', lw=0.6, alpha=0.8)
        ax.set_xlabel('u'); ax.set_ylabel('θ')
        ax.set_title('u-θ 相图')
        ax.grid(True, alpha=0.3)

        ax = axes2[2]
        ax.plot(J[i0:], u[i0:], 'g-', lw=0.6, alpha=0.8)
        ax.set_xlabel('J'); ax.set_ylabel('u')
        ax.set_title('J-u 相图')
        ax.grid(True, alpha=0.3)

        fig2.suptitle(f'极限环 (后半段)', fontsize=12, fontweight='bold')
        fig2.tight_layout()
        fig2.savefig(f"{outdir}/04_phase_portrait{suffix}.png", dpi=150)
        plt.close(fig2)
        print(f"  [4b] 相图{suffix} → 04_phase_portrait{suffix}.png")

    return sol, osc_info


# ═══════════════════════════════════════════════════════════════════
# 分析 5: 二维参数相图 (Da vs S_chi)
# ═══════════════════════════════════════════════════════════════════

def phase_diagram_2d(p_base, outdir=".",
                     x_param="Da", x_range=None,
                     y_param="S_chi", y_range=None,
                     nx=30, ny=30, t_end=400.0):
    """二维参数扫描，标记振荡/稳定区域"""
    if x_range is None:
        x_range = np.linspace(2, 25, nx)
    if y_range is None:
        y_range = np.linspace(0.3, 2.5, ny)
    nx = len(x_range)
    ny = len(y_range)

    # 方法1: 线性稳定性 (快)
    grid_stable = np.zeros((ny, nx))
    grid_theta = np.full((ny, nx), np.nan)
    grid_max_real = np.full((ny, nx), np.nan)

    print(f"  [5] 线性稳定性相图 ({x_param} vs {y_param}): {nx}×{ny} = {nx*ny} 点 ...")

    for j, yv in enumerate(y_range):
        for i, xv in enumerate(x_range):
            pp = SphereParams(**{**p_base.__dict__, x_param: xv, y_param: yv})
            pp.m_b = auto_m_b(pp)
            ss = find_steady_state(pp, pp.J_init, 0.5, 0.1)
            if ss is None:
                ss = find_steady_state(pp, pp.J_init*0.6, 0.3, 0.5)
            if ss is None:
                grid_stable[j, i] = -1
                continue
            info = stability_analysis(pp, ss)
            grid_max_real[j, i] = info["max_real"]
            grid_theta[j, i] = ss[2]
            if info["stable"]:
                grid_stable[j, i] = 0  # 稳定
            elif info["oscillatory_unstable"]:
                grid_stable[j, i] = 2  # 振荡不稳定 (Hopf)
            else:
                grid_stable[j, i] = 1  # 单调不稳定

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ext = [x_range[0], x_range[-1], y_range[0], y_range[-1]]

    ax = axes[0]
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(grid_stable, origin='lower', aspect='auto', extent=ext,
                   cmap=cmap, vmin=-1, vmax=2)
    ax.set_xlabel(x_param); ax.set_ylabel(y_param)
    ax.set_title('稳定性: 0=稳定, 1=单调不稳定, 2=Hopf')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(grid_max_real, origin='lower', aspect='auto', extent=ext,
                   cmap='coolwarm', vmin=-0.5, vmax=0.5)
    ax.contour(grid_max_real, levels=[0], origin='lower', extent=ext,
              colors='black', linewidths=2)
    ax.set_xlabel(x_param); ax.set_ylabel(y_param)
    ax.set_title(r'max Re($\lambda$)')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(grid_theta, origin='lower', aspect='auto', extent=ext, cmap='hot')
    ax.set_xlabel(x_param); ax.set_ylabel(y_param)
    ax.set_title(r'稳态 $\theta_{ss}$')
    fig.colorbar(im, ax=ax)

    fig.suptitle(f'线性稳定性相图 (Bi_T={p_base.Bi_T}, Γ_A={p_base.Gamma_A})',
                fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{outdir}/05_phase_diagram_{x_param}_vs_{y_param}.png", dpi=150)
    plt.close(fig)
    print(f"  [5] 相图 → 05_phase_diagram_{x_param}_vs_{y_param}.png")


# ═══════════════════════════════════════════════════════════════════
# 分析 6: 非线性验证 — 在 Hopf 区域实际积分
# ═══════════════════════════════════════════════════════════════════

def nonlinear_verification(p_base, outdir=".",
                           x_param="Da", x_range=None,
                           y_param="S_chi", y_range=None,
                           nx=20, ny=20, t_end=500.0):
    """在参数平面上实际积分，验证真正的非线性振荡"""
    if x_range is None:
        x_range = np.linspace(2, 25, nx)
    if y_range is None:
        y_range = np.linspace(0.3, 2.5, ny)
    nx = len(x_range)
    ny = len(y_range)

    grid_osc = np.zeros((ny, nx))
    grid_amp = np.full((ny, nx), np.nan)
    grid_period = np.full((ny, nx), np.nan)

    print(f"  [6] 非线性验证 ({x_param} vs {y_param}): {nx}×{ny} = {nx*ny} 点 ...")
    total = nx * ny
    done = 0

    for j, yv in enumerate(y_range):
        for i, xv in enumerate(x_range):
            done += 1
            pp = SphereParams(**{**p_base.__dict__, x_param: xv, y_param: yv})
            pp.m_b = auto_m_b(pp)
            try:
                sol = integrate_0d(pp, y0=[pp.J_init, 0.02, 0.01],
                                  t_end=t_end, n_pts=4000)
                if sol.success:
                    osc = detect_oscillation(sol.t, sol.y[2])
                    grid_osc[j, i] = 1 if osc["oscillatory"] else 0
                    grid_amp[j, i] = osc["amp"]
                    grid_period[j, i] = osc["period"]
            except:
                grid_osc[j, i] = -1

        pct = done / total * 100
        if (j+1) % max(1, ny//5) == 0:
            print(f"      ... {pct:.0f}%")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ext = [x_range[0], x_range[-1], y_range[0], y_range[-1]]

    ax = axes[0]
    im = ax.imshow(grid_osc, origin='lower', aspect='auto', extent=ext,
                   cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xlabel(x_param); ax.set_ylabel(y_param)
    ax.set_title('非线性振荡: 1=振荡, 0=稳定, -1=失败')
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(grid_amp, origin='lower', aspect='auto', extent=ext,
                   cmap='hot')
    ax.set_xlabel(x_param); ax.set_ylabel(y_param)
    ax.set_title(r'振荡振幅 $\Delta\theta$')
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(grid_period, origin='lower', aspect='auto', extent=ext,
                   cmap='viridis')
    ax.set_xlabel(x_param); ax.set_ylabel(y_param)
    ax.set_title('振荡周期')
    fig.colorbar(im, ax=ax)

    fig.suptitle(f'非线性振荡验证 (Bi_T={p_base.Bi_T}, Γ_A={p_base.Gamma_A})',
                fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{outdir}/06_nonlinear_{x_param}_vs_{y_param}.png", dpi=150)
    plt.close(fig)
    print(f"  [6] 非线性相图 → 06_nonlinear_{x_param}_vs_{y_param}.png")


# ═══════════════════════════════════════════════════════════════════
# 分析 7: 1D code vs 0D model 参数对比
# ═══════════════════════════════════════════════════════════════════

def compare_1d_vs_0d(outdir="."):
    """
    用 1D 代码的参数构建等效 0D 模型，检验一致性。
    1D slab 弹性: m_el = Ω_e(J - 1/J)
    Sphere 弹性:  m_el = Ω_e(J^{-1/3} - J^{-1})
    """
    print("\n" + "="*60)
    print("1D slab code vs 0D sphere model 参数对比")
    print("="*60)

    # 1D slab 参数 (from scan_optimized.py)
    print("\n  1D slab 参数 (scan_optimized.py):")
    slab = {
        "phi_p0": 0.15, "chi_inf": 0.60, "S_chi": 1.00, "chi1": 1.10,
        "Omega_e": 0.12, "Da": 4.0, "Gamma_A": 1.5, "eps_T": 0.03,
        "Bi_T": 0.10, "Bi_c": 0.70, "Bi_mu": 1.00, "m_act": 6.0,
        "delta": 0.08, "alpha": 0.20, "J_init": 1.30,
    }
    for k, v in slab.items():
        print(f"    {k:12s} = {v}")

    print("\n  0D sphere 等效参数（推荐调整）:")
    print("    弹性项: slab m_el = Ω_e(J-1/J) → sphere m_el = Ω_e(J^{-1/3}-J^{-1})")
    print("    面体比: slab 1/H₀ → sphere 3/R₀·J^{2/3} (散热更快)")
    print("    催化剂: slab 无密度因子 → sphere 含 φ 因子")
    print("    → Bi_T 需要适当增大以补偿球的高面体比")
    print("    → Da 可能需要增大以补偿催化剂密度效应")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    outdir = "/home/wang/gel_model/Code/sphere_0d_results"
    import os; os.makedirs(outdir, exist_ok=True)

    print("="*60)
    print("催化剂嫁接型 LCST 凝胶球 0D 集总模型分析")
    print("="*60)

    # ──────────────────────────────────────────────
    # 场景 A: 无催化剂密度因子 (与1D slab 对比)
    # ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("场景 A: 无催化剂密度因子 (类 slab 对比)")
    print("─"*50)
    pA = SphereParams(
        phi_p0=0.15, chi_inf=0.55, S_chi=1.0, chi1=0.80,
        Omega_e=0.10,
        Da=5.0, Gamma_A=1.5, eps_T=0.03,
        m_act=4.0, use_cat_density=False,
        Bi_u=1.0, Bi_T=0.12, Bi_mu=2.0,
        J_init=1.5,
    )
    pA.m_b = auto_m_b(pA)
    print(f"  m_b = {pA.m_b:.4f}, J_init={pA.J_init}")

    plot_reaction_factor(pA, outdir)
    analyze_steady_states(pA, outdir)
    hopf_scan_Da(pA, Da_range=np.linspace(0.5, 20.0, 150), outdir=outdir)

    # 尝试几组 Da
    for Da_try in [3.0, 5.0, 8.0, 12.0]:
        pp = SphereParams(**{**pA.__dict__, "Da": Da_try})
        pp.m_b = auto_m_b(pp)
        time_integration_demo(pp, outdir=outdir, t_end=500, label=f"A_Da{Da_try}")

    # ──────────────────────────────────────────────
    # 场景 B: 含催化剂密度因子
    # ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("场景 B: 含催化剂密度因子 (催化剂嫁接型)")
    print("─"*50)
    pB = SphereParams(
        phi_p0=0.15, chi_inf=0.55, S_chi=1.0, chi1=0.80,
        Omega_e=0.10,
        Da=15.0, Gamma_A=1.5, eps_T=0.03,  # Da 增大补偿 φ 因子
        m_act=3.0, use_cat_density=True,     # m_act 适当降低
        Bi_u=1.5, Bi_T=0.10, Bi_mu=2.0,
        J_init=1.5,
    )
    pB.m_b = auto_m_b(pB)
    print(f"  m_b = {pB.m_b:.4f}")

    hopf_scan_Da(pB, Da_range=np.linspace(2.0, 40.0, 150), outdir=outdir)

    for Da_try in [8.0, 15.0, 25.0]:
        pp = SphereParams(**{**pB.__dict__, "Da": Da_try})
        pp.m_b = auto_m_b(pp)
        time_integration_demo(pp, outdir=outdir, t_end=500, label=f"B_Da{Da_try}")

    # ──────────────────────────────────────────────
    # 场景 C: 高热敏性 + 弱弹性 (LCST 窗口优化)
    # ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("场景 C: 高热敏性 (LCST 附近)")
    print("─"*50)
    pC = SphereParams(
        phi_p0=0.12, chi_inf=0.50, S_chi=1.8, chi1=0.60,
        Omega_e=0.08,
        Da=6.0, Gamma_A=2.0, eps_T=0.03,
        m_act=3.0, use_cat_density=False,
        Bi_u=1.2, Bi_T=0.08, Bi_mu=2.0,
        J_init=2.0,
    )
    pC.m_b = auto_m_b(pC)
    print(f"  m_b = {pC.m_b:.4f}")

    hopf_scan_Da(pC, Da_range=np.linspace(0.5, 15.0, 150), outdir=outdir)

    for Da_try in [2.0, 4.0, 6.0, 8.0, 10.0]:
        pp = SphereParams(**{**pC.__dict__, "Da": Da_try})
        pp.m_b = auto_m_b(pp)
        time_integration_demo(pp, outdir=outdir, t_end=600, label=f"C_Da{Da_try}")

    # ──────────────────────────────────────────────
    # 线性稳定性相图 (用最有希望的场景)
    # ──────────────────────────────────────────────
    print("\n" + "─"*50)
    print("相图扫描")
    print("─"*50)

    # Da vs S_chi (场景 A 基底)
    phase_diagram_2d(pA, outdir=outdir,
                     x_param="Da", x_range=np.linspace(1, 20, 30),
                     y_param="S_chi", y_range=np.linspace(0.3, 2.5, 30))

    # Da vs Bi_T
    phase_diagram_2d(pA, outdir=outdir,
                     x_param="Da", x_range=np.linspace(1, 20, 30),
                     y_param="Bi_T", y_range=np.linspace(0.02, 0.4, 30))

    # Da vs Gamma_A
    phase_diagram_2d(pA, outdir=outdir,
                     x_param="Da", x_range=np.linspace(1, 20, 30),
                     y_param="Gamma_A", y_range=np.linspace(0.5, 4.0, 30))

    # 非线性验证 (Da vs S_chi)
    nonlinear_verification(pA, outdir=outdir,
                           x_param="Da", x_range=np.linspace(1, 20, 18),
                           y_param="S_chi", y_range=np.linspace(0.3, 2.5, 18),
                           t_end=400)

    # 参数对比
    compare_1d_vs_0d(outdir)

    print(f"\n  所有结果保存在: {outdir}")

    print("\n" + "="*60)
    print("分析完成。")
    print("="*60)


if __name__ == "__main__":
    main()
