#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_kymograph.py — 时空图 + 时间序列可视化

用法：
    python Code/complex/plot_kymograph.py          # Da=0.1, GA=3.0 (default)
    python Code/complex/plot_kymograph.py --Da 4.0 --GA 1.5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from Code.scan_optimized import Params, simulate


def run_and_plot(Da=0.1, GA=3.0, N=60, t_end=600, n_save=6000,
                 Bi_c=0.70, Bi_T=0.10, D0=2.0,
                 t_show=None,
                 out_path=None):

    print(f"Running simulation: Da={Da}, GA={GA}, N={N}, t_end={t_end}")
    p = Params(
        Da=Da, Gamma_A=GA, N=N, t_end=t_end, n_save=n_save,
        Bi_c=Bi_c, Bi_T=Bi_T, D0=D0,
    )
    res = simulate(p)

    if not res['success']:
        print("Simulation failed!")
        return

    t = res['t']           # (n_save,)
    x = res['x']           # (N,)
    J = res['J']           # (N, n_save)
    theta = res['theta']   # (N, n_save)
    u = res['u']           # (N, n_save)

    print(f"  t: {t[0]:.1f} – {t[-1]:.1f},  J range: {J.min():.3f} – {J.max():.3f}")
    print(f"  theta max: {theta.max():.3f},  u min: {u.min():.4f}")

    # ----------------------------------------------------------------
    # 只展示稳定振荡部分（去除前20%暂态）
    i_start = int(0.2 * len(t))
    t_plot = t[i_start:]
    J_plot = J[:, i_start:]
    th_plot = theta[:, i_start:]
    u_plot = u[:, i_start:]

    # 进一步裁剪到 t_show 范围
    if t_show is not None:
        t_max = t_plot[0] + t_show
        mask = t_plot <= t_max
        t_plot  = t_plot[mask]
        J_plot  = J_plot[:, mask]
        th_plot = th_plot[:, mask]
        u_plot  = u_plot[:, mask]

    # 归一化 x 轴 (0=center, 1=surface)
    x_norm = x / x[-1]

    # ----------------------------------------------------------------
    # 布局：2行
    # 第1行：3个 kymograph (J, θ, u)
    # 第2行：2个时间序列 (x=0中心，x=L表面)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.38, wspace=0.35)

    ax_J  = fig.add_subplot(gs[0, 0])
    ax_th = fig.add_subplot(gs[0, 1])
    ax_u  = fig.add_subplot(gs[0, 2])
    ax_ctr  = fig.add_subplot(gs[1, 0:2])
    ax_surf = fig.add_subplot(gs[1, 2])

    kymo_kw = dict(aspect='auto', origin='lower', interpolation='nearest')

    # --- J kymograph ---
    im_J = ax_J.imshow(J_plot, extent=[t_plot[0], t_plot[-1], 0, 1],
                       cmap='RdBu_r', vmin=0.1, vmax=1.5, **kymo_kw)
    fig.colorbar(im_J, ax=ax_J, label='$J$', fraction=0.046, pad=0.04)
    ax_J.set_xlabel('$t$ [τ]')
    ax_J.set_ylabel('$x/L$')
    ax_J.set_title('Swelling ratio $J(x,t)$')

    # --- θ kymograph ---
    vth_max = max(th_plot.max(), 0.5)
    im_th = ax_th.imshow(th_plot, extent=[t_plot[0], t_plot[-1], 0, 1],
                         cmap='hot', vmin=0, vmax=vth_max, **kymo_kw)
    fig.colorbar(im_th, ax=ax_th, label='$\\theta$', fraction=0.046, pad=0.04)
    ax_th.set_xlabel('$t$ [τ]')
    ax_th.set_ylabel('$x/L$')
    ax_th.set_title('Temperature $\\theta(x,t)$')

    # --- u kymograph ---
    im_u = ax_u.imshow(u_plot, extent=[t_plot[0], t_plot[-1], 0, 1],
                       cmap='YlOrRd_r', vmin=0, vmax=1, **kymo_kw)
    fig.colorbar(im_u, ax=ax_u, label='$u$', fraction=0.046, pad=0.04)
    ax_u.set_xlabel('$t$ [τ]')
    ax_u.set_ylabel('$x/L$')
    ax_u.set_title('Reactant $u(x,t)$')

    # ----------------------------------------------------------------
    # 时间序列：中心 x=0 (i=0) 和表面 x=L (i=-1)
    i_ctr  = 0
    i_surf = -1

    colors = {'J': '#1f77b4', 'theta': '#d62728', 'u': '#2ca02c'}

    # --- 中心时间序列 ---
    ax2 = ax_ctr.twinx()
    ln1, = ax_ctr.plot(t_plot, J_plot[i_ctr], color=colors['J'],  lw=1.5, label='$J$')
    ln2, = ax_ctr.plot(t_plot, u_plot[i_ctr], color=colors['u'],  lw=1.5, label='$u$', ls='--')
    ln3, = ax2.plot(   t_plot, th_plot[i_ctr], color=colors['theta'], lw=1.5, label='$\\theta$', ls=':')
    ax_ctr.set_xlabel('$t$ [τ]')
    ax_ctr.set_ylabel('$J$,  $u$')
    ax2.set_ylabel('$\\theta$', color=colors['theta'])
    ax2.tick_params(axis='y', labelcolor=colors['theta'])
    ax_ctr.set_title(f'Center ($x=0$):  $J$, $u$, $\\theta$   [Da={Da}, $\\Gamma_A$={GA}]')
    lns = [ln1, ln2, ln3]
    ax_ctr.legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=9)

    # --- 表面时间序列 ---
    ax2s = ax_surf.twinx()
    ax_surf.plot(t_plot, J_plot[i_surf], color=colors['J'],  lw=1.5, label='$J$')
    ax_surf.plot(t_plot, u_plot[i_surf], color=colors['u'],  lw=1.5, label='$u$', ls='--')
    ax2s.plot(   t_plot, th_plot[i_surf], color=colors['theta'], lw=1.5, label='$\\theta$', ls=':')
    ax_surf.set_xlabel('$t$ [τ]')
    ax_surf.set_ylabel('$J$,  $u$')
    ax2s.set_ylabel('$\\theta$', color=colors['theta'])
    ax2s.tick_params(axis='y', labelcolor=colors['theta'])
    ax_surf.set_title(f'Surface ($x=L$)')
    ax_surf.legend(loc='upper right', fontsize=9)

    # ----------------------------------------------------------------
    fig.suptitle(
        f'Accessibility-dominated oscillation:  Da={Da},  $\\Gamma_A$={GA},  N={N}',
        fontsize=13, y=1.01
    )

    if out_path is None:
        out_path = f"Figure/complex/kymograph_Da{Da:.1f}_GA{GA:.1f}.png"


    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Da',   type=float, default=0.1)
    parser.add_argument('--GA',   type=float, default=3.0)
    parser.add_argument('--N',    type=int,   default=60)
    parser.add_argument('--t',    type=float, default=600)
    parser.add_argument('--t_show', type=float, default=None, help='显示时间窗口长度 (从暂态后开始)')
    parser.add_argument('--out',  type=str,   default=None)
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    run_and_plot(Da=args.Da, GA=args.GA, N=args.N, t_end=args.t,
                 t_show=args.t_show, out_path=args.out)
