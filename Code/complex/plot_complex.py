#!/usr/bin/env python3
"""
plot_complex.py — Phase 4: 出版质量时空图

绘制 P1 / P2 / chaos 三类典型动力学的时空图，
每张图包含：J kymograph, θ kymograph, u kymograph (log), J_surface 时间序列, Poincaré。

使用方法:
  python plot_complex.py                     # 自动从缓存数据中选最佳参数
  python plot_complex.py --Da 12.0 --mode chaos   # 指定参数
  python plot_complex.py --Da 4.0 --mode p1

输入: Data/complex/ 或 Data/ 下的 .npz 文件
输出: Figure/complex/{mode}_kymograph.png
"""
import sys, argparse, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
DATA_DIR    = ROOT / 'Data'
COMPLEX_DIR = ROOT / 'Data' / 'complex'
FIG_DIR     = ROOT / 'Figure' / 'complex'

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans

from scan_optimized import Params, simulate

BASE_PARAMS = dict(
    Gamma_A=1.5, m_diff=2, m_act=6,
    D0=2.0, Bi_c=0.70, Bi_T=0.10, alpha=0.20, S_chi=1.0,
)

TAIL_FRAC = 0.50


def get_data(Da, N, t_end, tag=''):
    """Load or simulate; cache in Data/complex/long_runs/."""
    long_dir = COMPLEX_DIR / 'long_runs'
    long_dir.mkdir(parents=True, exist_ok=True)
    fname = long_dir / f'kymo_{tag}Da{Da:.1f}_N{N}_t{t_end}.npz'
    if fname.exists():
        d = np.load(fname)
        return d['t'], d['J'], d['u'], d['theta']
    p = Params(**BASE_PARAMS, Da=Da, N=N, t_end=t_end, n_save=t_end * 10)
    r = simulate(p)
    np.savez_compressed(fname, t=r['t'], J=r['J'], u=r['u'], theta=r['theta'])
    return r['t'], r['J'], r['u'], r['theta']


def classify_peaks(t, J, tail_frac=TAIL_FRAC):
    J_s = J[-1]
    pks, _ = find_peaks(J_s, prominence=0.03, distance=15)
    mask = t[pks] > t[-1] * tail_frac
    pv_t = J_s[pks[mask]]
    pks_t = pks[mask]
    if len(pv_t) < 4:
        return pks, J_s[pks], None, None
    # k-means 2 for L/S classification
    try:
        cents, _ = kmeans(pv_t.reshape(-1, 1).astype(float), 2)
        thresh = sorted(cents.flatten())
        thresh = (thresh[0] + thresh[1]) / 2
        lm = pv_t > thresh
    except Exception:
        lm = pv_t > pv_t.median()
    return pks, J_s[pks], pks_t, lm


def make_kymograph(t, J, u, theta, title, out_path, mode='chaos',
                   t_show_start=None):
    N = J.shape[0]
    x = np.linspace(0, 1, N)

    if t_show_start is not None:
        i0 = np.searchsorted(t, t_show_start)
        t, J, u, theta = t[i0:], J[:, i0:], u[:, i0:], theta[:, i0:]

    J_s = J[-1]
    pks_all, pv_all, pks_t, lm_t = classify_peaks(
        t, J, tail_frac=0.0)   # use all range for kymograph

    pv = J_s[pks_all]
    norm_p = (pv - pv.min()) / (pv.max() - pv.min() + 1e-12)

    ext = [t[0], t[-1], 0, 1]
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.35,
                            width_ratios=[3, 1], height_ratios=[2.2, 2.2, 2.2, 1.8])

    def vlines(ax, on_top=True):
        for i, pk in enumerate(pks_all):
            c = plt.cm.plasma(norm_p[i])
            ax.axvline(t[pk], color=c, lw=0.7, alpha=0.6)

    # J kymograph
    axJ = fig.add_subplot(gs[0, 0])
    imJ = axJ.imshow(J, aspect='auto', origin='lower', extent=ext,
                     cmap='RdBu_r', vmin=0.05, vmax=2.3)
    plt.colorbar(imJ, ax=axJ, label='J', shrink=0.90)
    vlines(axJ)
    axJ.set_ylabel('x'); axJ.set_title(title, fontweight='bold', fontsize=10)

    # θ kymograph
    axT = fig.add_subplot(gs[1, 0])
    imT = axT.imshow(theta, aspect='auto', origin='lower', extent=ext,
                     cmap='hot', vmin=0, vmax=theta.max())
    plt.colorbar(imT, ax=axT, label='θ', shrink=0.90)
    vlines(axT)
    axT.set_ylabel('x'); axT.set_title('θ(x,τ)', fontsize=10)

    # u kymograph
    axU = fig.add_subplot(gs[2, 0])
    upos = np.clip(u, 1e-10, None)
    imU = axU.imshow(upos, aspect='auto', origin='lower', extent=ext,
                     cmap='inferno',
                     norm=LogNorm(vmin=upos.min(), vmax=upos.max()))
    plt.colorbar(imU, ax=axU, label='u (log)', shrink=0.90)
    vlines(axU)
    axU.set_ylabel('x'); axU.set_title('u(x,τ)', fontsize=10)

    # Time series
    axL = fig.add_subplot(gs[3, 0])
    axL.plot(t, J_s, 'k-', lw=0.9, alpha=0.9)
    sc = axL.scatter(t[pks_all], pv, c=norm_p, cmap='plasma', s=25, zorder=5)
    axL.set_xlabel('τ', fontsize=11); axL.set_ylabel('J(1,τ)')
    axL.set_title('Surface flux time series', fontsize=9)
    axL.set_xlim(t[0], t[-1])
    if len(pks_all) > 1:
        T_mean = np.mean(np.diff(t[pks_all]))
        std    = pv.std()
        ac1    = np.corrcoef(pv[:-1] - pv.mean(), pv[1:] - pv.mean())[0, 1]
        axL.set_title(f'σ={std:.4f}  ac₁={ac1:.3f}  T̄={T_mean:.1f}τ', fontsize=9)

    # Poincaré
    axP = fig.add_subplot(gs[0, 1])
    if len(pv) > 1:
        axP.scatter(pv[:-1], pv[1:], c=norm_p[:-1], cmap='plasma', s=25, alpha=0.85)
        lo, hi = pv.min() - 0.005, pv.max() + 0.005
        axP.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.4)
    axP.set_xlabel(r'$J_n$', fontsize=9); axP.set_ylabel(r'$J_{n+1}$', fontsize=9)
    axP.set_title('Poincaré map', fontsize=9)
    axP.tick_params(labelsize=7)

    # Spatial profiles at representative peaks
    axSJ = fig.add_subplot(gs[1, 1])
    axSU = fig.add_subplot(gs[2, 1])
    if pks_t is not None and len(pks_t) > 1 and lm_t is not None:
        cL, cS = '#1a7a1a', '#cc3300'
        li = pks_t[lm_t]; si = pks_t[~lm_t]
        if len(li) > 0 and len(si) > 0:
            JL = J[:, li].mean(1); JS = J[:, si].mean(1)
            uL = u[:, li].mean(1); uS = u[:, si].mean(1)
            axSJ.plot(x, JL, color=cL, lw=2, label='L')
            axSJ.plot(x, JS, color=cS, lw=2, label='S')
            axSJ.fill_between(x, JS, JL, alpha=0.2, color='purple')
            axSU.semilogy(x, np.clip(uL, 1e-10, None), color=cL, lw=2, label='L')
            axSU.semilogy(x, np.clip(uS, 1e-10, None), color=cS, lw=2, label='S')
            axSJ.legend(fontsize=8); axSU.legend(fontsize=8)
    else:
        # single profile at last peak
        if len(pks_all) > 0:
            pk = pks_all[-1]
            axSJ.plot(x, J[:, pk], 'k-', lw=2)
            axSU.semilogy(x, np.clip(u[:, pk], 1e-10, None), 'k-', lw=2)
    axSJ.axhline(1, color='gray', ls=':', lw=0.8)
    axSJ.set_xlabel('x'); axSJ.set_ylabel('J')
    axSJ.set_title('J profile at peak', fontsize=8)
    axSU.set_xlabel('x'); axSU.set_ylabel('u')
    axSU.set_title('u profile at peak', fontsize=8)

    # Empty bottom-right for colorbar info
    ax_info = fig.add_subplot(gs[3, 1])
    ax_info.axis('off')
    ax_info.text(0.1, 0.8, f'Mode: {mode.upper()}', transform=ax_info.transAxes,
                 fontsize=10, fontweight='bold')
    ax_info.text(0.1, 0.6, f'Da = {title.split("Da=")[1].split(",")[0] if "Da=" in title else "?"}',
                 transform=ax_info.transAxes, fontsize=9)
    ax_info.text(0.1, 0.4, 'Peak colour: plasma scale\n(low=dark, high=bright)',
                 transform=ax_info.transAxes, fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Da',    type=float, default=None)
    ap.add_argument('--N',     type=int,   default=40)
    ap.add_argument('--mode',  choices=['p1', 'p2', 'chaos'], default=None)
    ap.add_argument('--t_end', type=int,   default=None)
    ap.add_argument('--t_show_start', type=float, default=None)
    args = ap.parse_args()

    # Default presets
    presets = {
        'p1':    dict(Da=4.0,   N=40, t_end=500,  title='Da=4.0, Bi_c=0.70  (Period-1)'),
        'p2':    dict(Da=12.5,  N=40, t_end=2000, title='Da=12.5, Bi_c=0.70  (Period-2 candidate)'),
        'chaos': dict(Da=12.0,  N=40, t_end=3000, title='Da=12.0, Bi_c=0.70  (Chaos)'),
    }

    if args.mode is None:
        # Run all three
        for mode, cfg in presets.items():
            Da = cfg['Da']; N = cfg['N']; t_end = cfg['t_end']
            print(f'\n=== {mode.upper()}: Da={Da}, N={N}, t_end={t_end} ===')
            t, J, u, theta = get_data(Da, N, t_end, tag=mode)
            out = FIG_DIR / f'{mode}_kymograph.png'
            make_kymograph(t, J, u, theta, cfg['title'], out, mode=mode)
    else:
        cfg = presets[args.mode]
        Da    = args.Da    or cfg['Da']
        N     = args.N     or cfg['N']
        t_end = args.t_end or cfg['t_end']
        title = f'Da={Da:.1f}, Bi_c=0.70  ({args.mode.upper()})'
        print(f'=== {args.mode.upper()}: Da={Da}, N={N}, t_end={t_end} ===')
        t, J, u, theta = get_data(Da, N, t_end, tag=args.mode)
        out = FIG_DIR / f'{args.mode}_kymograph.png'
        make_kymograph(t, J, u, theta, title, out, mode=args.mode,
                       t_show_start=args.t_show_start)


if __name__ == '__main__':
    main()
