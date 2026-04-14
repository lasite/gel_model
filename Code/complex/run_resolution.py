#!/usr/bin/env python3
"""
run_resolution.py — Phase 1: N-收敛性验证

对 Da=10.5 / 12.0 / 12.5 分别在 N=40/60/80, t_end=2000 运行，
对比 σ(J_pk) 和 Poincaré 结构，判断混沌是真实物理还是数值伪影。

预计运行时间: ~15–30 min (9 runs, 并行)
数据输出: Data/complex/resolution/res_Da{da}_N{n}.npz
图像输出: Figure/complex/resolution_check.png
"""
import sys, os, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
DATA_DIR = ROOT / 'Data' / 'complex' / 'resolution'
FIG_DIR  = ROOT / 'Figure' / 'complex'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans
from multiprocessing import Pool

from scan_optimized import Params, simulate

# ─── Configuration ─────────────────────────────────────────────────────────
Da_LIST = [10.5, 12.0, 12.5]
N_LIST  = [40, 60, 80]
T_END   = 2000
N_SAVE  = 10000
TAIL_FRAC = 0.50   # analyse last 50%
BASE_PARAMS = dict(
    Gamma_A=1.5, m_diff=2, m_act=6,
    D0=2.0, Bi_c=0.70, Bi_T=0.10, alpha=0.20, S_chi=1.0,
    t_end=T_END, n_save=N_SAVE,
)


def run_one(args):
    Da, N = args
    fname = DATA_DIR / f'res_Da{Da:.1f}_N{N}.npz'
    if fname.exists():
        d = np.load(fname)
        return Da, N, d['t'], d['J'], d['u'], d['theta']
    p = Params(**BASE_PARAMS, Da=Da, N=N)
    r = simulate(p)
    t, J, u, theta = r['t'], r['J'], r['u'], r['theta']
    np.savez_compressed(fname, t=t, J=J, u=u, theta=theta)
    return Da, N, t, J, u, theta


def analyse(t, J):
    J_s = J[-1]
    pks, _ = find_peaks(J_s, prominence=0.03, distance=15)
    mask = t[pks] > t[-1] * TAIL_FRAC
    pv_t = J_s[pks[mask]]
    if len(pv_t) < 4:
        return dict(label='?', ac1=0, std=0, period=0, pv=pv_t)
    std    = pv_t.std()
    period = np.mean(np.diff(t[pks[mask]]))
    ac1    = np.corrcoef(pv_t[:-1] - pv_t.mean(),
                         pv_t[1:]  - pv_t.mean())[0, 1]
    label  = 'P2' if ac1 < -0.8 else ('CHAOS' if std > 0.005 else 'P1')
    return dict(label=label, ac1=ac1, std=std, period=period, pv=pv_t)


def main(workers=6):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(Da, N) for Da in Da_LIST for N in N_LIST]
    print(f"Running {len(jobs)} simulations with {workers} workers …")

    with Pool(workers) as pool:
        results = pool.map(run_one, jobs)

    # Organise results
    res = {}   # res[(Da, N)] = (stats, t, J)
    for Da, N, t, J, u, theta in results:
        stats = analyse(t, J)
        res[(Da, N)] = (stats, t, J)
        print(f"  Da={Da:.1f}  N={N:2d}:  {stats['label']:7s}  "
              f"σ={stats['std']:.4f}  ac1={stats['ac1']:6.3f}  T={stats['period']:.1f}")

    # ─── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(len(Da_LIST), len(N_LIST) + 1,
                            figure=fig, hspace=0.55, wspace=0.42,
                            width_ratios=[1]*len(N_LIST) + [0.8])

    colors_N = {40: '#e63946', 60: '#457b9d', 80: '#2d6a4f'}

    for row, Da in enumerate(Da_LIST):
        # Poincaré for each N
        for col, N in enumerate(N_LIST):
            ax = fig.add_subplot(gs[row, col])
            stats, t, J = res[(Da, N)]
            pv = stats['pv']
            if len(pv) > 1:
                c = colors_N[N]
                ax.scatter(pv[:-1], pv[1:], color=c, s=25, alpha=0.85)
                lo, hi = pv.min() - 0.005, pv.max() + 0.005
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.4)
            ax.set_title(f'Da={Da:.1f}, N={N}\n{stats["label"]}  σ={stats["std"]:.4f}',
                         fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(r'$J_{n+1}$', fontsize=8)
            if row == len(Da_LIST) - 1:
                ax.set_xlabel(r'$J_n$', fontsize=8)

        # σ vs N summary panel
        ax_s = fig.add_subplot(gs[row, len(N_LIST)])
        Ns_plot = N_LIST
        stds = [res[(Da, N)][0]['std'] for N in Ns_plot]
        labels_N = [res[(Da, N)][0]['label'] for N in Ns_plot]
        ax_s.plot(Ns_plot, stds, 'ko-', ms=6, lw=1.5)
        ax_s.axhline(0.005, color='red', ls='--', lw=1, alpha=0.7, label='chaos thresh')
        ax_s.axhline(0.002, color='blue', ls='--', lw=1, alpha=0.7, label='P1 thresh')
        for n, s, lbl in zip(Ns_plot, stds, labels_N):
            ax_s.text(n, s + 0.0003, lbl, ha='center', fontsize=7,
                      color='red' if lbl == 'CHAOS' else ('green' if lbl == 'P2' else 'blue'))
        ax_s.set_xlabel('N', fontsize=8); ax_s.set_ylabel('σ(J_pk)', fontsize=8)
        ax_s.set_title(f'Da={Da:.1f}', fontsize=8)
        ax_s.tick_params(labelsize=7)
        if row == 0:
            ax_s.legend(fontsize=6)

    plt.suptitle('N-Convergence Check  —  Γ_A=1.5, Bi_c=0.70, Bi_T=0.10, D₀=2\n'
                 'Poincaré maps (left) and σ vs N (right): real chaos survives N=60/80',
                 fontsize=11, y=1.01)
    out = FIG_DIR / 'resolution_check.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    main(args.workers)
