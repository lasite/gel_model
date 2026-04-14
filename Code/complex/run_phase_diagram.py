#!/usr/bin/env python3
"""
run_phase_diagram.py — Da × Bc 二维相图

参数空间:
  Da  = 8, 9, 10, ..., 17   (10 点, step=1)
  Bc  = 0.40, 0.45, ..., 0.95 (12 点, step=0.05)
  固定: GA=1.5, Bi_T=0.10, D0=2.0, N=60, t=1500τ

策略:
  - 优先复用 Data/complex/p2_search/ 中已有的 N=60 数据（50 点）
  - 新数据存入 Data/complex/phase_diagram/pd_Da{da}_Bc{bc}_N60.npz
  - 使用 classify_oscillation.classify() 统一分类

输出:
  Figure/complex/phase_diagram_Da_Bc.png

用法:
  python run_phase_diagram.py              # 计算全部缺失点 + 绘图
  python run_phase_diagram.py --plot-only  # 仅用已有数据绘图
  python run_phase_diagram.py --workers 4  # 指定并行数
"""
import sys, os, argparse, time, numpy as np
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))

DATA_NEW = ROOT / 'Data' / 'complex' / 'phase_diagram'
DATA_P2S = ROOT / 'Data' / 'complex' / 'p2_search'
FIG_DIR  = ROOT / 'Figure' / 'complex'
DATA_NEW.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from scan_optimized import Params, simulate
from classify_oscillation import classify

# ── 参数网格 ──────────────────────────────────────────────────────────────────
DA_VALS = np.arange(8.0, 18.0, 1.0)       # 8,9,...,17 (10 点)
BC_VALS = np.arange(0.40, 0.96, 0.05)     # 0.40,0.45,...,0.95 (12 点)
BC_VALS = np.round(BC_VALS, 2)

BASE_PARAMS = dict(
    N=60, t_end=1500, n_save=7500,
    Gamma_A=1.5, m_diff=2, m_act=6,
    D0=2.0, Bi_T=0.10, alpha=0.20, S_chi=1.0,
)

LABEL_COLOR = {
    'P1':    '#2196F3',   # 蓝
    'P2':    '#4CAF50',   # 绿
    'P3':    '#FF9800',   # 橙
    'P4':    '#9C27B0',   # 紫
    'chaos': '#F44336',   # 红
    'steady':'#BDBDBD',   # 灰
    '?':     '#212121',   # 黑
}

# ── 数据查找 ──────────────────────────────────────────────────────────────────

def find_existing(Da, Bc):
    """查找已有数据文件（优先 p2_search，其次 phase_diagram）"""
    # p2_search 命名: s_GA1.5_Da{da}_Bc{bc}_N60.npz
    p1 = DATA_P2S / f's_GA1.5_Da{Da:.1f}_Bc{Bc:.2f}_N60.npz'
    if p1.exists():
        return str(p1)
    # phase_diagram 命名
    p2 = DATA_NEW / f'pd_Da{Da:.1f}_Bc{Bc:.2f}_N60.npz'
    if p2.exists():
        return str(p2)
    return None


def run_one(args):
    Da, Bc = args
    out = DATA_NEW / f'pd_Da{Da:.1f}_Bc{Bc:.2f}_N60.npz'
    if out.exists():
        return Da, Bc, str(out)
    existing = find_existing(Da, Bc)
    if existing:
        return Da, Bc, existing
    # 新计算
    p = Params(**BASE_PARAMS, Da=float(Da), Bi_c=float(Bc))
    r = simulate(p)
    np.savez_compressed(out, t=r['t'], J=r['J'], u=r['u'], theta=r['theta'])
    return Da, Bc, str(out)


# ── 分类 ─────────────────────────────────────────────────────────────────────

def classify_all(file_map):
    """对所有文件分类，返回 {(Da,Bc): result_dict}"""
    results = {}
    for (Da, Bc), fpath in sorted(file_map.items()):
        try:
            r = classify(fpath)
            results[(Da, Bc)] = r
        except Exception as e:
            results[(Da, Bc)] = {'label': '?', 'T_mean': np.nan, 'T_ratio': np.nan,
                                  'n_peaks': 0, 'pe': np.nan, 'confidence': 'low'}
            print(f"  ERROR classifying ({Da},{Bc}): {e}")
    return results


# ── 可视化 ────────────────────────────────────────────────────────────────────

def plot_phase_diagram(results, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    fig.suptitle('Phase diagram: Da × Bi_c  (GA=1.5, N=60, t=1500τ)', fontsize=13)

    # ── Panel 1: 相图（离散颜色格）──────────────────────────────────────
    ax = axes[0]
    all_labels = sorted(set(r['label'] for r in results.values()))

    for (Da, Bc), r in results.items():
        color = LABEL_COLOR.get(r['label'], '#212121')
        rect = mpatches.FancyBboxPatch(
            (Bc - 0.024, Da - 0.48), 0.048, 0.96,
            boxstyle="square,pad=0", fc=color, ec='white', lw=0.5, alpha=0.9
        )
        ax.add_patch(rect)
        # 标注 T_mean（仅非 steady）
        if r['label'] not in ('steady', '?') and not np.isnan(r['T_mean']):
            ax.text(Bc, Da, f"{r['T_mean']:.0f}", ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')

    ax.set_xlim(BC_VALS.min() - 0.05, BC_VALS.max() + 0.05)
    ax.set_ylim(DA_VALS.min() - 0.7, DA_VALS.max() + 0.7)
    ax.set_xticks(BC_VALS)
    ax.set_yticks(DA_VALS)
    ax.set_xticklabels([f'{b:.2f}' for b in BC_VALS], rotation=45, fontsize=8)
    ax.set_yticklabels([f'{d:.0f}' for d in DA_VALS], fontsize=8)
    ax.set_xlabel('Bi_c (reactant supply)', fontsize=10)
    ax.set_ylabel('Da (Damköhler)', fontsize=10)
    ax.set_title('Oscillation type (color = label, text = T̄ [τ])', fontsize=10)
    ax.set_aspect('auto')

    # 图例
    legend_handles = [mpatches.Patch(fc=LABEL_COLOR.get(lbl,'#212121'), label=lbl)
                      for lbl in ['steady', 'P1', 'P2', 'P3', 'P4', 'chaos'] if lbl in all_labels]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    # ── Panel 2: T_ratio 图（P2 强度）───────────────────────────────────
    ax2 = axes[1]
    T_ratio_grid = np.ones((len(DA_VALS), len(BC_VALS))) * np.nan
    for i, Da in enumerate(DA_VALS):
        for j, Bc in enumerate(BC_VALS):
            r = results.get((Da, Bc))
            if r and r['label'] == 'P2':
                T_ratio_grid[i, j] = r['T_ratio']

    im = ax2.imshow(T_ratio_grid, origin='lower', aspect='auto',
                    extent=[BC_VALS[0]-0.025, BC_VALS[-1]+0.025,
                            DA_VALS[0]-0.5,   DA_VALS[-1]+0.5],
                    cmap='YlOrRd', vmin=1.0, vmax=3.0)
    plt.colorbar(im, ax=ax2, label='T_long / T_short', shrink=0.8)

    # Overlay P2 markers
    for (Da, Bc), r in results.items():
        if r['label'] == 'P2':
            ax2.scatter(Bc, Da, c='green', s=60, marker='*', zorder=5)

    ax2.set_xlim(BC_VALS.min()-0.05, BC_VALS.max()+0.05)
    ax2.set_ylim(DA_VALS.min()-0.7,  DA_VALS.max()+0.7)
    ax2.set_xticks(BC_VALS)
    ax2.set_yticks(DA_VALS)
    ax2.set_xticklabels([f'{b:.2f}' for b in BC_VALS], rotation=45, fontsize=8)
    ax2.set_yticklabels([f'{d:.0f}' for d in DA_VALS], fontsize=8)
    ax2.set_xlabel('Bi_c', fontsize=10)
    ax2.set_ylabel('Da', fontsize=10)
    ax2.set_title('P2 period ratio T_long/T_short\n(★ = confirmed P2)', fontsize=10)

    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / 'phase_diagram_Da_Bc.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved {out_path}")
    plt.close()


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='List missing points without computing')
    args = parser.parse_args()

    # 构建完整任务列表
    all_jobs = [(Da, round(Bc, 2)) for Da in DA_VALS for Bc in BC_VALS]

    # 区分已有 / 需计算
    to_compute = [(Da, Bc) for Da, Bc in all_jobs if find_existing(Da, Bc) is None]
    existing   = [(Da, Bc) for Da, Bc in all_jobs if find_existing(Da, Bc) is not None]

    print(f"Grid: {len(DA_VALS)}×{len(BC_VALS)} = {len(all_jobs)} points")
    print(f"  Existing: {len(existing)}  (reused from p2_search or phase_diagram/)")
    print(f"  To compute: {len(to_compute)}")

    if args.dry_run:
        print("\nMissing points:")
        for Da, Bc in sorted(to_compute):
            print(f"  Da={Da:.1f}, Bc={Bc:.2f}")
        return

    # ── 计算缺失点 ──
    file_map = {(Da, Bc): find_existing(Da, Bc) for Da, Bc in existing}

    if to_compute and not args.plot_only:
        print(f"\nComputing {len(to_compute)} new points with {args.workers} workers...")
        t0 = time.time()
        with Pool(processes=args.workers) as pool:
            for i, (Da, Bc, fpath) in enumerate(pool.imap_unordered(run_one, to_compute)):
                file_map[(Da, Bc)] = fpath
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (len(to_compute) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(to_compute)}] Da={Da:.1f} Bc={Bc:.2f}  "
                      f"ETA: {eta/60:.1f}min", flush=True)
        print(f"Computation done in {(time.time()-t0)/60:.1f} min")
    elif args.plot_only:
        # Only include points with existing data
        for Da, Bc in to_compute:
            fp = find_existing(Da, Bc)
            if fp:
                file_map[(Da, Bc)] = fp
        print(f"Plot-only mode: using {len(file_map)} available points")

    # ── 分类 ──
    print(f"\nClassifying {len(file_map)} data files...")
    results = classify_all(file_map)

    # Print summary
    from collections import Counter
    counts = Counter(r['label'] for r in results.values())
    print("Label counts:", dict(counts))
    p2_points = [(Da, Bc) for (Da, Bc), r in results.items() if r['label'] == 'P2']
    print(f"P2 locations ({len(p2_points)}):", sorted(p2_points))

    # ── 绘图 ──
    plot_phase_diagram(results)


if __name__ == '__main__':
    main()
