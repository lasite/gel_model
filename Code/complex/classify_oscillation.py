#!/usr/bin/env python3
"""
classify_oscillation.py — 振荡类型分类器

基于 x=L 处的峰间隔序列判断振荡类型：
  steady / P1 / P2 / P{n} / chaos

用法（模块）:
    from classify_oscillation import classify
    result = classify('Data/complex/p2_search/s_GA1.5_Da12.0_Bc0.70_N60.npz')
    print(result['label'], result['T_mean'], result['T_ratio'])

用法（命令行批量测试）:
    python classify_oscillation.py Data/complex/p2_search/*.npz
    python classify_oscillation.py --all          # 测试所有已知数据集
    python classify_oscillation.py --demo         # 生成诊断可视化图
"""
import sys, os, re
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]

# ──────────────────────────────────────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────────────────────────────────────

def _permutation_entropy(x, m=5):
    """排列熵，m 为嵌入维度，用于混沌检测。返回归一化熵值 [0,1]。"""
    from itertools import permutations
    from math import factorial, log2
    n = len(x)
    if n < m + 1:
        return 0.0
    # 统计所有 m 阶排列的频率
    counts = {}
    for i in range(n - m):
        key = tuple(np.argsort(x[i:i+m]))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    H = -np.sum(probs * np.log2(probs + 1e-12))
    H_max = np.log2(factorial(m))
    return H / H_max if H_max > 0 else 0.0


def _kmeans_1d(x, k, n_init=10, max_iter=200):
    """一维 k-means，返回 (labels, centers, inertia)。多次初始化取最优。"""
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    rng = np.random.default_rng(42)
    for _ in range(n_init):
        # 随机初始化
        centers = rng.choice(x, size=k, replace=False)
        centers = np.sort(centers)
        for _ in range(max_iter):
            labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            new_centers = np.array([x[labels == j].mean() if (labels == j).any() else centers[j]
                                    for j in range(k)])
            if np.allclose(centers, new_centers, rtol=1e-6):
                break
            centers = new_centers
        inertia = sum(((x[labels == j] - centers[j])**2).sum() for j in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def _best_period(T, max_k=4):
    """
    对间隔序列 T 估计最佳周期数 k。
    返回 (k, labels, centers) —— labels 是每个间隔的 cluster 标签。
    若 k=1（单一周期）直接返回，否则用 elbow 法选 k。
    """
    if len(T) < 4:
        return 1, np.zeros(len(T), dtype=int), np.array([T.mean()])

    cv = T.std() / T.mean() if T.mean() > 0 else 0
    if cv < 0.08:
        return 1, np.zeros(len(T), dtype=int), np.array([T.mean()])

    inertias = []
    results = []
    for k in range(1, min(max_k + 1, len(T) // 2 + 1)):
        lbl, ctr, iner = _kmeans_1d(T, k)
        inertias.append(iner)
        results.append((k, lbl, ctr))

    if len(inertias) == 1:
        return results[0]

    # Elbow: 选最小 k 使得增加下一个 k 改善不超过 20%
    best_k = 1
    for i in range(1, len(inertias)):
        if inertias[0] > 0:
            improvement = (inertias[i-1] - inertias[i]) / inertias[0]
            if improvement > 0.15:
                best_k = i + 1  # k = i+1
    return results[best_k - 1]


def _check_cyclic_order(labels, k, max_error_frac=0.15):
    """
    检查 labels 序列是否严格按周期 k 循环（允许少量乱序）。
    例如 k=2: [0,1,0,1,0,1,...] 或 [1,0,1,0,...]
    返回 True/False。
    """
    if k == 1:
        return True
    n = len(labels)
    # 尝试所有可能的起始排列
    from itertools import permutations
    best_errors = n
    for perm in permutations(range(k)):
        pattern = [perm[i % k] for i in range(n)]
        errors = sum(1 for a, b in zip(labels, pattern) if a != b)
        best_errors = min(best_errors, errors)
    return best_errors / n <= max_error_frac


def classify(npz_path, tail_frac=0.50, min_peaks=4, verbose=False):
    """
    判断一个模拟文件的振荡类型。

    参数
    ----
    npz_path : str or Path
        .npz 文件路径，需含 'J', 't' 数组。
    tail_frac : float
        使用模拟时间后 tail_frac 比例的数据（去除瞬态）。
    min_peaks : int
        至少需要检测到这么多峰才判为振荡，否则为 steady。

    返回
    ----
    dict 含以下键：
        label      : str  — "steady" / "P1" / "P2" / "P{n}" / "chaos"
        T_mean     : float — 平均振荡周期 τ（steady 时为 NaN）
        T_ratio    : float — T_long/T_short（P1 时为 1.0，chaos 时为 NaN）
        n_peaks    : int  — 检测到的峰数
        pe         : float — 排列熵（混沌指标，0=规则，1=完全随机）
        confidence : str  — "high"（峰数≥8）或 "low"
    """
    d = np.load(npz_path, allow_pickle=True)
    J = d['J']; t = d['t']
    N = J.shape[0]

    # x=L 时间序列（后 tail_frac 部分）
    cut = int(len(t) * (1 - tail_frac))
    J_xL = J[N - 1, cut:]
    t_xL = t[cut:]

    # ── Step 1: 稳态检测 ──────────────────────────────────────────────
    mean_J = J_xL.mean()
    std_J  = J_xL.std()
    if mean_J > 0 and std_J / mean_J < 0.005:
        return dict(label='steady', T_mean=np.nan, T_ratio=np.nan,
                    n_peaks=0, pe=0.0, confidence='high')

    # ── Step 2: 峰值检测 ──────────────────────────────────────────────
    peaks, _ = find_peaks(J_xL, prominence=0.005, height=0.5)
    n_peaks = len(peaks)
    if n_peaks < min_peaks:
        return dict(label='steady', T_mean=np.nan, T_ratio=np.nan,
                    n_peaks=n_peaks, pe=0.0, confidence='low')

    T = np.diff(t_xL[peaks])   # 间隔序列
    Jpk = J_xL[peaks]          # 峰高序列
    confidence = 'high' if n_peaks >= 8 else 'low'

    # ── Step 3: 周期数估计（间隔 + 振幅双轨）────────────────────────
    # Track A: 间隔序列分析（适用于 T_long≠T_short 型 P2，如 Da=13.5）
    k_T, labels_T, centers_T = _best_period(T, max_k=4)

    # Track B: 峰高振幅分析（适用于 T≈常数但幅度交替型 P2，如 Da=12.0 N=60）
    ac1_J = np.corrcoef(Jpk[:-1], Jpk[1:])[0, 1] if len(Jpk) > 5 else 0.0
    # 若 k_T=1 但 ac1(J)<-0.5，判为 P2（振幅倍周期）
    if k_T == 1 and ac1_J < -0.5:
        k = 2
        # 用振幅做 k-means 确认两个幅值中心
        labels, centers, _ = _kmeans_1d(Jpk, 2)
        T_ratio = 1.0   # 间隔几乎相等，比值无意义，统一报 1.0
        pe = _permutation_entropy(Jpk, m=min(5, len(Jpk)-1))
        # 顺序验证
        if _check_cyclic_order(labels, 2):
            label = 'P2'
        else:
            label = 'chaos' if pe > 0.70 else 'P1'
        if verbose:
            print(f"  Amplitude-P2 detected: ac1(J)={ac1_J:.4f}, k=2, PE={pe:.3f}")
        return dict(label=label, T_mean=T.mean(), T_ratio=T_ratio,
                    n_peaks=n_peaks, pe=pe, confidence=confidence)

    k, labels, centers = k_T, labels_T, centers_T

    # ── Step 4: 周期顺序验证（k≥2）────────────────────────────────────
    if k >= 2:
        if not _check_cyclic_order(labels, k):
            # 顺序不规则 → chaos 候选，用排列熵确认
            pe = _permutation_entropy(T, m=min(5, len(T)-1))
            if verbose:
                print(f"  k={k} 但顺序不规则，PE={pe:.3f}")
            label = 'chaos' if pe > 0.70 else 'P1'
            T_sorted = np.sort(centers)
            T_ratio = T_sorted[-1] / T_sorted[0] if T_sorted[0] > 0 else np.nan
            return dict(label=label, T_mean=T.mean(), T_ratio=T_ratio,
                        n_peaks=n_peaks, pe=pe, confidence=confidence)

    # ── Step 5: 计算汇总统计 ──────────────────────────────────────────
    pe = _permutation_entropy(T, m=min(5, len(T)-1))

    if k == 1:
        label = 'P1'
        T_ratio = 1.0
    else:
        label = f'P{k}'
        T_sorted = np.sort(centers)
        T_ratio = T_sorted[-1] / T_sorted[0] if T_sorted[0] > 0 else np.nan

    # 高排列熵覆盖：即使 k=1 也可能是宽幅乱序
    if pe > 0.85:
        label = 'chaos'
        T_ratio = np.nan

    if verbose:
        T_ratio_str = f"{T_ratio:.2f}" if not np.isnan(T_ratio) else 'nan'
        print(f"  k={k}, centers={np.sort(centers).round(1)}, PE={pe:.3f}, T_ratio={T_ratio_str}")

    return dict(label=label, T_mean=T.mean(), T_ratio=T_ratio,
                n_peaks=n_peaks, pe=pe, confidence=confidence)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化诊断图
# ──────────────────────────────────────────────────────────────────────────────

def plot_diagnosis(npz_paths, out_path=None, tail_frac=0.50):
    """
    为若干文件生成诊断图：每行一个文件，4列（时序、间隔序列、间隔直方图、返回映射）。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(npz_paths)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n), facecolor='white')
    if n == 1:
        axes = axes[None, :]

    for row, path in enumerate(npz_paths):
        d = np.load(path, allow_pickle=True)
        J = d['J']; t = d['t']
        N = J.shape[0]
        cut = int(len(t) * (1 - tail_frac))
        J_xL = J[N-1, cut:]; t_xL = t[cut:]

        peaks, _ = find_peaks(J_xL, prominence=0.005, height=0.5)
        T = np.diff(t_xL[peaks]) if len(peaks) > 1 else np.array([])
        res = classify(path, tail_frac=tail_frac)
        fname = Path(path).stem

        # Colors for alternating intervals
        if len(T) >= 2 and res['label'] not in ('steady', 'P1'):
            k, labels, centers = _best_period(T, max_k=4)
            cmap = plt.cm.get_cmap('Set1', max(k, 2))
            peak_colors = [cmap(labels[i]) if i < len(labels) else 'gray' for i in range(len(peaks)-1)]
        else:
            peak_colors = ['#2471a3'] * max(len(peaks)-1, 0)

        # Panel 1: 时序
        ax = axes[row, 0]
        ax.plot(t_xL, J_xL, 'k-', lw=0.8, alpha=0.8)
        if len(peaks) > 0:
            ax.scatter(t_xL[peaks], J_xL[peaks], c='#c0392b', s=30, zorder=5)
        ax.set_xlabel('τ'); ax.set_ylabel('J(x=L)')
        ax.set_title(f'{fname}\n→ {res["label"]}  T={res["T_mean"]:.1f}τ', fontsize=9)

        # Panel 2: 间隔序列
        ax = axes[row, 1]
        if len(T) > 0:
            ax.plot(range(len(T)), T, 'o-', lw=1.2, ms=5, color='#2471a3')
            for i, (ti, c) in enumerate(zip(T, peak_colors)):
                ax.scatter(i, ti, color=c, s=50, zorder=5)
            ax.axhline(T.mean(), color='gray', ls='--', lw=1)
        ax.set_xlabel('Peak index'); ax.set_ylabel('Interval T (τ)')
        ax.set_title(f'Interval sequence  ratio={res["T_ratio"]:.2f}' if not np.isnan(res.get("T_ratio", np.nan)) else 'Interval sequence', fontsize=9)

        # Panel 3: 间隔直方图
        ax = axes[row, 2]
        if len(T) > 2:
            ax.hist(T, bins=max(5, len(T)//3), color='#2471a3', edgecolor='white', alpha=0.8)
        ax.set_xlabel('T (τ)'); ax.set_ylabel('Count')
        ax.set_title(f'Interval histogram  PE={res["pe"]:.3f}', fontsize=9)

        # Panel 4: 返回映射
        ax = axes[row, 3]
        if len(T) > 2:
            ax.scatter(T[:-1], T[1:], c=range(len(T)-1), cmap='viridis', s=40, zorder=5)
            lim = (T.min()*0.95, T.max()*1.05)
            ax.plot(lim, lim, 'k--', lw=1, alpha=0.5)
            ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('T_n (τ)'); ax.set_ylabel('T_{n+1} (τ)')
        ax.set_title(f'Return map  conf={res["confidence"]}', fontsize=9)

    plt.tight_layout()
    if out_path is None:
        out_path = ROOT / 'Figure' / 'complex' / 'classification_demo.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"Saved {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

def _batch_classify(paths, show_all=False):
    """批量分类并打印结果表格。"""
    results = []
    for p in sorted(paths):
        try:
            r = classify(p)
            results.append((Path(p).stem, r))
        except Exception as e:
            results.append((Path(p).stem, {'label': f'ERROR:{e}', 'T_mean': np.nan,
                                            'T_ratio': np.nan, 'n_peaks': 0,
                                            'pe': np.nan, 'confidence': ''}))

    # 打印表格
    hdr = f"{'File':<50} {'Label':<8} {'T_mean':>8} {'T_ratio':>8} {'n_peaks':>8} {'PE':>6} {'Conf':<6}"
    print(hdr)
    print('-' * len(hdr))
    counts = {}
    for stem, r in results:
        lbl = r['label']
        counts[lbl] = counts.get(lbl, 0) + 1
        if show_all or lbl not in ('P1', 'steady'):
            Tm = f"{r['T_mean']:.1f}" if not np.isnan(r['T_mean']) else '  —'
            Tr = f"{r['T_ratio']:.2f}" if not np.isnan(r['T_ratio']) else '  —'
            pe = f"{r['pe']:.3f}" if not np.isnan(r['pe']) else '  —'
            print(f"{stem:<50} {lbl:<8} {Tm:>8} {Tr:>8} {r['n_peaks']:>8} {pe:>6} {r['confidence']:<6}")

    print(f"\n{'Label counts:'}")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:<10}: {cnt}")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Classify gel oscillation type from .npz files')
    parser.add_argument('files', nargs='*', help='.npz file paths')
    parser.add_argument('--all', action='store_true', help='Test all known datasets')
    parser.add_argument('--demo', action='store_true', help='Generate demo visualization')
    parser.add_argument('--show-all', action='store_true', help='Show P1/steady rows too')
    parser.add_argument('--tail', type=float, default=0.50, help='Tail fraction (default 0.50)')
    args = parser.parse_args()

    if args.all or not args.files:
        # 收集所有已知数据
        import glob
        paths = (glob.glob(str(ROOT / 'Data' / 'scan_Da*_N40_t2000.npz')) +
                 glob.glob(str(ROOT / 'Data' / 'scan_Bic*_N40_t1500.npz')) +
                 glob.glob(str(ROOT / 'Data' / 'complex' / 'p2_search' / '*.npz')))
    else:
        paths = args.files

    results = _batch_classify(paths, show_all=args.show_all)

    if args.demo:
        # 为各类型各选一个代表性样本做可视化
        demo_map = {}
        for stem, r in results:
            lbl = r['label']
            if lbl not in demo_map and r['confidence'] == 'high':
                demo_map[lbl] = stem
        demo_paths = []
        for lbl in ['steady', 'P1', 'P2', 'chaos']:
            if lbl in demo_map:
                # 找回原路径
                for p in paths:
                    if Path(p).stem == demo_map[lbl]:
                        demo_paths.append(p)
                        break
        if demo_paths:
            plot_diagnosis(demo_paths)
        else:
            print("No demo samples found (need high-confidence examples of each type).")
