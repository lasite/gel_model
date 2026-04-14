#!/usr/bin/env /home/wang/venvs/jaxenv/bin/python
"""
run_2D_map.py — Parallel Da × Bi_c scan for N=40 complex-oscillation window.

Default scan:
  Da   = 8.0 ... 15.0   (step 0.5)
  Bi_c = 0.50 ... 1.00  (step 0.05)
  N    = 40
  t_end = 1500

What this script records:
  1. Per-point simulation files:
       Data/complex/map_2D/map_Da{Da}_Bic{Bi_c}_N{N}.npz
     containing t, J, u, theta, metadata.
  2. Scan summary table:
       Data/complex/map_2D/scan_summary_N{N}.csv
  3. Matrix bundle for plotting/post-processing:
       Data/complex/map_2D/scan_summary_N{N}.npz
  4. Publication-style overview figure:
       Figure/complex/phase_map_Da_Bic_N{N}.png

Classification uses Code/complex/classify_oscillation.py so the scan output
can be compared consistently with later P1 / P2 / chaos analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "complex"))

DATA_DIR = ROOT / "Data" / "complex" / "map_2D"
FIG_DIR = ROOT / "Figure" / "complex"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from classify_oscillation import classify
from scan_optimized import Params, simulate

DEFAULTS = dict(
    Gamma_A=1.5,
    m_diff=2.0,
    m_act=6.0,
    D0=2.0,
    Bi_T=0.10,
    alpha=0.20,
    S_chi=1.0,
)

LABEL_COLOR = {
    "steady": "#9E9E9E",
    "P1": "#2196F3",
    "P2": "#4CAF50",
    "P3": "#8BC34A",
    "P4": "#CDDC39",
    "chaos": "#F44336",
    "?": "#212121",
}


def point_path(da: float, bc: float, n: int) -> Path:
    return DATA_DIR / f"map_Da{da:.1f}_Bic{bc:.2f}_N{n}.npz"


def run_one(args: tuple[float, float, int, float, int, bool]) -> dict:
    da, bc, n, t_end, n_save, overwrite = args
    out = point_path(da, bc, n)
    started = time.time()

    if out.exists() and not overwrite:
        d = np.load(out, allow_pickle=True)
        return {
            "Da": da,
            "Bi_c": bc,
            "path": str(out),
            "elapsed_s": float(d.get("elapsed_s", np.array(np.nan))),
            "nfev": int(d.get("nfev", np.array(-1))),
            "from_cache": True,
        }

    p = Params(**DEFAULTS, N=n, t_end=t_end, n_save=n_save, Da=float(da), Bi_c=float(bc))
    r = simulate(p)
    elapsed = time.time() - started
    np.savez_compressed(
        out,
        t=r["t"],
        J=r["J"],
        u=r["u"],
        theta=r["theta"],
        Da=np.array([da]),
        Bi_c=np.array([bc]),
        N=np.array([n]),
        t_end=np.array([t_end]),
        n_save=np.array([n_save]),
        elapsed_s=np.array([elapsed]),
        nfev=np.array([r["nfev"]]),
    )
    return {
        "Da": da,
        "Bi_c": bc,
        "path": str(out),
        "elapsed_s": elapsed,
        "nfev": int(r["nfev"]),
        "from_cache": False,
    }


def classify_point(path: str) -> dict:
    res = classify(path, tail_frac=0.50, min_peaks=4, verbose=False)
    return {
        "label": res["label"],
        "T_mean": float(res["T_mean"]) if np.isfinite(res["T_mean"]) else np.nan,
        "T_ratio": float(res["T_ratio"]) if np.isfinite(res["T_ratio"]) else np.nan,
        "n_peaks": int(res["n_peaks"]),
        "pe": float(res["pe"]) if np.isfinite(res["pe"]) else np.nan,
        "confidence": res["confidence"],
    }


def save_summary(rows: list[dict], das: np.ndarray, bics: np.ndarray, n: int) -> None:
    csv_path = DATA_DIR / f"scan_summary_N{n}.csv"
    npz_path = DATA_DIR / f"scan_summary_N{n}.npz"

    fieldnames = [
        "Da", "Bi_c", "label", "T_mean", "T_ratio", "n_peaks", "pe",
        "confidence", "elapsed_s", "nfev", "path", "from_cache",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_names = ["steady", "P1", "P2", "P3", "P4", "chaos", "?"]
    label_to_code = {name: i for i, name in enumerate(label_names)}
    label_mat = np.full((len(das), len(bics)), label_to_code["?"], dtype=int)
    period_mat = np.full((len(das), len(bics)), np.nan)
    ratio_mat = np.full((len(das), len(bics)), np.nan)
    peaks_mat = np.full((len(das), len(bics)), np.nan)
    pe_mat = np.full((len(das), len(bics)), np.nan)
    elapsed_mat = np.full((len(das), len(bics)), np.nan)

    for row in rows:
        i = int(np.where(np.isclose(das, row["Da"]))[0][0])
        j = int(np.where(np.isclose(bics, row["Bi_c"]))[0][0])
        label_mat[i, j] = label_to_code.get(row["label"], label_to_code["?"])
        period_mat[i, j] = row["T_mean"]
        ratio_mat[i, j] = row["T_ratio"]
        peaks_mat[i, j] = row["n_peaks"]
        pe_mat[i, j] = row["pe"]
        elapsed_mat[i, j] = row["elapsed_s"]

    np.savez_compressed(
        npz_path,
        Das=das,
        Bics=bics,
        label_names=np.array(label_names, dtype=object),
        label_mat=label_mat,
        period_mat=period_mat,
        ratio_mat=ratio_mat,
        peaks_mat=peaks_mat,
        pe_mat=pe_mat,
        elapsed_mat=elapsed_mat,
    )
    print(f"Saved summary table: {csv_path}")
    print(f"Saved summary matrices: {npz_path}")


def plot_summary(rows: list[dict], das: np.ndarray, bics: np.ndarray, n: int) -> None:
    label_names = ["steady", "P1", "P2", "P3", "P4", "chaos", "?"]
    label_to_code = {name: i for i, name in enumerate(label_names)}

    label_mat = np.full((len(das), len(bics)), label_to_code["?"], dtype=int)
    pe_mat = np.full((len(das), len(bics)), np.nan)
    for row in rows:
        i = int(np.where(np.isclose(das, row["Da"]))[0][0])
        j = int(np.where(np.isclose(bics, row["Bi_c"]))[0][0])
        label_mat[i, j] = label_to_code.get(row["label"], label_to_code["?"])
        pe_mat[i, j] = row["pe"]

    cmap = mcolors.ListedColormap([LABEL_COLOR[name] for name in label_names])
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(label_names) + 0.5, 1), cmap.N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.38})

    ax1.imshow(
        label_mat.T,
        origin="lower",
        aspect="auto",
        extent=[das[0] - 0.25, das[-1] + 0.25, bics[0] - 0.025, bics[-1] + 0.025],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax1.set_xlabel("Da")
    ax1.set_ylabel("Bi_c")
    ax1.set_title("Oscillation class")
    ax1.legend(
        handles=[Patch(color=LABEL_COLOR[name], label=name) for name in label_names if name in {r["label"] for r in rows}],
        fontsize=8,
        loc="upper right",
    )

    im2 = ax2.imshow(
        pe_mat.T,
        origin="lower",
        aspect="auto",
        extent=[das[0] - 0.25, das[-1] + 0.25, bics[0] - 0.025, bics[-1] + 0.025],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.colorbar(im2, ax=ax2, label="Permutation entropy")
    ax2.set_xlabel("Da")
    ax2.set_ylabel("Bi_c")
    ax2.set_title("Irregularity map")

    fig.suptitle(f"Da × Bi_c scan  (N={n}, Bi_T=0.10, Gamma_A=1.5, D0=2.0)", fontsize=12)
    out = FIG_DIR / f"phase_map_Da_Bic_N{n}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


def main(
    workers: int = 8,
    da_min: float = 8.0,
    da_max: float = 15.0,
    da_step: float = 0.5,
    bc_min: float = 0.50,
    bc_max: float = 1.00,
    bc_step: float = 0.05,
    n: int = 40,
    t_end: float = 1500.0,
    n_save: int = 7500,
    overwrite: bool = False,
) -> None:
    das = np.round(np.arange(da_min, da_max + 0.5 * da_step, da_step), 6)
    bics = np.round(np.arange(bc_min, bc_max + 0.5 * bc_step, bc_step), 6)
    jobs = [(da, bc, n, t_end, n_save, overwrite) for da in das for bc in bics]

    print(
        f"Running {len(jobs)} simulations "
        f"({len(das)} Da × {len(bics)} Bi_c), N={n}, t_end={t_end}, workers={workers}"
    )
    started = time.time()
    rows = []
    with Pool(processes=workers) as pool:
        for idx, meta in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            cls = classify_point(meta["path"])
            row = {**meta, **cls}
            rows.append(row)
            dt = time.time() - started
            rate = idx / max(dt, 1e-9)
            eta = (len(jobs) - idx) / max(rate, 1e-9)
            print(
                f"[{idx:3d}/{len(jobs)}] Da={row['Da']:.1f} Bi_c={row['Bi_c']:.2f} "
                f"{row['label']:<6} peaks={row['n_peaks']:<3d} "
                f"elapsed={row['elapsed_s']:.1f}s ETA={eta/60:.1f}min"
            )

    rows.sort(key=lambda r: (r["Da"], r["Bi_c"]))
    save_summary(rows, das, bics, n)
    plot_summary(rows, das, bics, n)
    print(f"Total wall time: {(time.time() - started)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--Da_min", type=float, default=8.0)
    ap.add_argument("--Da_max", type=float, default=15.0)
    ap.add_argument("--Da_step", type=float, default=0.5)
    ap.add_argument("--Bc_min", type=float, default=0.50)
    ap.add_argument("--Bc_max", type=float, default=1.00)
    ap.add_argument("--Bc_step", type=float, default=0.05)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--t_end", type=float, default=1500.0)
    ap.add_argument("--n_save", type=int, default=7500)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(
        workers=args.workers,
        da_min=args.Da_min,
        da_max=args.Da_max,
        da_step=args.Da_step,
        bc_min=args.Bc_min,
        bc_max=args.Bc_max,
        bc_step=args.Bc_step,
        n=args.N,
        t_end=args.t_end,
        n_save=args.n_save,
        overwrite=args.overwrite,
    )
