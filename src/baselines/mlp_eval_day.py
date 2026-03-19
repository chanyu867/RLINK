#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def parse_seed_day_from_name(fname: str) -> tuple[int | None, int | None]:
    """
    Try to parse seed/day from filename like:
      mlp_random_b0.33_0.66_hs128-64_seed8_Nclasses3_day99_test_perday_valacc.npz
    """
    base = os.path.basename(fname)
    m_seed = re.search(r"seed(\d+)", base)
    m_day = re.search(r"_day(\d+)_", base)
    seed = int(m_seed.group(1)) if m_seed else None
    day = int(m_day.group(1)) if m_day else None
    return seed, day


def load_one_npz(path: str) -> tuple[int | None, int | None, float | None, np.ndarray | None, np.ndarray | None]:
    """
    Returns: seed, day, score (F1/bAcc/Acc), y_true, y_pred
    """
    seed_f, day_f = parse_seed_day_from_name(path)

    try:
        z = np.load(path, allow_pickle=False)
        
        seed = int(np.asarray(z["seed"]).item()) if "seed" in z.files else seed_f
        day = int(np.asarray(z["day_value"]).item()) if "day_value" in z.files else day_f
        
        # Prioritize F1-Macro, fallback to Balanced Acc, fallback to regular Acc
        score = None
        if "f1" in z.files:
            val = float(np.asarray(z["f1"]).item())
            if np.isfinite(val): score = val
        elif "bacc" in z.files and score is None:
            val = float(np.asarray(z["bacc"]).item())
            if np.isfinite(val): score = val
        elif "acc" in z.files and score is None:
            val = float(np.asarray(z["acc"]).item())
            if np.isfinite(val): score = val

        # Extract true and predicted labels for per-class metrics
        y_true = z["y_true"] if "y_true" in z.files else None
        y_pred = z["y_pred"] if "y_pred" in z.files else None
        
        return seed, day, score, y_true, y_pred

    except Exception as e:
        print(f"[WARN] failed to read npz content: {path} ({type(e).__name__}: {e})")
        return seed_f, day_f, None, None, None


def build_matrices(records: list[tuple[int, int, float, dict]], class_ids: list[int]):
    """
    Builds the matrices for overall metrics and per-class metrics.
    """
    seeds = sorted({r[0] for r in records})
    days = sorted({r[1] for r in records})

    seed_to_j = {s: j for j, s in enumerate(seeds)}
    day_to_i = {d: i for i, d in enumerate(days)}

    # mat: Overall score matrix
    mat = np.full((len(days), len(seeds)), np.nan, dtype=float)
    # mat_pc: Per-class score matrix
    mat_pc = np.full((len(days), len(seeds), len(class_ids)), np.nan, dtype=float)

    class_to_k = {c: k for k, c in enumerate(class_ids)}

    for s, d, score, pc_acc in records:
        i, j = day_to_i[d], seed_to_j[s]
        mat[i, j] = score
        for c, acc in pc_acc.items():
            if c in class_to_k:
                mat_pc[i, j, class_to_k[c]] = float(acc)

    return np.asarray(days, dtype=int), np.asarray(seeds, dtype=int), mat, mat_pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help=".../results directory")
    ap.add_argument("--glob_pat", type=str, default="*_outputs_day*.npz")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="mlp_daywise", help="prefix for saved files")
    ap.add_argument("--min_seeds", type=int, default=1, help="minimum #seeds present to keep a day")
    ap.add_argument("--max_day", type=int, default=100, help="Plot only days <= max_day (default=100)")
    ap.add_argument("--x_tick_step", type=int, default=3, help="x-axis tick step in days (default=3)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.glob_pat)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.results_dir, args.glob_pat)}")

    # (1) Load all files and compute per-class accuracies
    records = []
    all_classes = set()

    for p in paths:
        seed, day, score, yt, yp = load_one_npz(p)
        if seed is None or day is None or score is None:
            continue
            
        pc_acc = {}
        if yt is not None and yp is not None:
            classes_in_file = np.unique(yt)
            all_classes.update(classes_in_file.tolist())
            for c in classes_in_file:
                mask = (yt == c)
                if np.sum(mask) > 0:
                    pc_acc[int(c)] = float(np.sum(yp[mask] == yt[mask]) / np.sum(mask))
                    
        records.append((int(seed), int(day), float(score), pc_acc))

    if len(records) == 0:
        raise RuntimeError("No valid (seed, day, score) records were loaded.")

    class_ids = sorted(list(all_classes))
    days, seeds, mat, mat_pc = build_matrices(records, class_ids)

    # (2) Mean/std across seeds (ignore missing)
    seed_count = np.sum(~np.isnan(mat), axis=1)
    keep = seed_count >= int(args.min_seeds)

    days_k = days[keep]
    mat_k = mat[keep]
    mat_pc_k = mat_pc[keep]
    
    mean_acc = np.nanmean(mat_k, axis=1)
    std_acc = np.nanstd(mat_k, axis=1)

    # Apply max_day filter
    if args.max_day is not None:
        keep2 = days_k <= int(args.max_day)
        days_k = days_k[keep2]
        mean_acc = mean_acc[keep2]
        std_acc = std_acc[keep2]
        mat_pc_k = mat_pc_k[keep2]
        
    mean_pc = np.nanmean(mat_pc_k, axis=1)
    std_pc = np.nanstd(mat_pc_k, axis=1)

    # (3) Slope (linear fit of overall mean vs day)
    if len(days_k) >= 2:
        slope, intercept = np.polyfit(days_k.astype(float), mean_acc.astype(float), 1)
        fit_line = slope * days_k + intercept
    else:
        slope, intercept = np.nan, np.nan
        fit_line = np.full_like(days_k, np.nan, dtype=float)

    print(f"[Loaded] seeds={seeds.tolist()}")
    print(f"[Loaded] days={days.tolist()}  (kept={days_k.tolist()}, min_seeds={args.min_seeds})")
    print(f"[Classes] Found classes: {class_ids}")
    print(f"[Slope] mean metric ~ slope*day + intercept: slope={slope:.6g}, intercept={intercept:.6g}")

    # Calculate x-axis ticks
    # x_min = int(np.min(days_k))
    # x_max = int(np.max(days_k))
    # xt = list(np.arange(x_min, x_max + 1, int(args.x_tick_step), dtype=int))
    xt = days_k.tolist()

    # =========================================================
    # (4) Plot 1: Overall Metric (F1/bAcc/Acc)
    # =========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(days_k, mean_acc, marker="o")
    plt.fill_between(days_k, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    plt.plot(days_k, fit_line, color="red", linewidth=2, label=f"slope={slope:.3g}")

    plt.xlabel("Day")
    plt.ylabel("Test Metric (F1/bAcc)")
    plt.ylim(0, 1)
    plt.title(f"{args.tag}: Overall Metric mean±std across seeds")
    plt.legend()
    plt.xticks(xt, [str(v) for v in xt], rotation=45, ha="right")
    plt.tight_layout()

    out_png = os.path.join(args.out_dir, f"{args.tag}_overall_mean_std.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)

    # =========================================================
    # (5) Plot 2: Per-Class Accuracy
    # =========================================================
    if len(class_ids) > 0:
        plt.figure(figsize=(10, 6))
        for k, c in enumerate(class_ids):
            m = mean_pc[:, k]
            s = std_pc[:, k]
            plt.plot(days_k, m, marker="o", label=f"Class {c}")
            plt.fill_between(days_k, m - s, m + s, alpha=0.15)

        plt.xlabel("Day")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        plt.title(f"{args.tag}: Per-Class Accuracy mean±std")
        plt.legend(loc="best", title="Classes")
        plt.xticks(xt, [str(v) for v in xt], rotation=45, ha="right")
        plt.tight_layout()

        out_png_pc = os.path.join(args.out_dir, f"{args.tag}_perclass_mean_std.png")
        plt.savefig(out_png_pc, dpi=200)
        plt.close()
        print("[Saved]", out_png_pc)


if __name__ == "__main__":
    main()