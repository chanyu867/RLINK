#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def parse_seed_day_from_name(fname: str) -> tuple[int | None, int | None]:
    base = os.path.basename(fname)
    m_seed = re.search(r"seed(\d+)", base)
    m_day = re.search(r"_day(\d+)_", base)
    seed = int(m_seed.group(1)) if m_seed else None
    day = int(m_day.group(1)) if m_day else None
    return seed, day


def load_one_npz(path: str) -> tuple[int | None, int | None, float | None, np.ndarray | None, np.ndarray | None]:
    seed_f, day_f = parse_seed_day_from_name(path)

    try:
        z = np.load(path, allow_pickle=False)
        
        seed = int(np.asarray(z["seed"]).item()) if "seed" in z.files else seed_f
        day = int(np.asarray(z["day_value"]).item()) if "day_value" in z.files else day_f
        
        # Priority: F1 > Balanced Accuracy > Standard Accuracy
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

        y_true = z["y_true"] if "y_true" in z.files else None
        y_pred = z["y_pred"] if "y_pred" in z.files else None
        
        return seed, day, score, y_true, y_pred

    except Exception as e:
        print(f"[WARN] failed to read npz content: {path} ({type(e).__name__}: {e})")
        return seed_f, day_f, None, None, None


def build_matrices(records: list[tuple[int, int, float, dict]], class_ids: list[int]):
    seeds = sorted({r[0] for r in records})
    days = sorted({r[1] for r in records})

    seed_to_j = {s: j for j, s in enumerate(seeds)}
    day_to_i = {d: i for i, d in enumerate(days)}

    mat = np.full((len(days), len(seeds)), np.nan, dtype=float)
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
    ap.add_argument("--fixed_npz", type=str, default=None, help="Path to aggregated _perday_meanstd.npz from fixed model")
    
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.glob_pat)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.results_dir, args.glob_pat)}")

    # (1) Load Day-wise data
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

    seed_count = np.sum(~np.isnan(mat), axis=1)
    keep = seed_count >= int(args.min_seeds)

    days_k = days[keep]
    mat_k = mat[keep]
    mat_pc_k = mat_pc[keep]
    
    mean_acc = np.nanmean(mat_k, axis=1)
    std_acc = np.nanstd(mat_k, axis=1)

    if args.max_day is not None:
        keep2 = days_k <= int(args.max_day)
        days_k = days_k[keep2]
        mean_acc = mean_acc[keep2]
        std_acc = std_acc[keep2]
        mat_pc_k = mat_pc_k[keep2]
        
    mean_pc = np.nanmean(mat_pc_k, axis=1)
    std_pc = np.nanstd(mat_pc_k, axis=1)

    # (2) Load Fixed Model data
    fixed_days, f_mean, f_std = None, None, None
    f_mean_pc, f_std_pc, f_class_ids = None, None, []
    
    if args.fixed_npz and os.path.exists(args.fixed_npz):
        fz = np.load(args.fixed_npz, allow_pickle=False)
        
        # Safe extraction of days
        if "days" in fz.files:
            fixed_days = fz["days"]
        elif "days_sorted" in fz.files:
            fixed_days = fz["days_sorted"]
        else:
            raise KeyError(f"CRITICAL: Could not find 'days' in {args.fixed_npz}")
            
        # Match metric priority
        if "mean_f1" in fz.files and not np.all(np.isnan(fz["mean_f1"])):
            f_mean, f_std = fz["mean_f1"], fz["std_f1"]
        elif "mean_bacc" in fz.files and not np.all(np.isnan(fz["mean_bacc"])):
            f_mean, f_std = fz["mean_bacc"], fz["std_bacc"]
        else:
            f_mean, f_std = fz["mean"], fz["std"]
            
        if "mean_pc" in fz.files:
            f_mean_pc, f_std_pc = fz["mean_pc"], fz["std_pc"]
            f_class_ids = fz["class_ids"] if "class_ids" in fz.files else []
            
        # Apply max_day filter to Fixed Model
        if args.max_day is not None and fixed_days is not None:
            fm_keep = fixed_days <= int(args.max_day)
            fixed_days = fixed_days[fm_keep]
            f_mean, f_std = f_mean[fm_keep], f_std[fm_keep]
            if f_mean_pc is not None:
                f_mean_pc, f_std_pc = f_mean_pc[fm_keep], f_std_pc[fm_keep]
        print(f"[Loaded Fixed Model Baseline] {args.fixed_npz}")

    # (3) Calculate Slopes
    # Slope for Day-wise
    if len(days_k) >= 2:
        slope, intercept = np.polyfit(days_k.astype(float), mean_acc.astype(float), 1)
        fit_line = slope * days_k + intercept
    else:
        slope, intercept, fit_line = np.nan, np.nan, np.full_like(days_k, np.nan, dtype=float)

    # Slope for Fixed Model
    f_slope, f_intercept, f_fit_line = np.nan, np.nan, None
    if fixed_days is not None and len(fixed_days) >= 2:
        valid = ~np.isnan(f_mean)
        if np.sum(valid) >= 2:
            f_slope, f_intercept = np.polyfit(fixed_days[valid].astype(float), f_mean[valid].astype(float), 1)
            f_fit_line = f_slope * fixed_days + f_intercept

    print(f"[Classes] Found classes: {class_ids}")
    xt = days_k.tolist()

    # =========================================================
    # (4) Plot 1: Overall Metric (with both slopes)
    # =========================================================
    plt.figure(figsize=(10, 6))
    
    # Plot Day-wise
    plt.plot(days_k, mean_acc, marker="o", color="blue", label="Day-wise Model")
    plt.fill_between(days_k, mean_acc - std_acc, mean_acc + std_acc, color="blue", alpha=0.2)
    if not np.isnan(slope):
        plt.plot(days_k, fit_line, color="red", linewidth=2, label=f"Day-wise slope={slope:.4g}/day")

    # Plot Fixed Baseline
    if fixed_days is not None:
        plt.plot(fixed_days, f_mean, marker="s", linestyle="--", color="orange", label="Fixed Model")
        plt.fill_between(fixed_days, f_mean - f_std, f_mean + f_std, color="orange", alpha=0.2)
        if f_fit_line is not None:
            plt.plot(fixed_days, f_fit_line, color="darkred", linestyle=":", linewidth=2, label=f"Fixed slope={f_slope:.4g}/day")

    plt.xlabel("Day")
    plt.ylabel("F1 score")
    plt.ylim(0, 1)
    plt.title(f"{args.tag}: Day-wise vs Fixed Model Overall Performance")
    plt.legend()
    plt.xticks(xt, [str(v) for v in xt], rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(args.out_dir, f"{args.tag}_overall_comparison.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)

    # =========================================================
    # (5) Plot 2: Per-Class Accuracy (2 Subplots)
    # =========================================================
    if len(class_ids) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        class_color_map = {}
        
        # Subplot 1: Day-wise Per Class
        for k, c in enumerate(class_ids):
            color = colors[k % len(colors)]
            class_color_map[c] = color
            m = mean_pc[:, k]
            s = std_pc[:, k]
            ax1.plot(days_k, m, marker="o", linestyle="-", color=color, label=f"Class {c}")
            ax1.fill_between(days_k, m - s, m + s, color=color, alpha=0.15)

        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.set_title("Day-wise Model (Retrained Daily)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

        # Subplot 2: Fixed Per Class
        if fixed_days is not None and f_mean_pc is not None:
            for k, c in enumerate(f_class_ids):
                color = class_color_map.get(c, "gray") 
                m = f_mean_pc[:, k]
                s = f_std_pc[:, k]
                ax2.plot(fixed_days, m, marker="s", linestyle="--", color=color, label=f"Class {c}")
                ax2.fill_between(fixed_days, m - s, m + s, color=color, alpha=0.10)

        ax2.set_xlabel("Day")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.set_title("Fixed Model (Trained on Day 1 Only)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

        plt.xticks(xt, [str(v) for v in xt], rotation=45, ha="right")
        fig.suptitle(f"{args.tag}: Per-Class Accuracy Comparison", fontsize=14)
        fig.tight_layout()

        out_png_pc = os.path.join(args.out_dir, f"{args.tag}_perclass_comparison.png")
        plt.savefig(out_png_pc, dpi=200)
        plt.close()
        print("[Saved]", out_png_pc)

if __name__ == "__main__":
    main()


"""
python /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/baselines/eval_summary.py \
    --results_dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/mlp_day/random/results \
    --glob_pat "mlp_random_b0.33_0.66_eliminate1_hs1024-64_seed0_Nclasses2_day*_outputs_day*.npz" \
    --out_dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/mlp_day/random/plots \
    --max_day 91 \
    --x_tick_step 3 \
    --fixed_npz /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/mlp_cv/random/results/evals/mlp_random_b0.33_0.66_eliminate1_hs1024-64-64-64_seeds0-0_future_slicing1_value10_perday_meanstd.npz
    

"""