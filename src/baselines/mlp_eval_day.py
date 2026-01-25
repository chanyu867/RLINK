#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

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


def load_one_npz(path: str) -> tuple[int | None, int | None, float | None, int | None]:
    """
    Returns: seed, day, acc, n
    Prefers reading from npz content; falls back to filename parsing.
    """
    seed_f, day_f = parse_seed_day_from_name(path)

    try:
        z = np.load(path, allow_pickle=False)
        seed = int(z["seed"]) if "seed" in z else seed_f

        # days/acc/n are arrays (often length 1)
        day_arr = z["days"] if "days" in z else np.array([day_f])
        acc_arr = z["acc"] if "acc" in z else None
        n_arr = z["n"] if "n" in z else None

        if day_arr is None or len(day_arr) == 0:
            day = day_f
        else:
            day = int(np.asarray(day_arr).ravel()[0])

        if acc_arr is None or len(acc_arr) == 0:
            acc = None
        else:
            acc = float(np.asarray(acc_arr).ravel()[0])

        n = int(np.asarray(n_arr).ravel()[0]) if (n_arr is not None and len(n_arr) > 0) else None
        return seed, day, acc, n

    except Exception as e:
        # if any issue, fallback to name-only
        print(f"[WARN] failed to read npz content: {path} ({e})")
        return seed_f, day_f, None, None


def build_matrix(records: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    records: list of (seed, day, acc)
    Returns:
      days_sorted (D,), seeds_sorted (S,), mat (D,S) with NaN for missing
    """
    seeds = sorted({s for s, d, a in records})
    days = sorted({d for s, d, a in records})

    seed_to_j = {s: j for j, s in enumerate(seeds)}
    day_to_i = {d: i for i, d in enumerate(days)}

    mat = np.full((len(days), len(seeds)), np.nan, dtype=float)
    for s, d, a in records:
        mat[day_to_i[d], seed_to_j[s]] = float(a)

    return np.asarray(days, dtype=int), np.asarray(seeds, dtype=int), mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help=".../results directory")
    ap.add_argument("--glob_pat", type=str, default="*_test_perday_valacc.npz")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="mlp_daywise", help="prefix for saved files")
    ap.add_argument("--min_seeds", type=int, default=1, help="minimum #seeds present to keep a day")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.glob_pat)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.results_dir, args.glob_pat)}")

    # (1) load all
    records = []
    for p in paths:
        seed, day, acc, n = load_one_npz(p)
        if seed is None or day is None or acc is None:
            continue
        records.append((int(seed), int(day), float(acc)))

    if len(records) == 0:
        raise RuntimeError("No valid (seed, day, acc) records were loaded.")

    days, seeds, mat = build_matrix(records)

    # (2) mean/std across seeds (ignore missing)
    seed_count = np.sum(~np.isnan(mat), axis=1)
    keep = seed_count >= int(args.min_seeds)

    days_k = days[keep]
    mat_k = mat[keep]
    mean_acc = np.nanmean(mat_k, axis=1)
    std_acc = np.nanstd(mat_k, axis=1)

    # (3) slope (linear fit of mean vs day)
    #     If days are not contiguous, it's still fine.
    if len(days_k) >= 2:
        slope, intercept = np.polyfit(days_k.astype(float), mean_acc.astype(float), 1)
        fit_line = slope * days_k + intercept
    else:
        slope, intercept = np.nan, np.nan
        fit_line = np.full_like(days_k, np.nan, dtype=float)

    print(f"[Loaded] seeds={seeds.tolist()}")
    print(f"[Loaded] days={days.tolist()}  (kept={days_k.tolist()}, min_seeds={args.min_seeds})")
    print(f"[Slope] mean_acc ~ slope*day + intercept: slope={slope:.6g}, intercept={intercept:.6g}")

    # (4) save aggregated arrays
    # out_npz = os.path.join(args.out_dir, f"{args.tag}_agg_mean_std.npz")
    # np.savez(
    #     out_npz,
    #     days=days_k.astype(int),
    #     seeds=seeds.astype(int),
    #     acc_matrix=mat_k.astype(float),
    #     mean_acc=mean_acc.astype(float),
    #     std_acc=std_acc.astype(float),
    #     slope=float(slope) if np.isfinite(slope) else np.nan,
    #     intercept=float(intercept) if np.isfinite(intercept) else np.nan,
    #     seed_count=seed_count[keep].astype(int),
    # )
    # print("[Saved]", out_npz)

    # (5) plot
    plt.figure()
    plt.plot(days_k, mean_acc, marker="o")  # use default color
    plt.fill_between(days_k, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)  # default color band

    # slope line in RED (as requested)
    if np.all(np.isfinite(fit_line)):
        plt.plot(days_k, fit_line, color="red", linewidth=2, label=f"slope={slope:.3g}")

    plt.xlabel("Day")
    plt.ylabel("Test accuracy")
    plt.ylim(0, 1)
    plt.title(f"{args.tag}: mean±std across seeds")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(args.out_dir, f"{args.tag}_mean_std_slope.png")
    plt.savefig(out_png, dpi=200)
    print("[Saved]", out_png)


if __name__ == "__main__":
    main()
