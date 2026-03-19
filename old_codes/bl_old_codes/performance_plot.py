import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def _parse_npz_filename(fname: str):
    """
    Expected tail pattern (example):
    ..._day10_b0.2_0.4_0.6_0.8_seed0_perday_acc_Nclasses5_day10.npz

    Returns dict with: boundary(str), seed(int), slicing_day(int), n_classes(int or None)
    """
    base = os.path.basename(fname)

    m = re.search(
        r"_b(?P<boundary>\d+(?:\.\d+)?(?:_\d+(?:\.\d+)?)+)"   # b0.2_0.4_...
        r"_seed(?P<seed>\d+)"
        r"_perday_acc_Nclasses(?P<ncls>\d+)"
        r"_day(?P<sday>\d+)\.npz$",
        base
    )
    print("m: ", m)
    if not m:
        return None

    return {
        "boundary": m.group("boundary"),
        "seed": int(m.group("seed")),
        "slicing_day": int(m.group("sday")),
        "n_classes": int(m.group("ncls")),
    }


def _collect_seed_files(out_dir: str, target_type: str, boundary: str, slicing_day: int):
    """
    Collect per-seed perday npz files for a given (target_type, boundary, slicing_day).
    Files are expected under:
      out_dir/<target_type>/*.npz
    and filtered by filename fields (boundary, slicing_day).
    """
    pattern = os.path.join(out_dir, target_type, "*.npz")
    files = glob.glob(pattern)

    matched = []
    for f in files:
        info = _parse_npz_filename(f)
        if info is None:
            continue
        if info["boundary"] != boundary:
            continue
        if info["slicing_day"] != slicing_day:
            continue
        matched.append((f, info["seed"], info["n_classes"]))

    # sort by seed for reproducibility
    matched.sort(key=lambda x: x[1])
    return matched


def _load_day_acc_matrix(seed_files):
    """
    seed_files: list of (filepath, seed, n_classes)

    Returns:
      days_sorted (int array)
      A (float matrix) shape (n_seeds, n_days) with NaN for missing days
      seeds (int array)
      n_classes (int) inferred (assumes consistent)
    """
    if len(seed_files) == 0:
        return None, None, None, None

    # load all seed dicts {day -> acc}
    seed_to_map = {}
    all_days = set()
    n_classes = seed_files[0][2]

    for path, seed, ncls in seed_files:
        data = np.load(path, allow_pickle=False)
        days = data["days"].astype(int)
        acc = data["acc"].astype(float)
        seed_to_map[seed] = {int(d): float(a) for d, a in zip(days, acc)}
        all_days.update(days.tolist())

        # (optional) sanity: n_classes consistent
        if ncls != n_classes:
            # keep first, but you can raise if you want strictness
            pass

    days_sorted = np.array(sorted(all_days), dtype=int)
    seeds = np.array(sorted(seed_to_map.keys()), dtype=int)

    A = np.full((len(seeds), len(days_sorted)), np.nan, dtype=float)
    for si, seed in enumerate(seeds):
        d2a = seed_to_map[seed]
        for di, d in enumerate(days_sorted):
            if int(d) in d2a:
                A[si, di] = d2a[int(d)]

    return days_sorted, A, seeds, n_classes


def plot_perday_meanstd_for_slicingdays(
    out_dir: str,
    boundaries=("0.33_0.66", "0.25_0.5_0.75", "0.2_0.4_0.6_0.8"),
    target_types=("center-out", "random"),
    slicing_days=(10, 30, 50, 90, 150),
    save_dir: str | None = None,
):

    for target_type in target_types:
        
        save_dir = os.path.join("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/idx_position/plots", target_type)
        
        for boundary in boundaries:
            plt.figure()

            any_plotted = False
            used_nclasses = None

            for sday in slicing_days:
                seed_files = _collect_seed_files(out_dir, target_type, boundary, sday)
                if len(seed_files) == 0:
                    continue

                days_sorted, A, seeds, n_classes = _load_day_acc_matrix(seed_files)
                if days_sorted is None:
                    continue

                mean_acc = np.nanmean(A, axis=0)
                std_acc = np.nanstd(A, axis=0)

                # line + std band (no manual colors; matplotlib cycle handles it)
                plt.plot(days_sorted, mean_acc, marker="o", label=f"slicing_day={sday} (nSeeds={len(seeds)})")
                plt.fill_between(days_sorted, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)

                any_plotted = True
                used_nclasses = n_classes if used_nclasses is None else used_nclasses

            if not any_plotted:
                plt.close()
                continue

            plt.xlabel("Test day")
            plt.ylabel("Accuracy (mean ± std over seeds)")
            plt.ylim(0, 1)
            title = f"{target_type} | boundary={boundary} | Nclasses={used_nclasses}"
            plt.title(title)
            plt.legend()
            plt.tight_layout()

            out_png = os.path.join(
                save_dir,
                f"perday_meanstd_{target_type}_b{boundary}_Nclasses{used_nclasses}.png"
            )
            plt.savefig(out_png, dpi=200)
            plt.close()

            print("saved:", out_png)


# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    out_dir = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/idx_position"

    boundaries = ("0.33_0.66", "0.25_0.5_0.75", "0.2_0.4_0.6_0.8")
    target_types = ("center-out", "random")
    slicing_days = (10, 30, 50, 90, 150)

    plot_perday_meanstd_for_slicingdays(
        out_dir=out_dir,
        boundaries=boundaries,
        target_types=target_types,
        slicing_days=slicing_days,
        save_dir=os.path.join(out_dir, "plots"),
    )
