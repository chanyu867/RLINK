import os
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------- small utils ----------
def npy_loader(path: str):
    return np.load(path, allow_pickle=False)


def parse_label_mask(s: str | None):
    """
    Examples:
      None / ""  -> None
      "0,1,2"    -> [0,1,2]
      "0-2,5,7"  -> [0,1,2,5,7]
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None

    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            out.extend(list(range(min(a, b), max(a, b) + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def apply_masks(labels, day_info, target_style, target_type, label_mask_allowed):
    # 1) label_mask
    if label_mask_allowed is not None:
        keep = np.isin(labels.astype(int), label_mask_allowed)
        labels = labels[keep]
        day_info = day_info[keep]
        if target_style is not None:
            target_style = target_style[keep]

    # 2) target_type masking (center-out => ~t, random => t)
    if target_type is not None:
        if target_style is None:
            raise ValueError("--target_type was provided but --target_style_path is None.")
        t = np.asarray(target_style).astype(bool)
        if target_type == "center-out":
            keep = ~t
        elif target_type == "random":
            keep = t
        else:
            raise ValueError("--target_type must be one of: center-out, random")
        labels = labels[keep]
        day_info = day_info[keep]
        target_style = t[keep]

    return labels, day_info, target_style


# ---------- file collection ----------
def _parse_seed_file(fname: str):
    """
    Matches your real filenames, e.g.
      mlp_idx_position_center-out_day50_b0.25_0.5_0.75_seed1_perday_acc_Nclasses4_day1.npz

    We treat the slicing_day as the one right after target_type: "_day50_".
    The final "_day1" is ignored (often just a run/day tag).
    """
    base = os.path.basename(fname)
    m = re.search(
        r"_(?P<tt>center-out|random)_day\d+"  # ignore the first day
        r"_b(?P<boundary>\d+(?:\.\d+)?(?:_\d+(?:\.\d+)?)+)"
        r"_seed(?P<seed>\d+)_perday_acc_Nclasses(?P<ncls>\d+)_day(?P<slicing>\d+)\.npz$",
        base,
    )
    if not m:
        return None
    return {
        "target_type": m.group("tt"),
        "slicing_day": int(m.group("slicing")),
        "boundary": m.group("boundary"),
        "seed": int(m.group("seed")),
        "n_classes": int(m.group("ncls")),
    }


def _collect_seed_npz(out_dir: str, target_type: str, boundary: str, slicing_day: int):
    folder = os.path.join(out_dir, target_type)
    files = glob.glob(os.path.join(folder, "*.npz"))

    matched = []
    for f in files:
        info = _parse_seed_file(f)
        # print("info: ", info)
        if info is None:
            continue
        # if info["target_type"] != target_type:
        #     continue
        if info["boundary"] != boundary:
            continue
        if info["slicing_day"] != slicing_day:
            continue
        matched.append((f, info["seed"], info["n_classes"]))

    matched.sort(key=lambda x: x[1])  # by seed
    return matched


def _correct01_path_from_npz(npz_path: str):
    # swap "..._seedX_perday_acc_..." -> "..._seedX_test_correct01_..."
    return npz_path.replace("_perday_acc_", "_test_correct01_").replace(".npz", ".npy")


# ---------- loaders ----------
def _load_overall_perday(seed_files):
    """
    From each per-seed npz, load:
      days: (n_days,)
      acc : (n_days,)  (or sometimes (1,n_days))
    Returns:
      days_sorted (n_days,)
      A_overall   (n_seeds, n_days) with NaN for missing
      seeds       (n_seeds,)
      n_classes   (int)
    """
    if not seed_files:
        return None, None, None, None

    all_days = set()
    per_seed_data = []
    seeds = []
    n_classes = seed_files[0][2]

    for path, seed, ncls in seed_files:
        d = np.load(path, allow_pickle=False)
        if "days" not in d.files or "acc" not in d.files:
            raise KeyError(f"Expected keys 'days' and 'acc' in {path}. keys={d.files}")
        days = d["days"].astype(int)
        acc = d["acc"].astype(float)
        if acc.ndim == 2 and acc.shape[0] == 1:
            acc = acc[0]
        acc = acc.reshape(-1)

        all_days.update(days.tolist())
        seeds.append(int(seed))
        per_seed_data.append((int(seed), days, acc))

    days_sorted = np.array(sorted(all_days), dtype=int)
    seeds = np.array(sorted(seeds), dtype=int)
    seed_to_row = {s: i for i, s in enumerate(seeds)}
    day_to_col = {int(day): j for j, day in enumerate(days_sorted)}

    A_overall = np.full((len(seeds), len(days_sorted)), np.nan, dtype=float)
    for seed, days, acc in per_seed_data:
        si = seed_to_row[seed]
        for i, day in enumerate(days):
            A_overall[si, day_to_col[int(day)]] = float(acc[i])

    return days_sorted, A_overall, seeds, n_classes


def _load_perlabel_from_correct01(
    seed_files,
    y_test: np.ndarray,
    day_test: np.ndarray,
):
    """
    Build per-label per-day accuracy using:
      correct01 (per test sample, aligned with y_test/day_test ordering)
    Returns:
      days_sorted (n_days,)
      class_ids  (K,)  original label ids
      A_perlbl   (n_seeds, n_days, K)
      seeds      (n_seeds,)
    """
    class_ids = np.array(sorted(np.unique(y_test).astype(int).tolist()), dtype=int)
    K = len(class_ids)

    days_sorted = np.array(sorted(np.unique(day_test).astype(int).tolist()), dtype=int)

    seeds = np.array(sorted([int(s) for _, s, _ in seed_files]), dtype=int)
    seed_to_row = {s: i for i, s in enumerate(seeds)}
    day_to_col = {int(day): j for j, day in enumerate(days_sorted)}
    cls_to_k = {int(c): k for k, c in enumerate(class_ids)}

    A_perlbl = np.full((len(seeds), len(days_sorted), K), np.nan, dtype=float)

    for npz_path, seed, _ in seed_files:
        si = seed_to_row[int(seed)]
        cpath = _correct01_path_from_npz(npz_path)
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"Missing correct01 file: {cpath}")

        correct01 = np.load(cpath, allow_pickle=False).astype(float).reshape(-1)
        if correct01.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Length mismatch for seed{seed}:\n"
                f"  correct01: {correct01.shape[0]}\n"
                f"  y_test   : {y_test.shape[0]}\n"
                f"File: {cpath}\n"
                f"(This assumes correct01 is saved in the same order as test samples after masking + day split.)"
            )

        # per day, per label
        for day in days_sorted:
            dm = (day_test == int(day))
            if not np.any(dm):
                continue
            y_d = y_test[dm]
            c_d = correct01[dm]

            for c in class_ids:
                cm = (y_d == int(c))
                if not np.any(cm):
                    continue
                A_perlbl[si, day_to_col[int(day)], cls_to_k[int(c)]] = float(np.mean(c_d[cm]))

    return days_sorted, class_ids, A_perlbl, seeds


# ---------- main plotting ----------
def plot_perday_overall_and_perlabel(args):
    # ---- load label info (for per-label accuracy) ----
    labels = npy_loader(args.label_path)       # (N,)
    day_info = npy_loader(args.day_info_path)  # (N,)

    target_style = None
    if args.target_style_path is not None:
        target_style = npy_loader(args.target_style_path)
        target_style = np.asarray(target_style).astype(bool)

    allowed = parse_label_mask(args.label_mask)

    labels, day_info, target_style = apply_masks(
        labels=labels,
        day_info=day_info,
        target_style=target_style,
        target_type=args.target_type,
        label_mask_allowed=allowed,
    )

    y = labels.astype(int)
    day_info_i = np.asarray(day_info).astype(int)

    slicing_day = np.unique(day_info_i)[args.slicing_day-1]
    test_mask = day_info_i > slicing_day
    # test_mask = day_info_i > int(args.slicing_day)


    y_test = y[test_mask]
    day_test = day_info_i[test_mask]

    # ---- load per-seed per-day overall accuracy ----
    seed_files = _collect_seed_npz(args.out_dir, args.target_type, args.boundary, int(args.slicing_day))
    print(f"Found {seed_files}")

    days_overall, A_overall, seeds, n_classes = _load_overall_perday(seed_files)
    mean_acc = np.nanmean(A_overall, axis=0)
    std_acc = np.nanstd(A_overall, axis=0)

    # ---- compute per-label accuracy from correct01 + labels/day ----
    days_lbl, class_ids, A_perlbl, seeds2 = _load_perlabel_from_correct01(seed_files, y_test=y_test, day_test=day_test)

    # Align x-axis (use intersection, just in case)
    days_common = np.array(sorted(set(days_overall.tolist()).intersection(days_lbl.tolist())), dtype=int)
    if days_common.size == 0:
        raise ValueError("No common days between overall-perday files and label/day_info test days.")

    # index maps
    d_overall = {int(d): i for i, d in enumerate(days_overall)}
    d_lbl = {int(d): i for i, d in enumerate(days_lbl)}
    overall_idx = np.array([d_overall[int(d)] for d in days_common], dtype=int)
    lbl_idx = np.array([d_lbl[int(d)] for d in days_common], dtype=int)

    mean_acc_c = mean_acc[overall_idx]
    std_acc_c = std_acc[overall_idx]
    A_perlbl_c = A_perlbl[:, lbl_idx, :]

    # ---- plot (ONE figure, two subplots) ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

    ax0 = axes[0]
    ax0.plot(days_common, mean_acc_c, marker="o", label=f"mean (nSeeds={len(seeds)})")
    ax0.fill_between(days_common, mean_acc_c - std_acc_c, mean_acc_c + std_acc_c, alpha=0.2)
    ax0.set_xlabel("Test day")
    ax0.set_ylabel("Accuracy (mean ± std over seeds)")
    ax0.set_ylim(0, 1)
    ax0.set_title("Overall accuracy")
    ax0.legend()
    ax0.grid(True, alpha=0.2)

    ax1 = axes[1]
    for k, c in enumerate(class_ids):
        m = np.nanmean(A_perlbl_c[:, :, k], axis=0)
        s = np.nanstd(A_perlbl_c[:, :, k], axis=0)
        ax1.plot(days_common, m, marker="o", label=f"label {int(c)}")
        ax1.fill_between(days_common, m - s, m + s, alpha=0.15)

    ax1.set_xlabel("Test day")
    ax1.set_title("Accuracy per label (mean ± std over seeds)")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # ---- Subplot 3: label sample counts (stacked bar) ----
    ax2 = axes[2]

    counts = np.zeros((len(days_common), len(class_ids)), dtype=int)
    for i, day in enumerate(days_common):
        dm = (day_test == int(day))
        y_d = y_test[dm]
        for k, c in enumerate(class_ids):
            counts[i, k] = int(np.sum(y_d == int(c)))

    bottom = np.zeros(len(days_common), dtype=int)
    for k, c in enumerate(class_ids):
        ax2.bar(days_common, counts[:, k], bottom=bottom, label=f"label {int(c)}")
        bottom += counts[:, k]

    ax2.set_xlabel("Test day")
    ax2.set_ylabel("#samples")
    ax2.set_title("Test samples per label")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    fig.suptitle(
        f"{args.target_type} | boundary={args.boundary} | slicing_day={args.slicing_day} | seeds={len(seeds)}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save_path is not None:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        fig.savefig(args.save_path, dpi=200)
        print("saved:", args.save_path)

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # results
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--target_type", type=str, required=True, choices=["center-out", "random"])
    p.add_argument("--boundary", type=str, required=True, help='e.g. "0.25_0.5_0.75"')
    p.add_argument("--slicing_day", type=int, required=True)

    # label/day info (for per-label plot)
    p.add_argument("--label_path", type=str, required=True)
    p.add_argument("--day_info_path", type=str, required=True)
    p.add_argument("--target_style_path", type=str, default=None)
    p.add_argument("--label_mask", type=str, default=None, help='e.g. "0,1,2" or "0-2,5"')

    # io
    p.add_argument("--save_path", type=str, default=None, help="Optional: save ONE combined figure")
    p.add_argument("--no_show", action="store_true")

    args = p.parse_args()
    plot_perday_overall_and_perlabel(args)
