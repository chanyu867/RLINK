import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# filename matching
# -------------------------

def _collect_perday_npz(run_dir: str, boundary: str, slicing_day: int):
    """
    Find files like:
      * _b{boundary}_seed{seed}_perday_acc_Nclasses{n}_day{slicing_day}.npz
    """
    files = glob.glob(os.path.join(run_dir, "*.npz"))
    out = []
    for f in files:
        base = os.path.basename(f)
        m = re.search(
            r"_b(?P<b>\d+(?:\.\d+)?(?:_\d+(?:\.\d+)?)+)"
            r"_seed(?P<seed>\d+)_perday_acc_Nclasses(?P<ncls>\d+)_day(?P<sday>\d+)\.npz$",
            base
        )
        if not m:
            continue
        if m.group("b") != boundary:
            continue
        if int(m.group("sday")) != int(slicing_day):
            continue
        out.append((f, int(m.group("seed")), int(m.group("ncls"))))
    out.sort(key=lambda x: x[1])
    return out


def _collect_correct01_npy(run_dir: str, boundary: str, slicing_day: int):
    """
    Find files like:
      * _b{boundary}_seed{seed}_test_correct01_Nclasses{n}_day{slicing_day}.npy
    """
    files = glob.glob(os.path.join(run_dir, "*_test_correct01_*.npy"))
    out = []
    for f in files:
        base = os.path.basename(f)
        m = re.search(
            r"_b(?P<b>\d+(?:\.\d+)?(?:_\d+(?:\.\d+)?)+)"
            r"_seed(?P<seed>\d+)_test_correct01_Nclasses(?P<ncls>\d+)_day(?P<sday>\d+)\.npy$",
            base
        )
        if not m:
            continue
        if m.group("b") != boundary:
            continue
        if int(m.group("sday")) != int(slicing_day):
            continue
        out.append((f, int(m.group("seed")), int(m.group("ncls"))))
    out.sort(key=lambda x: x[1])
    return out


# -------------------------
# load + aggregate helpers
# -------------------------

def _load_perday_matrix(perday_files):
    """
    perday_files: [(npz_path, seed, ncls), ...]
    returns: days_sorted, A (nSeeds,nDays), seeds, ncls
    """
    if not perday_files:
        return None, None, None, None

    ncls = perday_files[0][2]
    seed_to_map = {}
    all_days = set()

    for path, seed, _ncls in perday_files:
        z = np.load(path, allow_pickle=False)
        days = z["days"].astype(int)
        acc = z["acc"].astype(float)
        seed_to_map[seed] = {int(d): float(a) for d, a in zip(days, acc)}
        all_days.update(days.tolist())

    days_sorted = np.array(sorted(all_days), dtype=int)
    seeds = np.array(sorted(seed_to_map.keys()), dtype=int)

    A = np.full((len(seeds), len(days_sorted)), np.nan, dtype=float)
    for si, s in enumerate(seeds):
        d2a = seed_to_map[s]
        for di, d in enumerate(days_sorted):
            if int(d) in d2a:
                A[si, di] = d2a[int(d)]

    return days_sorted, A, seeds, ncls


def _load_correct01_matrix(correct_files):
    """
    correct_files: [(npy_path, seed, ncls), ...]
    returns: C (nSeeds, nTestSamples), seeds, ncls
    """
    if not correct_files:
        return None, None, None

    ncls = correct_files[0][2]
    seeds = []
    rows = []
    for path, seed, _ncls in correct_files:
        c = np.load(path, allow_pickle=False).astype(float).ravel()
        rows.append(c)
        seeds.append(seed)

    L = [r.shape[0] for r in rows]
    if len(set(L)) != 1:
        raise ValueError(f"correct01 length mismatch across seeds: {L}")

    return np.stack(rows, axis=0), np.array(seeds, dtype=int), ncls


# -------------------------
# reproduce test positions (same masking logic as training code)
# -------------------------

def _apply_masks_like_training(day_info, target_style, target_type):
    """
    Matches your training code exactly:
      if target_type=="center-out": keep = ~t
      if target_type=="random":    keep = t
    """
    if target_type is None:
        keep = np.ones_like(day_info, dtype=bool)
        return keep

    t = np.asarray(target_style).astype(bool)
    if target_type == "center-out":
        keep = ~t
    elif target_type == "random":
        keep = t
    else:
        raise ValueError("target_type must be 'center-out' or 'random'")
    return keep


def _get_test_positions(
    day_info_path: str,
    target_style_path: str,
    label_path: str,
    target_type: str,
    slicing_day: int,
):
    """
    Returns test_pos aligned with test_mask ordering in training:
      day_info_i = day_info[mask]
      test_mask = day_info_i > slicing_day
      test_pos = pos[mask][test_mask]
    """
    day_info = np.load(day_info_path, allow_pickle=False).astype(int).ravel()
    target_style = np.load(target_style_path, allow_pickle=False).astype(bool).ravel()
    pos = np.load(label_path, allow_pickle=False)
    print("pos: ", pos[:10])

    if not (day_info.shape[0] == target_style.shape[0] == pos.shape[0]):
        raise ValueError("day_info / target_style / position_value length mismatch")

    keep = _apply_masks_like_training(day_info, target_style, target_type)

    day_kept = day_info[keep]
    pos_kept = pos[keep]

    test_mask = day_kept > int(slicing_day)
    test_pos = pos_kept[test_mask]
    return test_pos


def _bin_positions(pos, n_blocks: int):
    p = np.clip(np.asarray(pos, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_blocks + 1)
    b = np.digitize(p, edges, right=False) - 1
    b[b == n_blocks] = n_blocks - 1
    return b.astype(int)


# -------------------------
# main plotting function (requested)
# -------------------------

def plot_performance_and_stackedbars(
    out_root: str,  # .../Results/Perceptron/idx_position
    day_info_path: str,
    target_style_path: str,
    label_path: str,   # continuous finger position in [0,1]
    boundaries=("0.33_0.66", "0.25_0.5_0.75", "0.2_0.4_0.6_0.8"),
    target_types=("center-out", "random"),
    slicing_days=(10, 30, 50, 90, 150),
    n_blocks: int = 20,
    save_dir: str | None = None,
):
    """
    For each (target_type, boundary):
      A) Accuracy figure: subplots per slicing_day (1 subplot = 1 slicing_day)
         curve is per-test-day mean±std across seeds.
      B) Stacked bar figure: subplots per slicing_day (1 subplot = 1 slicing_day)
         x=binned finger position (n_blocks), y=#samples
         bottom=fail (blue), top=correct (red)
         Aggregation across seeds uses expected correct count:
           p_correct(sample)=mean(correct01 across seeds)
    """
    if save_dir is None:
        save_dir = os.path.join(out_root, "plots")
    os.makedirs(save_dir, exist_ok=True)

    ncols = 3
    nrows = int(np.ceil(len(slicing_days) / ncols))

    for target_type in target_types:
        run_dir = os.path.join(out_root, target_type)

        for boundary in boundaries:
            # -------------------------
            # A) Accuracy subplots (no mixing slicing_days in one axis)
            # -------------------------
            figA, axesA = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
            axesA = np.array(axesA).reshape(-1)

            used_ncls = None

            for i, sday in enumerate(slicing_days):
                ax = axesA[i]

                perday_files = _collect_perday_npz(run_dir, boundary, sday)
                if not perday_files:
                    ax.set_title(f"slicing_day={sday} (no files)")
                    ax.axis("off")
                    continue

                days_sorted, A, seeds, ncls = _load_perday_matrix(perday_files)
                mean_acc = np.nanmean(A, axis=0)
                std_acc = np.nanstd(A, axis=0)

                ax.plot(days_sorted, mean_acc, marker="o")
                ax.fill_between(days_sorted, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Test day")
                ax.set_ylabel("Acc (mean±std)")
                ax.set_title(f"slicing_day={sday} (nSeeds={len(seeds)})")

                used_ncls = ncls if used_ncls is None else used_ncls

            for j in range(len(slicing_days), len(axesA)):
                axesA[j].axis("off")

            figA.suptitle(f"{target_type} | b={boundary} | Nclasses={used_ncls}", y=1.02)
            figA.tight_layout()
            outA = os.path.join(save_dir, f"ACC_subplots_{target_type}_b{boundary}_Nclasses{used_ncls}.png")
            figA.savefig(outA, dpi=200, bbox_inches="tight")
            plt.close(figA)
            print("saved:", outA)

            # -------------------------
            # B) Stacked bar subplots per slicing_day
            # -------------------------
            figB, axesB = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
            axesB = np.array(axesB).reshape(-1)

            for i, sday in enumerate(slicing_days):
                ax = axesB[i]

                correct_files = _collect_correct01_npy(run_dir, boundary, sday)
                if not correct_files:
                    ax.set_title(f"slicing_day={sday} (no correct01)")
                    ax.axis("off")
                    continue

                C, seeds, ncls = _load_correct01_matrix(correct_files)
                used_ncls = ncls if used_ncls is None else used_ncls

                # reproduce test finger positions (aligned with correct01 ordering)
                test_pos = _get_test_positions(
                    day_info_path=day_info_path,
                    target_style_path=target_style_path,
                    label_path=label_path,
                    target_type=target_type,
                    slicing_day=sday,
                )
                if test_pos.shape[0] != C.shape[1]:
                    raise ValueError(
                        f"Length mismatch for stacked bar at {target_type} b{boundary} day{sday}: "
                        f"test_pos={test_pos.shape[0]} vs correct01={C.shape[1]}"
                    )

                # expected correctness per sample across seeds
                p_correct = np.mean(C, axis=0)  # (nTest,)

                # bin positions
                bin_id = _bin_positions(test_pos, n_blocks=n_blocks)

                total = np.bincount(bin_id, minlength=n_blocks).astype(float)
                corr = np.bincount(bin_id, weights=p_correct, minlength=n_blocks).astype(float)
                fail = total - corr

                edges = np.linspace(0.0, 1.0, n_blocks + 1)
                x = edges[:-1]                    # left edge of each bin in [0,1)
                w = edges[1] - edges[0]           # bin width

                ax.bar(x, fail, width=w, align="edge", color="blue", label="incorrect")
                ax.bar(x, corr, width=w, align="edge", bottom=fail, color="red", label="correct")
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel(f"finger position (0–1), binned into {n_blocks} blocks")
                ax.set_ylabel("#samples")
                ax.set_title(f"slicing_day={sday} (nSeeds={len(seeds)})")

            for j in range(len(slicing_days), len(axesB)):
                axesB[j].axis("off")

            # one legend for the whole fig
            handles, labels = axesB[0].get_legend_handles_labels()
            if handles:
                figB.legend(handles, labels, loc="upper right")

            figB.suptitle(f"{target_type} | b={boundary} | stacked counts | Nclasses={used_ncls}", y=1.02)
            figB.tight_layout()
            outB = os.path.join(save_dir, f"BAR_subplots_{target_type}_b{boundary}_Nclasses{used_ncls}_blocks{n_blocks}.png")
            figB.savefig(outB, dpi=200, bbox_inches="tight")
            plt.close(figB)
            print("saved:", outB)


# -------------------------
# Example call
# -------------------------
if __name__ == "__main__":
    out_root = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/idx_position"

    day_info_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
    target_style_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

    # IMPORTANT: continuous finger position values in [0,1], aligned to the original (unmasked) samples.
    # You must point to the correct file for idx/mrs position values.
    label_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/idx_position_all.npy"

    plot_performance_and_stackedbars(
        out_root=out_root,
        day_info_path=day_info_path,
        target_style_path=target_style_path,
        label_path=label_path,
        boundaries=("0.33_0.66", "0.25_0.5_0.75", "0.2_0.4_0.6_0.8"),
        target_types=("center-out", "random"),
        slicing_days=(10, 30, 50, 90, 150),
        n_blocks=20,
        save_dir=os.path.join(out_root, "plots_v2"),
    )
