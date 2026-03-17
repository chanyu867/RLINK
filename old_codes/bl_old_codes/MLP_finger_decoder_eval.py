#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from MLP_finger_decoder_with_lag import make_lagged_features


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ----------------- utilities -----------------
def npy_loader(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def parse_label_mask(s: str | None):
    if s is None or str(s).strip() == "":
        return None
    s = str(s).strip()
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def apply_masks(sbp, labels, day_info, time_within_trial, target_style, target_type, label_mask_allowed):
    # 1) label_mask
    if label_mask_allowed is not None:
        keep = np.isin(labels.astype(int), np.asarray(label_mask_allowed, dtype=int))
        sbp = sbp[keep]
        labels = labels[keep]
        day_info = day_info[keep]
        time_within_trial = time_within_trial[keep]
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

        sbp = sbp[keep]
        labels = labels[keep]
        day_info = day_info[keep]
        time_within_trial = time_within_trial[keep]
        target_style = t[keep]

    return sbp, labels, day_info, time_within_trial, target_style

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_hidden_sizes(s: str) -> list[int]:
    s = str(s).strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def per_day_accuracy_from_arrays(day_arr: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray):
    """
    day_arr: (N_test,) int day number for each test sample
    """
    uniq_days = np.unique(day_arr.astype(int))
    day_acc = np.zeros(len(uniq_days), dtype=float)
    day_n = np.zeros(len(uniq_days), dtype=int)

    for i, d in enumerate(uniq_days):
        m = (day_arr == d)
        day_n[i] = int(m.sum())
        day_acc[i] = float((y_pred[m] == y_true[m]).mean()) if day_n[i] > 0 else np.nan

    return uniq_days.astype(int), day_acc, day_n


# ----------------- weight file parsing/loading -----------------
def _parse_weight_file(path: str):
    base = os.path.basename(path)
    m = re.search(
        r"^(?P<prefix>.+)_(?P<tt>center-out|random)"
        r"_b(?P<boundary>\d+(?:\.\d+)?(?:_\d+(?:\.\d+)?)+)"
        r"_hs(?P<hs>\d+(?:-\d+)?)"
        r"_seed(?P<seed>\d+)"
        r"_Nclasses(?P<ncls>\d+)"
        r"_day(?P<dayidx>\d+)"
        r"_weights\.npz$",
        base,
    )
    if not m:
        return None
    return {
        "prefix": m.group("prefix"),
        "target_type": m.group("tt"),
        "boundary": m.group("boundary"),
        "hidden_sizes": m.group("hs"),  # "128-64" (string)
        "seed": int(m.group("seed")),
        "n_classes": int(m.group("ncls")),
        "dayidx": int(m.group("dayidx")),
        "path": path,
    }


def load_model_from_weight_npz(w_path: str, in_dim: int, device: torch.device):
    """
    Loads:
      - hidden_sizes
      - state_dict (numpy arrays)
      - scale/log_scale + scaler stats if present
    """
    d = np.load(w_path, allow_pickle=True)

    if "hidden_sizes" not in d.files or "state_dict" not in d.files:
        raise KeyError(f"Missing hidden_sizes/state_dict in {w_path}. keys={d.files}")

    hidden = [int(x) for x in np.array(d["hidden_sizes"]).tolist()]
    # metadata
    scale = bool(d["scale"]) if "scale" in d.files else False
    log_scale = bool(d["log_scale"]) if "log_scale" in d.files else False

    scaler_mean = d["scaler_mean"] if "scaler_mean" in d.files else None
    scaler_scale = d["scaler_scale"] if "scaler_scale" in d.files else None

    # state dict saved as object array containing a dict
    state_obj = d["state_dict"]
    if isinstance(state_obj, np.ndarray) and state_obj.dtype == object:
        state = state_obj.item()
    elif isinstance(state_obj, dict):
        state = state_obj
    else:
        state = state_obj

    # infer out_dim from last layer weight
    # expect keys like "net.0.weight", ..., "net.<last>.weight"
    last_w_key = None
    for k in state.keys():
        if k.endswith(".weight"):
            last_w_key = k
    if last_w_key is None:
        raise ValueError(f"Could not find a .weight key in state_dict for {w_path}")
    out_dim = int(state[last_w_key].shape[0])

    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)

    torch_state = {k: torch.from_numpy(v) for k, v in state.items()}
    model.load_state_dict(torch_state, strict=True)
    model.eval()

    meta = {
        "hidden": hidden,
        "out_dim": out_dim,
        "scale": scale,
        "log_scale": log_scale,
        "scaler_mean": None if scaler_mean is None else np.array(scaler_mean, dtype=float),
        "scaler_scale": None if scaler_scale is None else np.array(scaler_scale, dtype=float),
        "n_lags": int(d["n_lags"]) if "n_lags" in d.files else 0,
        "lag_step": int(d["lag_step"]) if "lag_step" in d.files else 1,
        "lag_group": str(d["lag_group"]) if "lag_group" in d.files else "day",
    }
    return model, meta


def apply_saved_scaler(X: np.ndarray, meta: dict):
    """
    Uses training-time scaler stats from weight file (if scale=True).
    """
    X2 = X.copy()
    if meta["scale"]:
        print("Applying saved scaler to tests data...")
        if meta["log_scale"]:
            X2 = np.log1p(X2)
        mean = meta["scaler_mean"]
        scale = meta["scaler_scale"]
        if mean is None or scale is None:
            raise ValueError("scale=True but scaler_mean/scaler_scale not saved in weight file.")
        X2 = (X2 - mean) / scale
    return X2.astype(np.float32)


# ----------------- plotting -----------------
def plot_two_panel(days: np.ndarray,
                   mean: np.ndarray, std: np.ndarray,
                   class_ids: np.ndarray,
                   mean_pc: np.ndarray, std_pc: np.ndarray,
                   title: str, out_png: str):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: overall
    ax0.plot(days, mean, marker="o", label="overall mean (over seeds)")
    ax0.fill_between(days, mean - std, mean + std, alpha=0.2)
    mask = np.isfinite(days) & np.isfinite(mean)
    if np.sum(mask) >= 2:
        x = days[mask].astype(float)
        y = mean[mask].astype(float)
        slope, intercept = np.polyfit(x, y, deg=1)  # y ≈ slope*x + intercept
        y_fit = slope * x + intercept
        ax0.plot(x, y_fit, color="red", linewidth=2, label=f"slope={slope:.4g}/day")

        angle_deg = float(np.degrees(np.arctan(slope)))  # in data units (accuracy per day)
        mean_over_days = float(np.nanmean(y))
        ax0.text(
            0.02, 0.95,
            f"mean(acc) = {mean_over_days:.4f}\n"
            f"slope = {slope:.4g} / day\n"
            f"angle = {angle_deg:.2f}°",
            transform=ax0.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.2)
        )

    ax0.set_ylabel("Accuracy")
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.2)
    ax0.legend()

    # Bottom: per-class
    for k, c in enumerate(class_ids):
        m = mean_pc[:, k]
        s = std_pc[:, k]
        ax1.plot(days, m, marker="o", label=f"class {int(c)}")
        ax1.fill_between(days, m - s, m + s, alpha=0.15)

    ax1.set_xlabel("Test day")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def per_day_perclass_accuracy(day_arr: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, class_ids: np.ndarray):
    """
    Returns:
      uniq_days: (n_days,)
      acc_pc:   (n_days, K)  per-class accuracy, NaN if that class not present on that day
      n_pc:     (n_days, K)  counts
    """
    uniq_days = np.unique(day_arr.astype(int))
    class_ids = np.asarray(class_ids).astype(int)
    K = len(class_ids)

    acc_pc = np.full((len(uniq_days), K), np.nan, dtype=float)
    n_pc = np.zeros((len(uniq_days), K), dtype=int)

    for i, d in enumerate(uniq_days):
        dm = (day_arr == d)
        yt = y_true[dm]
        yp = y_pred[dm]
        for k, c in enumerate(class_ids):
            cm = (yt == c)
            n_pc[i, k] = int(cm.sum())
            if n_pc[i, k] > 0:
                acc_pc[i, k] = float((yp[cm] == yt[cm]).mean())

    return uniq_days.astype(int), acc_pc, n_pc



def make_lagged_features_with_day(X: np.ndarray,
                                 y: np.ndarray,
                                 day: np.ndarray,
                                 group: np.ndarray | None,
                                 n_lags: int,
                                 lag_step: int = 1):
    """
    Same as training lag stack, but ALSO returns day aligned to each output sample (day[t]).
    group is used only to prevent crossing boundaries (trial/day).
    """
    if n_lags <= 0:
        return X, y, day

    if lag_step < 1:
        raise ValueError("lag_step must be >= 1")

    N, D = X.shape
    if group is None:
        group = np.zeros(N, dtype=int)

    X_out, y_out, day_out = [], [], []

    for g in np.unique(group):
        idx = np.where(group == g)[0]
        if idx.size == 0:
            continue
        start = n_lags * lag_step
        if idx.size <= start:
            continue

        for j in range(start, idx.size):
            t = idx[j]
            feats = [X[t]]
            for k in range(1, n_lags + 1):
                feats.append(X[idx[j - k * lag_step]])
            X_out.append(np.concatenate(feats, axis=0))
            y_out.append(y[t])
            day_out.append(day[t])

    X_lag = np.stack(X_out, axis=0).astype(np.float32)
    y_eff = np.asarray(y_out, dtype=int)
    day_eff = np.asarray(day_out, dtype=int)
    return X_lag, y_eff, day_eff




# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)
    ap.add_argument("--target_style_path", type=str, default=None)
    ap.add_argument("--target_type", type=str, required=True, choices=["center-out", "random"])
    ap.add_argument("--label_mask", type=str, default=None)

    # where weights live
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Base results dir. Expect weights under: out_dir/<target_type>/")
    ap.add_argument("--save_dir", type=str, required=True,
                    help="Base results dir. Expect weights under: out_dir/<target_type>/")
    ap.add_argument("--prefix", type=str, default="mlp",
                    help="Prefix used in filenames (used for filtering loosely; OK if broader)")

    # filter
    ap.add_argument("--boundary", type=str, required=True, help='e.g. "0.33_0.66"')

    # slicing_day index (IMPORTANT: use the same algorithm)
    ap.add_argument("--slicing_day", type=int, required=True,
                    help="1-indexed index into np.unique(day_info_i)")

    # inference
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--max_test_samples", type=int, default=0,
                help="If >0, evaluate only the first N test samples (after lag/scale).")

    ap.add_argument("--trial_bin_path", type=str, required=True,
                help="trial_bin.npy (0.0 at trial start)")

    args = ap.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # ---- load arrays ----
    sbp = npy_loader(args.sbp_path)
    labels = npy_loader(args.label_path)
    day_info = npy_loader(args.day_info_path)
    time_within_trial = npy_loader(args.trial_bin_path)
    if time_within_trial.shape[0] != sbp.shape[0]:
        raise ValueError("Length mismatch: trial_bin vs sbp")

    if sbp.shape[0] != labels.shape[0] or sbp.shape[0] != day_info.shape[0]:
        raise ValueError("Length mismatch among sbp/labels/day_info")

    target_style = None
    if args.target_style_path is not None:
        target_style = npy_loader(args.target_style_path)
        if target_style.shape[0] != sbp.shape[0]:
            raise ValueError("Length mismatch: target_style vs sbp")
        target_style = np.asarray(target_style).astype(bool)

    allowed = parse_label_mask(args.label_mask)

    # ---- apply same masks as training ----
    sbp, labels, day_info, time_within_trial, _ = apply_masks(
        sbp=sbp,
        labels=labels,
        day_info=day_info,
        time_within_trial=time_within_trial,
        target_style=target_style,
        target_type=args.target_type,
        label_mask_allowed=allowed,
    )

    X = sbp.astype(np.float32)
    y = labels.astype(int)
    day_info_i = np.asarray(day_info).astype(int)

    # ---- slicing_day (must match your rule) ----
    uniq_days = np.unique(day_info_i)
    if args.slicing_day < 1 or args.slicing_day > len(uniq_days):
        raise ValueError(f"--slicing_day must be in [1, {len(uniq_days)}]")

    slicing_day_value = uniq_days[args.slicing_day - 1]
    test_mask = day_info_i > slicing_day_value
    if not np.any(test_mask):
        raise ValueError("No samples for test_mask (day_info_i > slicing_day_value).")

    X_test_raw = X[test_mask]
    y_test = y[test_mask]
    day_test = day_info_i[test_mask]
    class_ids = np.array(sorted(np.unique(y_test).astype(int).tolist()), dtype=int)
    time_test = time_within_trial[test_mask]

    trial_starts = (time_test == 0.0)
    if len(time_test) > 0 and (time_test[0] != 0.0):
        trial_starts[0] = True  # safety: masked/sliced starts mid-trial

    trial_id_test = np.cumsum(trial_starts).astype(int) - 1  # 0..n_trials-1

    # summarize: which trial belongs to which day (mode day within trial)
    trial_to_day = {}
    trial_to_len = {}
    bad_trials = []

    for tid in np.unique(trial_id_test):
        m = (trial_id_test == tid)
        days_in_trial = day_test[m]
        vals, cnts = np.unique(days_in_trial, return_counts=True)
        day_mode = int(vals[np.argmax(cnts)])
        trial_to_day[int(tid)] = day_mode
        trial_to_len[int(tid)] = int(m.sum())
        if len(vals) > 1:
            bad_trials.append((int(tid), vals.astype(int).tolist(), cnts.tolist()))

    # print a compact summary
    day_vals, day_counts = np.unique(np.array(list(trial_to_day.values()), dtype=int), return_counts=True)
    print("Trials per day (in test):", dict(zip(day_vals.tolist(), day_counts.tolist())))
    print("Total test trials:", int(np.unique(trial_id_test).size))
    if bad_trials:
        print("WARNING: trials spanning multiple days (showing up to 10):")
        for row in bad_trials[:10]:
            print("  trial", row[0], "days", row[1], "counts", row[2])



    logger.info(f"slicing_day_value={slicing_day_value} (idx={args.slicing_day})  testN={len(y_test)}")

    # ---- find weight files (same target_type folder) ----
    weight_dir = os.path.join(args.out_dir, args.target_type, "weights")
    if not os.path.isdir(weight_dir):
        raise FileNotFoundError(f"Missing target_type folder: {weight_dir}")

    weight_files = sorted(glob.glob(os.path.join(weight_dir, "*.npz")))
    weight_infos = []
    for wp in weight_files:
        info = _parse_weight_file(wp)
        if info is None:
            continue
        # if info["target_type"] != args.target_type:
        #     continue
        if info["boundary"] != args.boundary:
            continue
        # only use weights for this dayidx (args.slicing_day)
        if info["dayidx"] != int(args.slicing_day):
            continue
        weight_infos.append(info)

    if not weight_infos:
        raise FileNotFoundError(
            f"No matching weights found in {weight_dir}\n"
            f"Expected something like: {args.prefix}_{args.target_type}_b{args.boundary}_seedX_weights_..._day{args.slicing_day}.npz"
        )

    # ---- run inference per seed weight ----
    per_seed = []  # list of (seed, uniq_days, day_acc)
    all_days_set = set()
    X_test_raw_base = X_test_raw
    y_test_base     = y_test
    day_test_base   = day_test

    for info in weight_infos:
        seed = info["seed"]
        w_path = info["path"]

        # --- load weight file once ---
        w = np.load(w_path, allow_pickle=True)

        # --- meta for BOTH lag + scaling ---
        meta0 = {
            # lag
            "n_lags": int(w["n_lags"]) if "n_lags" in w.files else 0,
            "lag_step": int(w["lag_step"]) if "lag_step" in w.files else 1,
            "lag_group": str(w["lag_group"]) if "lag_group" in w.files else "day",
            # scaler (train-time)
            "scale": bool(w["scale"]) if "scale" in w.files else False,
            "log_scale": bool(w["log_scale"]) if "log_scale" in w.files else False,
            "scaler_mean": (None if w["scaler_mean"] is None else np.array(w["scaler_mean"], dtype=float)) if "scaler_mean" in w.files else None,
            "scaler_scale": (None if w["scaler_scale"] is None else np.array(w["scaler_scale"], dtype=float)) if "scaler_scale" in w.files else None,
        }

        # --- start from base arrays every seed (重要) ---
        Xr = X_test_raw_base
        yr = y_test_base
        dr = day_test_base

        # --- apply lag stacking BEFORE model load ---
        if args.max_test_samples and args.max_test_samples > 0:
            print("max_test_samples: ", args.max_test_samples)
            Xr = Xr[:args.max_test_samples]
            yr = yr[:args.max_test_samples]
            dr = dr[:args.max_test_samples]

        group = None
        if meta0["lag_group"] == "day":
            group = dr
        elif meta0["lag_group"] == "trial":
            # build trial_id aligned to current Xr/yr/dr BEFORE lagging
            # (Xr/yr/dr are the test arrays here)
            # We already computed trial_id_test for the full test set; but if you slice max_test_samples,
            # make sure to slice trial_id_test the same way.
            group = trial_id_test[:len(dr)]  # safe if no extra slicing; if you slice, slice group too

        Xr, yr, dr = make_lagged_features_with_day(
            X=Xr, y=yr, day=dr, group=group,
            n_lags=meta0["n_lags"],
            lag_step=meta0["lag_step"],
        )

        # --- now input dim matches checkpoint ---
        model, meta = load_model_from_weight_npz(w_path, in_dim=Xr.shape[1], device=device)
        model.eval()

        # --- apply saved scaler (train stats) ---
        X_test = apply_saved_scaler(Xr, meta)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(yr.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=False,
        )

        y_pred_list = []
        with torch.no_grad():
            for b_i, (xb, _) in enumerate(test_loader):
                if b_i%1000 == 0:
                    print(f"test is running {b_i}")
                xb = xb.to(device, non_blocking=False)
                logits = model(xb)
                y_pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_pred = np.concatenate(y_pred_list, axis=0).astype(int)
        y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
        yr     = np.asarray(yr, dtype=int).reshape(-1)
        dr     = np.asarray(dr, dtype=int).reshape(-1)

        n = min(len(y_pred), len(yr))
        if n == 0:
            logger.warning(f"[seed={seed}] No samples after lag/limit. Skip.")
            continue

        y_pred = y_pred[:n]
        yr     = yr[:n]
        dr     = dr[:n]

        # --- per-day / per-class must use dr/yr (lagged-aligned) ---
        uniq_d, day_acc, day_n = per_day_accuracy_from_arrays(dr, y_pred, yr)
        uniq_d2, acc_pc, n_pc  = per_day_perclass_accuracy(dr, y_pred, yr, class_ids)

        # uniq_d が空ならスキップ（全NaN防止）
        if len(uniq_d) == 0:
            logger.warning(f"[seed={seed}] per-day arrays empty. Skip.")
            continue

        per_seed.append((seed, uniq_d, day_acc, acc_pc))
        all_days_set.update(uniq_d.tolist())

        overall = float(np.mean(y_pred == yr))
        logger.info(f"[seed={seed}] overall_acc(future)={overall:.4f}  days={len(uniq_d)}")

    # ---- aggregate mean/std across seeds for each future day ----
    days_sorted = np.array(sorted(all_days_set), dtype=int)
    seeds_sorted = np.array(sorted([s for (s, _, _, _) in per_seed]), dtype=int)

    A = np.full((len(seeds_sorted), len(days_sorted)), np.nan, dtype=float)
    A_pc = np.full((len(seeds_sorted), len(days_sorted), len(class_ids)), np.nan, dtype=float)

    seed_to_row = {int(s): i for i, s in enumerate(seeds_sorted)}
    day_to_col = {int(d): j for j, d in enumerate(days_sorted)}

    for seed, dlist, alist, acc_pc in per_seed:
        r = seed_to_row[int(seed)]
        for i, d in enumerate(dlist):
            c = day_to_col[int(d)]
            A[r, c] = float(alist[i])
            A_pc[r, c, :] = acc_pc[i, :]
    
    mean_acc = np.nanmean(A, axis=0)
    std_acc = np.nanstd(A, axis=0)
    mean_pc = np.nanmean(A_pc, axis=0)   # (n_days, K)
    std_pc  = np.nanstd(A_pc, axis=0)

    # ---- save plot to subfolder under same dir ----
    os.makedirs(args.save_dir, exist_ok=True)

    out_png = os.path.join(
        args.save_dir,
        f"{args.prefix}_{args.target_type}_b{args.boundary}_slicing{args.slicing_day}_future_perday_meanstd.png"
    )
    title = f"{args.target_type} | boundary={args.boundary} | slicing_day={args.slicing_day} (value={slicing_day_value}) | seeds={len(seeds_sorted)}"
    plot_two_panel(days_sorted, mean_acc, std_acc, class_ids, mean_pc, std_pc, title=title, out_png=out_png)

    # also save aggregate npz for reuse
    out_npz = out_png.replace(".png", ".npz")
    np.savez(
        out_npz,
        target_type=args.target_type,
        boundary=args.boundary,
        slicing_day_idx=int(args.slicing_day),
        slicing_day_value=int(slicing_day_value),
        seeds=seeds_sorted,
        days=days_sorted,
        perday_acc_matrix=A,
        mean=mean_acc,
        std=std_acc,
    )

    print("saved:")
    print(" ", out_png)
    print(" ", out_npz)


if __name__ == "__main__":
    main()
