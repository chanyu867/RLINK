#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =========================
# Utilities
# =========================
def npy_loader(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_label_mask(s: Optional[str]) -> Optional[np.ndarray]:
    """'0,1,2' -> np.array([0,1,2])"""
    if s is None or str(s).strip() == "":
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    return np.array([int(p) for p in parts], dtype=int)


# =========================
# Output dirs (match training style)
# =========================
@dataclass(frozen=True)
class OutPaths:
    root: str
    weights: str
    results: str
    plots: str


def make_outdirs(out_dir: str) -> OutPaths:
    root = out_dir
    weights = os.path.join(root, "weights")
    results = os.path.join(root, "results")
    plots = os.path.join(root, "plots")
    os.makedirs(weights, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return OutPaths(root=root, weights=weights, results=results, plots=plots)


def make_subdirs(base: str, name: str) -> str:
    p = os.path.join(base, name)
    os.makedirs(p, exist_ok=True)
    return p


# =========================
# Masking (must match training logic)
# =========================
@dataclass
class LoadedData:
    sbp: np.ndarray                      # (N, D)
    labels: np.ndarray                   # (N,)
    day_info: np.ndarray                 # (N,)
    trial_bin: np.ndarray                # (N,) float, 0.0 at trial start
    target_style: Optional[np.ndarray]   # (N,) bool or None


def load_all_arrays(args) -> LoadedData:
    sbp = npy_loader(args.sbp_path)
    labels = npy_loader(args.label_path)
    day_info = npy_loader(args.day_info_path)
    trial_bin = npy_loader(args.trial_bin_path)

    if sbp.shape[0] != labels.shape[0] or sbp.shape[0] != day_info.shape[0]:
        raise ValueError("Length mismatch among sbp/labels/day_info")
    if trial_bin.shape[0] != sbp.shape[0]:
        raise ValueError("Length mismatch: trial_bin vs sbp")

    target_style = None
    if args.target_style_path is not None:
        target_style = np.load(args.target_style_path, allow_pickle=False)
        if target_style.shape[0] != sbp.shape[0]:
            raise ValueError("Length mismatch: target_style vs sbp")
        target_style = np.asarray(target_style).astype(bool)

    return LoadedData(
        sbp=sbp,
        labels=labels,
        day_info=day_info,
        trial_bin=trial_bin,
        target_style=target_style,
    )


def apply_task_masks(
    data: LoadedData,
    target_type: Optional[str],
    allowed_labels: Optional[np.ndarray],
) -> LoadedData:
    """
    EXACT same logic as your Perceptron training script:
      (1) label_mask: keep labels in allowed_labels
      (2) target_type:
          center-out => keep ~target_style
          random     => keep target_style
    """
    keep = np.ones(data.sbp.shape[0], dtype=bool)

    if allowed_labels is not None:
        keep &= np.isin(data.labels.astype(int), allowed_labels)

    if target_type is not None:
        if data.target_style is None:
            raise ValueError("--target_type was provided but --target_style_path is None.")
        t = np.asarray(data.target_style).astype(bool)
        if target_type == "center-out":
            keep &= ~t
        elif target_type == "random":
            keep &= t
        else:
            raise ValueError("--target_type must be one of: center-out, random")

    return LoadedData(
        sbp=data.sbp[keep],
        labels=data.labels[keep],
        day_info=data.day_info[keep],
        trial_bin=data.trial_bin[keep],
        target_style=(t[keep] if data.target_style is not None else None),
    )


def compute_slicing_day_value(day_info_i: np.ndarray, slicing_day_index_1based: int) -> int:
    """
    Must match training rule:
      slicing_day_value = unique_days[ slicing_day-1 ]  (within masked data)
    """
    uniq_days = np.unique(day_info_i.astype(int))
    if slicing_day_index_1based < 1 or slicing_day_index_1based > len(uniq_days):
        raise ValueError(f"--slicing_day must be in [1, {len(uniq_days)}]")
    return int(uniq_days[slicing_day_index_1based - 1])


# =========================
# Trial IDs + Trial->Day summary
# =========================
def build_trial_ids_from_trialbin(trial_bin_vec: np.ndarray) -> Tuple[np.ndarray, int]:
    trial_starts = (trial_bin_vec == 0.0)
    if len(trial_bin_vec) > 0 and (trial_bin_vec[0] != 0.0):
        trial_starts[0] = True
    trial_id = np.cumsum(trial_starts).astype(int) - 1
    return trial_id, int(trial_starts.sum())


def summarize_trials_per_day(day_vec: np.ndarray, trial_id_vec: np.ndarray) -> None:
    """
    Same logic as your eval:
      - assign each trial to day_mode (mode of day within that trial)
      - print trials per day + warnings if a trial spans multiple days
    """
    trial_to_day: Dict[int, int] = {}
    bad_trials = []

    for tid in np.unique(trial_id_vec):
        m = (trial_id_vec == tid)
        days_in_trial = day_vec[m]
        vals, cnts = np.unique(days_in_trial, return_counts=True)
        day_mode = int(vals[np.argmax(cnts)])
        trial_to_day[int(tid)] = day_mode
        if len(vals) > 1:
            bad_trials.append((int(tid), vals.astype(int).tolist(), cnts.tolist()))

    day_vals, day_counts = np.unique(np.array(list(trial_to_day.values()), dtype=int), return_counts=True)
    print("Trials per day (in test):", dict(zip(day_vals.tolist(), day_counts.tolist())))
    print("Total test trials:", int(np.unique(trial_id_vec).size))
    if bad_trials:
        print("WARNING: trials spanning multiple days (showing up to 10):")
        for row in bad_trials[:10]:
            print("  trial", row[0], "days", row[1], "counts", row[2])


# =========================
# Lag features (unchanged logic; keep day aligned)
# =========================
def make_lagged_features_with_day(
    X: np.ndarray,
    y: np.ndarray,
    day: np.ndarray,
    group: Optional[np.ndarray],
    n_lags: int,
    lag_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same lag stack as training, but ALSO returns day aligned to each output sample (day[t]).
    group prevents crossing boundaries (trial/day).
    """
    if n_lags <= 0:
        return X.astype(np.float32), y.astype(int), day.astype(int)

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


# =========================
# Model + weight loading (Perceptron)
# =========================
class Perceptron(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@dataclass
class WeightMeta:
    seed: int
    hs_tag: str
    out_dim: int
    n_lags: int
    lag_step: int
    lag_group: str
    scale: bool
    log_scale: bool
    scaler_mean: Optional[np.ndarray]
    scaler_scale: Optional[np.ndarray]
    target_type: Optional[str]
    label_mask: Optional[str]
    slicing_day: Optional[int]


def _maybe_none(x) -> Optional[np.ndarray]:
    # np.savez may store None as object array; be robust
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.size == 1 and x.item() is None:
            return None
        if x.size == 0:
            return None
    if (isinstance(x, np.ndarray) and x.shape == ()) and (x.item() is None):
        return None
    try:
        arr = np.array(x, dtype=float)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def load_model_and_meta(w_path: str, in_dim: int, device: torch.device) -> Tuple[Perceptron, WeightMeta]:
    d = np.load(w_path, allow_pickle=True)

    if "state_dict" not in d.files:
        raise KeyError(f"Missing state_dict in {w_path}. keys={d.files}")

    # state_dict stored as object array containing dict
    state_obj = d["state_dict"]
    state = state_obj.item() if isinstance(state_obj, np.ndarray) and state_obj.dtype == object else state_obj

    if "fc.weight" not in state:
        raise KeyError(f"Expected 'fc.weight' in state_dict for {w_path}. keys={list(state.keys())[:10]}...")

    out_dim = int(state["fc.weight"].shape[0])
    model = Perceptron(in_dim=in_dim, out_dim=out_dim).to(device)

    torch_state = {}
    for k, v in state.items():
        tv = torch.from_numpy(v)
        if tv.dtype not in (torch.float16, torch.float32, torch.float64):
            tv = tv.float()
        torch_state[k] = tv
    model.load_state_dict(torch_state, strict=True)
    model.eval()

    meta = WeightMeta(
        seed=int(d["seed"]) if "seed" in d.files else -1,
        hs_tag="perceptron",
        out_dim=out_dim,
        n_lags=int(d["n_lags"]) if "n_lags" in d.files else 0,
        lag_step=int(d["lag_step"]) if "lag_step" in d.files else 1,
        lag_group=str(d["lag_group"]) if "lag_group" in d.files else "day",
        scale=bool(d["scale"]) if "scale" in d.files else False,
        log_scale=bool(d["log_scale"]) if "log_scale" in d.files else False,
        scaler_mean=_maybe_none(d["scaler_mean"]) if "scaler_mean" in d.files else None,
        scaler_scale=_maybe_none(d["scaler_scale"]) if "scaler_scale" in d.files else None,
        target_type=(None if "target_type" not in d.files else str(d["target_type"])),
        label_mask=(None if "label_mask" not in d.files else str(d["label_mask"])),
        slicing_day=(None if "slicing_day" not in d.files else int(d["slicing_day"])),
    )
    return model, meta


def apply_saved_scaler(X: np.ndarray, meta: WeightMeta) -> np.ndarray:
    X2 = X.copy()
    if meta.scale:
        if meta.log_scale:
            X2 = np.log1p(X2)
        if meta.scaler_mean is None or meta.scaler_scale is None:
            raise ValueError("scale=True but scaler_mean/scaler_scale not saved in weight file.")
        X2 = (X2 - meta.scaler_mean) / meta.scaler_scale
    return X2.astype(np.float32)


# =========================
# Metrics
# =========================
def per_day_accuracy(day_arr: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq_days = np.unique(day_arr.astype(int))
    day_acc = np.zeros(len(uniq_days), dtype=float)
    day_n = np.zeros(len(uniq_days), dtype=int)
    for i, d in enumerate(uniq_days):
        m = (day_arr == d)
        day_n[i] = int(m.sum())
        day_acc[i] = float((y_pred[m] == y_true[m]).mean()) if day_n[i] > 0 else np.nan
    return uniq_days.astype(int), day_acc, day_n


def per_day_perclass_accuracy(
    day_arr: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# =========================
# Plotting (same as current eval: overall + per-class + slope fit)
# =========================
def plot_two_panel(
    days: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    class_ids: np.ndarray,
    mean_pc: np.ndarray,
    std_pc: np.ndarray,
    title: str,
    out_png: str,
) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: overall
    ax0.plot(days, mean, marker="o", label="overall mean (over seeds)")
    ax0.fill_between(days, mean - std, mean + std, alpha=0.2)

    mask = np.isfinite(days) & np.isfinite(mean)
    if np.sum(mask) >= 2:
        x = days[mask].astype(float)
        y = mean[mask].astype(float)
        slope, intercept = np.polyfit(x, y, deg=1)
        y_fit = slope * x + intercept
        ax0.plot(x, y_fit, color="red", linewidth=2, label=f"slope={slope:.4g}/day")

        angle_deg = float(np.degrees(np.arctan(slope)))
        mean_over_days = float(np.nanmean(y))
        ax0.text(
            0.02, 0.95,
            f"mean(acc) = {mean_over_days:.4f}\n"
            f"slope = {slope:.4g} / day\n"
            f"angle = {angle_deg:.2f}°",
            transform=ax0.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.2),
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


# =========================
# Weight discovery (match Perceptron training naming + metadata)
# =========================
def find_matching_weights(
    weights_dir: str,
    prefix: str,
    slicing_day: int,
    target_type: Optional[str],
    label_mask: Optional[str],
) -> List[str]:
    """
    Perceptron training saves weights under:
      out_dir/weights/{prefix}_perceptron_seed{seed}_Nclasses{K}_day{slicing_day}_weights.npz

    We do:
      (1) glob by prefix + day
      (2) filter by reading meta target_type/label_mask/slicing_day inside the npz
    """
    pattern = os.path.join(weights_dir, f"{prefix}_perceptron_seed*_Nclasses*_day{slicing_day}_weights.npz")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return []

    out = []
    for wp in candidates:
        d = np.load(wp, allow_pickle=True)
        tt = str(d["target_type"]) if "target_type" in d.files else None
        lm = str(d["label_mask"]) if "label_mask" in d.files else None
        sd = int(d["slicing_day"]) if "slicing_day" in d.files else None

        if sd is not None and int(sd) != int(slicing_day):
            continue
        if target_type is not None and tt != target_type:
            continue
        if label_mask is not None and str(label_mask) != str(lm):
            continue

        out.append(wp)
    return out


# =========================
# Inference
# =========================
def predict_labels(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    ys = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=False)
            logits = model(xb)
            ys.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ys, axis=0).astype(int)


# =========================
# Main
# =========================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # data (must match training masks)
    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)
    ap.add_argument("--trial_bin_path", type=str, required=True)
    ap.add_argument("--target_style_path", type=str, default=None)
    ap.add_argument("--target_type", type=str, required=True, choices=["center-out", "random"])
    ap.add_argument("--label_mask", type=str, default=None)

    # model/results root (same out_dir used for training)
    ap.add_argument("--out_dir", type=str, required=True, help="Same out_dir used for training (contains weights/results/plots)")
    ap.add_argument("--prefix", type=str, required=True, help="Same prefix used at training time")

    # slicing day index (must match training rule)
    ap.add_argument("--slicing_day", type=int, required=True, help="1-indexed index into np.unique(day_info_i) AFTER masks")

    # inference
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--max_test_samples", type=int, default=0,
                    help="If >0, evaluate only the first N test samples (before lag/scale).")

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    outp = make_outdirs(args.out_dir)

    device = get_device()
    logger.info(f"Using device: {device}")

    # -------------------------
    # (1) Load + apply masks (same as training)
    # -------------------------
    data0 = load_all_arrays(args)
    allowed = parse_label_mask(args.label_mask)
    data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=allowed)

    X = data.sbp.astype(np.float32)
    y = data.labels.astype(int)
    day_info_i = np.asarray(data.day_info).astype(int)

    # -------------------------
    # (2) slicing_day_value must match training rule
    # -------------------------
    slicing_day_value = compute_slicing_day_value(day_info_i, args.slicing_day)
    test_mask = (day_info_i > slicing_day_value)
    if not np.any(test_mask):
        raise ValueError("No samples for future test (day_info_i > slicing_day_value).")

    X_test_raw = X[test_mask]
    y_test = y[test_mask]
    day_test = day_info_i[test_mask]
    trial_bin_test = data.trial_bin[test_mask]

    logger.info(f"slicing_day_value={slicing_day_value} (idx={args.slicing_day})  testN={len(y_test)}")

    # class ids for per-class plot
    class_ids = np.array(sorted(np.unique(y_test).astype(int).tolist()), dtype=int)

    # -------------------------
    # (3) trial summary (future test only)
    # -------------------------
    trial_id_test, n_trials = build_trial_ids_from_trialbin(trial_bin_test)
    print("count_1 (trials within future-test data):", n_trials)
    summarize_trials_per_day(day_test, trial_id_test)

    # -------------------------
    # (4) Find matching weights under out_dir/weights/
    # -------------------------
    weight_files = find_matching_weights(
        weights_dir=outp.weights,
        prefix=args.prefix,
        slicing_day=args.slicing_day,
        target_type=args.target_type,
        label_mask=args.label_mask,
    )
    if not weight_files:
        raise FileNotFoundError(
            "No matching perceptron weights found.\n"
            f"Looked under: {outp.weights}\n"
            f"Pattern: {args.prefix}_perceptron_seed*_Nclasses*_day{args.slicing_day}_weights.npz\n"
            f"Filters: target_type={args.target_type}, label_mask={args.label_mask}"
        )

    logger.info(f"Found {len(weight_files)} weight files.")

    # Save eval artifacts in separated subdirs
    res_eval_dir = make_subdirs(outp.results, "evals")
    plot_eval_dir = make_subdirs(outp.plots, "evals")

    # -------------------------
    # (5) Run inference per seed-weight
    # -------------------------
    per_seed = []   # list of (seed, uniq_days, day_acc, acc_pc)
    all_days_set = set()

    # base arrays (reset each seed)
    X_base = X_test_raw
    y_base = y_test
    d_base = day_test
    trial_base = trial_id_test

    for w_path in weight_files:
        d_preview = np.load(w_path, allow_pickle=True)
        seed_preview = int(d_preview["seed"]) if "seed" in d_preview.files else -1

        # reset
        Xr = X_base
        yr = y_base
        dr = d_base
        tr = trial_base

        # optional max_test_samples BEFORE lag/scale
        if args.max_test_samples and args.max_test_samples > 0:
            n = int(args.max_test_samples)
            Xr = Xr[:n]
            yr = yr[:n]
            dr = dr[:n]
            tr = tr[:n]

        # read lag params from weight file
        lag_group = str(d_preview["lag_group"]) if "lag_group" in d_preview.files else "day"
        n_lags = int(d_preview["n_lags"]) if "n_lags" in d_preview.files else 0
        lag_step = int(d_preview["lag_step"]) if "lag_step" in d_preview.files else 1

        group = None
        if lag_group == "day":
            group = dr
        elif lag_group == "trial":
            group = tr

        # lag stack (keep day as day[t])
        Xr, yr, dr = make_lagged_features_with_day(
            X=Xr, y=yr, day=dr, group=group, n_lags=n_lags, lag_step=lag_step
        )

        if Xr.shape[0] == 0:
            logger.warning(f"[seed={seed_preview}] No samples after lag. Skip.")
            continue

        # load model + meta (scaler, etc.)
        model, meta = load_model_and_meta(w_path, in_dim=Xr.shape[1], device=device)

        # apply scaler
        X_test = apply_saved_scaler(Xr, meta)

        # predict
        y_pred = predict_labels(model, X_test, device=device, batch_size=args.batch_size)

        n_eff = min(len(y_pred), len(yr))
        if n_eff == 0:
            logger.warning(f"[seed={meta.seed}] No samples after alignment. Skip.")
            continue
        y_pred = y_pred[:n_eff]
        yr = yr[:n_eff]
        dr = dr[:n_eff]

        uniq_d, day_acc, _day_n = per_day_accuracy(dr, y_pred, yr)
        _, acc_pc, _n_pc = per_day_perclass_accuracy(dr, y_pred, yr, class_ids)

        per_seed.append((meta.seed, uniq_d, day_acc, acc_pc))
        all_days_set.update(uniq_d.tolist())

        overall = float(np.mean(y_pred == yr))
        logger.info(f"[seed={meta.seed}] overall_acc(future)={overall:.4f}  days={len(uniq_d)}")

    if not per_seed:
        raise RuntimeError("All seeds were skipped (no valid evaluation outputs).")

    # -------------------------
    # (6) Aggregate mean/std across seeds for each future day
    # -------------------------
    days_sorted = np.array(sorted(all_days_set), dtype=int)
    seeds_sorted = np.array(sorted({int(s) for (s, _, _, _) in per_seed}), dtype=int)

    A = np.full((len(seeds_sorted), len(days_sorted)), np.nan, dtype=float)
    K = int(len(class_ids))
    A_pc = np.full((len(seeds_sorted), len(days_sorted), K), np.nan, dtype=float)

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
    mean_pc = np.nanmean(A_pc, axis=0)  # (n_days, K)
    std_pc = np.nanstd(A_pc, axis=0)

    # -------------------------
    # (7) Save plot + aggregate npz (directory-separated)
    # -------------------------
    seed_tag = f"seeds{int(np.min(seeds_sorted))}-{int(np.max(seeds_sorted))}"
    base = f"{args.prefix}_perceptron_{seed_tag}_future_slicing{args.slicing_day}_value{slicing_day_value}"

    out_png = os.path.join(plot_eval_dir, f"{base}_perday_meanstd.png")
    title = (
        f"{args.target_type} | slicing_day={args.slicing_day} (value={slicing_day_value}) | "
        f"seeds={len(seeds_sorted)} | hs=perceptron"
    )
    plot_two_panel(days_sorted, mean_acc, std_acc, class_ids, mean_pc, std_pc, title=title, out_png=out_png)

    out_npz = os.path.join(res_eval_dir, f"{base}_perday_meanstd.npz")
    np.savez(
        out_npz,
        target_type=args.target_type,
        label_mask=args.label_mask,
        slicing_day_idx=int(args.slicing_day),
        slicing_day_value=int(slicing_day_value),
        seeds=seeds_sorted,
        days=days_sorted,
        perday_acc_matrix=A,
        perday_acc_perclass=A_pc,
        mean=mean_acc,
        std=std_acc,
        mean_pc=mean_pc,
        std_pc=std_pc,
        class_ids=class_ids,
    )

    print("saved:")
    print(" ", out_png)
    print(" ", out_npz)


if __name__ == "__main__":
    main()
