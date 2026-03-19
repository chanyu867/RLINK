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
from eval_metrics import (
    per_day_balanced_accuracy,
    per_day_f1_macro,
    balanced_accuracy_macro,   # NEW
    f1_macro,                  # NEW
)

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

### plot performance for each class
def plot_perclass_accuracy(
    days: np.ndarray,
    mean_pc: np.ndarray,
    std_pc: np.ndarray,
    class_ids: np.ndarray,
    title: str,
    out_png: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a line for each class
    for k, c_id in enumerate(class_ids):
        m = mean_pc[:, k]
        s = std_pc[:, k]
        ax.plot(days, m, marker="o", label=f"Class {int(c_id)}")
        ax.fill_between(days, m - s, m + s, alpha=0.15)

    ax.set_xlabel("Day (day_info value)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", title="Classes")

    # Set x-ticks to match the days
    xt = list(days.astype(int).tolist())
    ax.set_xticks(xt)
    ax.set_xticklabels([str(v) for v in xt], rotation=45, ha="right")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)




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
    """
    Must match the cleaned training script:
      '0,1,2' -> np.array([0,1,2])
    """
    if s is None or str(s).strip() == "":
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    return np.array([int(p) for p in parts], dtype=int)


def parse_hidden_sizes_from_npz(arr: np.ndarray) -> List[int]:
    if arr is None:
        return []
    a = np.array(arr).astype(int).reshape(-1)
    return [int(x) for x in a.tolist()]


def hidden_tag(hidden: List[int]) -> str:
    return "hsnone" if len(hidden) == 0 else "hs" + "-".join(map(str, hidden))

def load_split_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    need = ["test_trials", "slicing_day_value"]
    for k in need:
        if k not in z.files:
            raise ValueError(f"split_npz missing key '{k}': {path}. keys={z.files}")
    return {k: z[k] for k in z.files}


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
    sbp: np.ndarray                 # (N, D)
    labels: np.ndarray              # (N,)
    day_info: np.ndarray            # (N,)
    trial_bin: np.ndarray           # (N,) float, 0.0 at trial start
    target_style: Optional[np.ndarray]  # (N,) bool or None


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
        target_style = npy_loader(args.target_style_path)
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
    EXACT same logic as cleaned training script:
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

    sbp = data.sbp[keep]
    labels = data.labels[keep]
    day_info = data.day_info[keep]
    trial_bin = data.trial_bin[keep]
    target_style = (np.asarray(data.target_style).astype(bool)[keep] if data.target_style is not None else None)

    return LoadedData(sbp=sbp, labels=labels, day_info=day_info, trial_bin=trial_bin, target_style=target_style)


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
    Same logic as your current eval:
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
    Same as training lag stack, but ALSO returns day aligned to each output sample (day[t]).
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
# Model + weight loading
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class WeightMeta:
    seed: int
    hidden: List[int]
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


def load_model_and_meta(w_path: str, in_dim: int, device: torch.device) -> Tuple[MLP, WeightMeta]:
    d = np.load(w_path, allow_pickle=True)

    if "state_dict" not in d.files or "hidden_sizes" not in d.files:
        raise KeyError(f"Missing state_dict/hidden_sizes in {w_path}. keys={d.files}")

    # state_dict stored as object array containing dict
    state_obj = d["state_dict"]
    state = state_obj.item() if isinstance(state_obj, np.ndarray) and state_obj.dtype == object else state_obj

    # infer out_dim from last layer weight (same logic as current eval)
    last_w_key = None
    for k in state.keys():
        if k.endswith(".weight"):
            last_w_key = k
    if last_w_key is None:
        raise ValueError(f"Could not find a .weight key in state_dict for {w_path}")
    out_dim = int(state[last_w_key].shape[0])

    hidden = parse_hidden_sizes_from_npz(d["hidden_sizes"])
    hs = hidden_tag(hidden)

    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    torch_state = {k: torch.from_numpy(v) for k, v in state.items()}
    model.load_state_dict(torch_state, strict=True)
    model.eval()

    meta = WeightMeta(
        seed=int(d["seed"]) if "seed" in d.files else -1,
        hidden=hidden,
        hs_tag=hs,
        out_dim=out_dim,
        n_lags=int(d["n_lags"]) if "n_lags" in d.files else 0,
        lag_step=int(d["lag_step"]) if "lag_step" in d.files else 1,
        lag_group=str(d["lag_group"]) if "lag_group" in d.files else "day",
        scale=bool(d["scale"]) if "scale" in d.files else False,
        log_scale=bool(d["log_scale"]) if "log_scale" in d.files else False,
        scaler_mean=(None if ("scaler_mean" not in d.files or d["scaler_mean"] is None) else np.array(d["scaler_mean"], dtype=float)),
        scaler_scale=(None if ("scaler_scale" not in d.files or d["scaler_scale"] is None) else np.array(d["scaler_scale"], dtype=float)),
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
# def plot_two_panel( #-> plot only performance change over days
#     days: np.ndarray,
#     mean: np.ndarray,
#     std: np.ndarray,
#     class_ids: np.ndarray,
#     mean_pc: np.ndarray,
#     std_pc: np.ndarray,
#     title: str,
#     out_png: str,
#     # NEW: baseline point (held-out test within training window)
#     baseline_x: Optional[int] = None,
#     baseline_mean: Optional[float] = None,
#     baseline_std: Optional[float] = None,
#     baseline_label: str = "held-out(test, train-window)",
# ) -> None:
#     fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Top: overall (future days)
#     ax0.plot(days, mean, marker="o", label="future mean (over seeds)")
#     ax0.fill_between(days, mean - std, mean + std, alpha=0.2)

#     # NEW: red dot for day1 held-out test
#     if baseline_x is not None and baseline_mean is not None:
#         if baseline_std is None:
#             ax0.scatter([baseline_x], [baseline_mean], color="red", zorder=5, label=baseline_label)
#         else:
#             ax0.errorbar(
#                 [baseline_x], [baseline_mean],
#                 yerr=[baseline_std],
#                 fmt="o", color="red", capsize=3, zorder=5, label=baseline_label
#             )

#     # slope fit only on future days (keep original behavior)
#     mask = np.isfinite(days) & np.isfinite(mean)
#     if np.sum(mask) >= 2:
#         x = days[mask].astype(float)
#         y = mean[mask].astype(float)
#         slope, intercept = np.polyfit(x, y, deg=1)
#         y_fit = slope * x + intercept
#         ax0.plot(x, y_fit, color="red", linewidth=2, label=f"slope={slope:.4g}/day")

#         angle_deg = float(np.degrees(np.arctan(slope)))
#         mean_over_days = float(np.nanmean(y))
#         ax0.text(
#             0.02, 0.95,
#             f"mean(acc) = {mean_over_days:.4f}\n"
#             f"slope = {slope:.4g} / day\n"
#             f"angle = {angle_deg:.2f}°",
#             transform=ax0.transAxes,
#             va="top", ha="left",
#             bbox=dict(boxstyle="round", alpha=0.2),
#         )

#     ax0.set_ylabel("Accuracy")
#     ax0.set_ylim(0, 1)
#     ax0.grid(True, alpha=0.2)
#     ax0.legend()

#     # Bottom: per-class (future days only)
#     for k, c in enumerate(class_ids):
#         m = mean_pc[:, k]
#         s = std_pc[:, k]
#         ax1.plot(days, m, marker="o", label=f"class {int(c)}")
#         ax1.fill_between(days, m - s, m + s, alpha=0.15)

#     ax1.set_xlabel("Day (day_info value)")
#     ax1.set_ylabel("Accuracy")
#     ax1.set_ylim(0, 1)
#     ax1.grid(True, alpha=0.2)
#     ax1.legend()

#     # NEW: explicit x tick labels including baseline day
#     xt = list(days.astype(int).tolist())
#     if baseline_x is not None:
#         xt = sorted(set(xt + [int(baseline_x)]))
#     ax1.set_xticks(xt)
#     ax1.set_xticklabels([str(v) for v in xt], rotation=45, ha="right")

#     fig.suptitle(title)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     fig.savefig(out_png, dpi=200)
#     plt.close(fig)


def _fit_slope_with_baseline(
    days: np.ndarray,
    mean: np.ndarray,
    baseline_x: Optional[int],
    baseline_mean: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    # build x,y including baseline
    xs = []
    ys = []
    m = np.isfinite(days) & np.isfinite(mean)
    xs.extend(days[m].astype(float).tolist())
    ys.extend(mean[m].astype(float).tolist())

    if baseline_x is not None and baseline_mean is not None and np.isfinite(baseline_mean):
        xs.append(float(baseline_x))
        ys.append(float(baseline_mean))

    if len(xs) < 2:
        return None, None, None, None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept), x, (slope * x + intercept)


def plot_three_metrics(
    *,
    days: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    mean_bacc: np.ndarray,
    std_bacc: np.ndarray,
    mean_f1: np.ndarray,
    std_f1: np.ndarray,
    title: str,
    out_png: str,
    baseline_x: Optional[int] = None,
    baseline_acc_mean: Optional[float] = None,
    baseline_acc_std: Optional[float] = None,
    baseline_bacc_mean: Optional[float] = None,
    baseline_bacc_std: Optional[float] = None,
    baseline_f1_mean: Optional[float] = None,
    baseline_f1_std: Optional[float] = None,
    baseline_label: str = "train-day held-out (from split_npz)",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    def _panel(ax, y_mean, y_std, ylab, b_mean, b_std):
        ax.plot(days, y_mean, marker="o", label="future mean (over seeds)")
        ax.fill_between(days, y_mean - y_std, y_mean + y_std, alpha=0.2)

        # baseline red point
        if baseline_x is not None and b_mean is not None:
            if b_std is None or (not np.isfinite(b_std)):
                ax.scatter([baseline_x], [b_mean], color="red", zorder=5, label=baseline_label)
            else:
                ax.errorbar([baseline_x], [b_mean], yerr=[b_std], fmt="o", color="red", capsize=3, zorder=5, label=baseline_label)

        # slope INCLUDING baseline point
        slope, intercept, xfit, yfit = _fit_slope_with_baseline(days, y_mean, baseline_x, b_mean)
        if slope is not None and xfit is not None and yfit is not None:
            ax.plot(xfit, yfit, color="red", linewidth=2, label=f"slope={slope:.4g}/day")
            angle_deg = float(np.degrees(np.arctan(slope)))
            ax.text(
                0.02, 0.95,
                f"slope = {slope:.4g} / day\nangle = {angle_deg:.2f}°",
                transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(boxstyle="round", alpha=0.2),
            )

        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")

    _panel(axes[0], mean_acc, std_acc, "Accuracy", baseline_acc_mean, baseline_acc_std)
    _panel(axes[1], mean_bacc, std_bacc, "Balanced Accuracy", baseline_bacc_mean, baseline_bacc_std)
    _panel(axes[2], mean_f1, std_f1, "Macro-F1", baseline_f1_mean, baseline_f1_std)

    all_x = list(days.astype(int).tolist())
    if baseline_x is not None:
        all_x.append(int(baseline_x))
    all_x = sorted(set(all_x))

    x_min = int(np.min(all_x))
    x_max = int(np.max(all_x))

    # ticks at numeric interval = 3
    xt = list(np.arange(x_min, x_max + 1, 3, dtype=int))

    # make sure baseline_x is visible as a tick even if not aligned to step=3
    if baseline_x is not None and int(baseline_x) not in xt:
        xt.append(int(baseline_x))
        xt = sorted(set(xt))

    # apply the same ticks/labels to ALL three axes
    for ax in axes:
        ax.set_xlabel("Day (day_info value)")  # xlabel on every subplot
        ax.set_xticks(xt)
        ax.set_xticklabels([str(v) for v in xt], rotation=45, ha="right")
        ax.tick_params(labelbottom=True)       # show tick labels even on upper axes (sharex=True)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================
# Weight discovery (match new training naming + metadata)
# =========================
def find_matching_weights(
    weights_dir: str,
    prefix: str,
    slicing_day: int,
    target_type: Optional[str],
    label_mask: Optional[str],
) -> List[str]:
    """
    New training saves weights under:
      out_dir/weights/{prefix}_{hs_tag}_seed{seed}_Nclasses{K}_day{slicing_day}_{maybe something depending on the mode}weights.npz

    We do:
      (1) glob by prefix + day
      (2) filter by reading meta target_type/label_mask/slicing_day inside the npz
    """
    pattern = os.path.join(weights_dir, f"{prefix}_*_seed*_Nclasses*_day{slicing_day}_*weights.npz")
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
        for xb, in loader:
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

    # model/results root (same as training out_dir)
    ap.add_argument("--out_dir", type=str, required=True, help="Same out_dir used for training (contains weights/results/plots)")
    ap.add_argument("--prefix", type=str, required=True, help="Same prefix used at training time")

    # slicing day index (must match training rule)
    ap.add_argument("--slicing_day", type=int, required=True, help="1-indexed index into np.unique(day_info_i) AFTER masks")

    # inference
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--max_test_samples", type=int, default=0,
                    help="If >0, evaluate only the first N test samples (before lag/scale).")
    

    ap.add_argument(
        "--split_npz",
        type=str,
        default=None,
        help="split_indices.npz (from mlp_hpo.py). If given, we also evaluate held-out test within the training window "
             "(<= slicing_day_value) and plot it as a red dot at x=slicing_day_value.",
    )


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
    data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=None)

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

    logger.info(f"slicing_day_value={slicing_day_value}-th day (idx={args.slicing_day})  testN={len(y_test)}")

    # class ids (for per-class plot) -- identical logic
    if allowed is not None:
        class_ids = np.arange(len(allowed), dtype=int) # e.g., creates [0, 1]
    else:
        class_ids = np.array(sorted(np.unique(y_test).astype(int).tolist()), dtype=int)

    # -------------------------
    # (2.5) Also prepare TRAIN-WINDOW subset (<= slicing_day_value) for day1 held-out test
    # -------------------------
    trainwin_mask = (day_info_i <= slicing_day_value)
    if args.split_npz is not None:
        if not np.any(trainwin_mask):
            raise ValueError("No samples in train-window (day_info_i <= slicing_day_value).")

        X_tw_raw = X[trainwin_mask]
        y_tw = y[trainwin_mask]
        day_tw = day_info_i[trainwin_mask]
        trial_bin_tw = data.trial_bin[trainwin_mask]

        trial_id_tw, _ = build_trial_ids_from_trialbin(trial_bin_tw)

        split = load_split_npz(args.split_npz)
        split_sdv = int(split["slicing_day_value"])
        if split_sdv != int(slicing_day_value):
            raise ValueError(
                f"split_npz slicing_day_value mismatch: split={split_sdv} vs current={int(slicing_day_value)}. "
                "Use the SAME slicing_day and SAME masks (target_type/label_mask) as used in HPO."
            )

        test_trials = split["test_trials"].astype(int)
        # held-out test within train-window: samples belonging to test_trials (trial-wise, safe for lag)
        heldout_mask = np.isin(trial_id_tw.astype(int), test_trials)

        if not np.any(heldout_mask):
            raise ValueError("No held-out (day1) test samples found from split_npz test_trials.")

        X_hold_raw = X_tw_raw[heldout_mask]
        y_hold = y_tw[heldout_mask]
        day_hold = day_tw[heldout_mask]
        trial_hold = trial_id_tw[heldout_mask]


    # -------------------------
    # (3) trial summary (future test only)
    # -------------------------
    trial_id_test, n_trials = build_trial_ids_from_trialbin(trial_bin_test)
    print("count_1 (trials within future-test data):", n_trials)
    summarize_trials_per_day(day_test, trial_id_test)

    # -------------------------
    # (4) Find matching weight files under out_dir/weights/
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
            "No matching weights found.\n"
            f"Looked under: {outp.weights}\n"
            f"Pattern: {args.prefix}_*_seed*_Nclasses*_day{args.slicing_day}_weights.npz\n"
            f"Filters: target_type={args.target_type}, label_mask={args.label_mask}"
        )

    logger.info(f"Found {len(weight_files)} weight files.")

    # Save eval artifacts in separated subdirs (new directories, no logic change)
    res_eval_dir = make_subdirs(outp.results, "evals")
    plot_eval_dir = make_subdirs(outp.plots, "evals")

    # -------------------------
    # (5) Run inference per seed-weight
    # -------------------------
    per_seed = []   # list of (seed, uniq_days, day_acc, acc_pc)
    heldout_acc_by_seed: Dict[int, float] = {}
    heldout_bacc_by_seed: Dict[int, float] = {}
    heldout_f1_by_seed: Dict[int, float] = {}
    all_days_set = set()

    # base arrays (important: reset every seed)
    X_base = X_test_raw
    y_base = y_test
    d_base = day_test
    trial_base = trial_id_test

    for w_path in weight_files:
        model_meta_preview = np.load(w_path, allow_pickle=True)
        seed = int(model_meta_preview["seed"]) if "seed" in model_meta_preview.files else -1


        logger.info(f"Loading weights from: {w_path}")

        # start from base every seed
        Xr = X_base
        yr = y_base
        dr = d_base
        tr = trial_base

        # (optional) max_test_samples BEFORE lag/scale (same behavior)
        if args.max_test_samples and args.max_test_samples > 0:
            n = int(args.max_test_samples)
            Xr = Xr[:n]
            yr = yr[:n]
            dr = dr[:n]
            tr = tr[:n]
            logger.info(f"[seed={seed}] max_test_samples={n}, yr unique counts: {dict(zip(*np.unique(yr, return_counts=True)))}")

        # Read meta for lag/scaler (consistent with weight)
        # (We still load the model after lag so input_dim matches.)
        w_npz = np.load(w_path, allow_pickle=True)
        lag_group = str(w_npz["lag_group"]) if "lag_group" in w_npz.files else "day"
        n_lags = int(w_npz["n_lags"]) if "n_lags" in w_npz.files else 0
        lag_step = int(w_npz["lag_step"]) if "lag_step" in w_npz.files else 1

        # group for boundary prevention
        group = None
        if lag_group == "day":
            group = dr
        elif lag_group == "trial":
            group = tr  # trial id aligned with (possibly sliced) arrays

        # lag stack (day must remain day, not group-id)
        Xr, yr, dr = make_lagged_features_with_day(
            X=Xr, y=yr, day=dr, group=group, n_lags=n_lags, lag_step=lag_step
        )

        if allowed is not None:
            valid_idx = np.isin(yr, allowed)
            Xr = Xr[valid_idx]
            yr = yr[valid_idx]
            dr = dr[valid_idx]
            
        if Xr.shape[0] == 0:
            logger.warning(f"[seed={seed}] No samples after lag and mask. Skip.")
            continue
            
        # Remap labels so [0, 2] becomes [0, 1] to match the model's output
        if allowed is not None:
            unique_labels = np.sort(np.unique(yr))
            remapper = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
            yr = np.vectorize(remapper.get)(yr)

        # load model + meta (scaler params etc.)
        model, meta = load_model_and_meta(w_path, in_dim=Xr.shape[1], device=device)

        # apply saved scaler (training stats)
        X_test = apply_saved_scaler(Xr, meta)

        # predict
        y_pred = predict_labels(model, X_test, device=device, batch_size=args.batch_size)

        n_eff = min(len(y_pred), len(yr))
        if n_eff == 0:
            logger.warning(f"[seed={seed}] No samples after predict alignment. Skip.")
            continue

        y_pred = y_pred[:n_eff]
        yr = yr[:n_eff]
        dr = dr[:n_eff]

        # per-day / per-class (must use dr/yr aligned after lag)
        uniq_d, day_acc, _day_n = per_day_accuracy(dr, y_pred, yr)
        _, acc_pc, _n_pc = per_day_perclass_accuracy(dr, y_pred, yr, class_ids)

        # NEW: per-day balanced accuracy and macro-F1
        uniq_d_b, day_bacc, _ = per_day_balanced_accuracy(dr, y_pred, yr, class_ids)
        uniq_d_f, day_f1, _ = per_day_f1_macro(dr, y_pred, yr, class_ids)

        # sanity: day order should match
        if not np.array_equal(uniq_d, uniq_d_b) or not np.array_equal(uniq_d, uniq_d_f):
            raise RuntimeError("per-day day indexing mismatch among metrics.")

        if len(uniq_d) == 0:
            logger.warning(f"[seed={seed}] per-day arrays empty. Skip.")
            continue

        per_seed.append((meta.seed, meta.hs_tag, uniq_d, day_acc, day_bacc, day_f1, acc_pc))
        all_days_set.update(uniq_d.tolist())

        overall = float(np.mean(y_pred == yr))
        logger.info(f"[seed={meta.seed}] overall_acc(future)={overall:.4f}  days={len(uniq_d)}")

        # NEW: evaluate held-out test within training day if split_npz is given
        if args.split_npz is not None:
            Xh = X_hold_raw
            yh = y_hold
            dh = day_hold
            th = trial_hold

            # apply the SAME max_test_samples rule (optional) BEFORE lag/scale
            if args.max_test_samples and args.max_test_samples > 0:
                n = int(args.max_test_samples)
                Xh = Xh[:n]
                yh = yh[:n]
                dh = dh[:n]
                th = th[:n]

            # group boundary prevention (must match weight meta)
            group_h = None
            if lag_group == "day":
                group_h = dh
            elif lag_group == "trial":
                group_h = th

            Xh, yh, dh = make_lagged_features_with_day(
                X=Xh, y=yh, day=dh, group=group_h, n_lags=n_lags, lag_step=lag_step
            )
            
            # NEW: FILTER AND REMAP HELDOUT DATA
            if allowed is not None:
                valid_idx_h = np.isin(yh, allowed)
                Xh = Xh[valid_idx_h]
                yh = yh[valid_idx_h]
                dh = dh[valid_idx_h]

            if Xh.shape[0] > 0:
                if allowed is not None:
                    yh = np.vectorize(remapper.get)(yh)
                    
                Xh = apply_saved_scaler(Xh, meta)
                yph = predict_labels(model, Xh, device=device, batch_size=args.batch_size)
                n_eff_h = min(len(yph), len(yh))

                if n_eff_h <= 0:
                    acc_h = np.nan
                    bacc_h = np.nan
                    f1_h = np.nan
                else:
                    yph = yph[:n_eff_h]
                    yh_eff = yh[:n_eff_h]
                    dh_eff = dh[:n_eff_h]

                    m_day = (dh_eff == int(slicing_day_value))
                    if np.any(m_day):
                        yt0 = yh_eff[m_day]
                        yp0 = yph[m_day]
                    else:
                        yt0 = yh_eff
                        yp0 = yph

                    acc_h = float(np.mean(yp0 == yt0))
                    bacc_h = float(balanced_accuracy_macro(yt0, yp0, class_ids))
                    f1_h = float(f1_macro(yt0, yp0, class_ids))

                heldout_acc_by_seed[int(meta.seed)] = acc_h
                heldout_bacc_by_seed[int(meta.seed)] = bacc_h
                heldout_f1_by_seed[int(meta.seed)] = f1_h

                logger.info(
                    f"[seed={meta.seed}] heldout(train-day) acc={acc_h:.4f} bAcc={bacc_h:.4f} F1={f1_h:.4f}"
                )



    if not per_seed:
        raise RuntimeError("All seeds were skipped (no valid evaluation outputs).")

    # -------------------------
    # (6) Aggregate mean/std across seeds for each future day
    # -------------------------
    days_sorted = np.array(sorted(all_days_set), dtype=int)
    seeds_sorted = np.array(sorted({int(t[0]) for t in per_seed}), dtype=int)

    # assume all seeds share the same hs_tag; if not, we still pick the first for naming
    hs_tag_out = per_seed[0][1]

    A = np.full((len(seeds_sorted), len(days_sorted)), np.nan, dtype=float)       # acc
    A_b = np.full((len(seeds_sorted), len(days_sorted)), np.nan, dtype=float)     # bAcc
    A_f = np.full((len(seeds_sorted), len(days_sorted)), np.nan, dtype=float)     # F1
    K = int(len(class_ids))
    A_pc = np.full((len(seeds_sorted), len(days_sorted), K), np.nan, dtype=float)

    seed_to_row = {int(s): i for i, s in enumerate(seeds_sorted)}
    day_to_col = {int(d): j for j, d in enumerate(days_sorted)}

    for seed, _hs, dlist, alist, blist, flist, acc_pc in per_seed:
        r = seed_to_row[int(seed)]
        for i, d in enumerate(dlist):
            c = day_to_col[int(d)]
            A[r, c] = float(alist[i])
            A_b[r, c] = float(blist[i])
            A_f[r, c] = float(flist[i])
            A_pc[r, c, :] = acc_pc[i, :]

    mean_acc = np.nanmean(A, axis=0);    std_acc = np.nanstd(A, axis=0)
    mean_bacc = np.nanmean(A_b, axis=0); std_bacc = np.nanstd(A_b, axis=0)
    mean_f1 = np.nanmean(A_f, axis=0);   std_f1 = np.nanstd(A_f, axis=0)
    std_acc = np.nanstd(A, axis=0)
    mean_pc = np.nanmean(A_pc, axis=0)  # (n_days, K)
    std_pc = np.nanstd(A_pc, axis=0)

    # -------------------------
    # (7) Save plot + aggregate npz (directory-separated)
    # -------------------------
    seed_tag = f"seeds{int(np.min(seeds_sorted))}-{int(np.max(seeds_sorted))}"
    base = f"{args.prefix}_{hs_tag_out}_{seed_tag}_future_slicing{args.slicing_day}_value{slicing_day_value}"

    out_png = os.path.join(plot_eval_dir, f"{base}_perday_meanstd.png")
    title = (
        f"{args.target_type} | slicing_day={args.slicing_day} (value={slicing_day_value}) | "
        f"seeds={len(seeds_sorted)} | hs={hs_tag_out}"
    )
    baseline_x = None

    b_acc_mean = b_acc_std = None
    b_bacc_mean = b_bacc_std = None
    b_f1_mean = b_f1_std = None

    if args.split_npz is not None:
        baseline_x = int(slicing_day_value)

        def _mean_std_from_dict(dct: Dict[int, float]) -> Tuple[Optional[float], Optional[float]]:
            vals = []
            for s in seeds_sorted.tolist():
                v = dct.get(int(s), np.nan)
                if np.isfinite(v):
                    vals.append(float(v))
            if len(vals) == 0:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))

        b_acc_mean, b_acc_std = _mean_std_from_dict(heldout_acc_by_seed)
        b_bacc_mean, b_bacc_std = _mean_std_from_dict(heldout_bacc_by_seed)
        b_f1_mean, b_f1_std = _mean_std_from_dict(heldout_f1_by_seed)

    plot_three_metrics(
        days=days_sorted,
        mean_acc=mean_acc, std_acc=std_acc,
        mean_bacc=mean_bacc, std_bacc=std_bacc,
        mean_f1=mean_f1, std_f1=std_f1,
        title=title,
        out_png=out_png,
        baseline_x=baseline_x,
        baseline_acc_mean=b_acc_mean, baseline_acc_std=b_acc_std,
        baseline_bacc_mean=b_bacc_mean, baseline_bacc_std=b_bacc_std,
        baseline_f1_mean=b_f1_mean, baseline_f1_std=b_f1_std,
        baseline_label="train-day held-out (from split_npz)",
    )

    # NEW: Plot per-class accuracy
    out_png_pc = os.path.join(plot_eval_dir, f"{base}_perclass_meanstd.png")
    plot_perclass_accuracy(
        days=days_sorted,
        mean_pc=mean_pc,
        std_pc=std_pc,
        class_ids=class_ids,
        title=f"Per-Class Accuracy | {title}",
        out_png=out_png_pc
    )


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
        perday_bacc_matrix=A_b,
        perday_f1_matrix=A_f,
        mean_bacc=mean_bacc,
        std_bacc=std_bacc,
        mean_f1=mean_f1,
        std_f1=std_f1
    )

    print("saved:")
    print(" ", out_png)
    print(" ", out_png_pc)
    print(" ", out_npz)


if __name__ == "__main__":
    main()
