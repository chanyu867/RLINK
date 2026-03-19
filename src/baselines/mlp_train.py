#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import json

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =========================
# Small Utilities
# =========================
def npy_loader(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_label_mask(s: Optional[str]) -> Optional[np.ndarray]:
    """Parse label list like '0,1,2' -> np.array([0,1,2])."""
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([int(p) for p in parts], dtype=int)


def parse_hidden_sizes(s: str) -> List[int]:
    """Parse hidden sizes like '256,128' -> [256,128]."""
    s = (s or "").strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def hidden_tag(hidden: List[int]) -> str:
    """Tag for filenames, e.g., [256,128] -> 'hs256-128', [] -> 'hsnone'."""
    return "hsnone" if len(hidden) == 0 else "hs" + "-".join(map(str, hidden))


def load_split_npz(path: str) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=False)
    need = ["train_trials", "test_trials", "slicing_day_value"]
    for k in need:
        if k not in obj:
            raise ValueError(f"split_npz missing key '{k}': {path}")
    return {k: obj[k] for k in obj.files}


# =========================
# Output Directories
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


# =========================
# Plotting
# =========================
def plot_label_distribution(y: np.ndarray, out_png: str, n_classes: int) -> None:
    classes, counts = np.unique(y.astype(int), return_counts=True)
    plt.figure()
    plt.bar(classes.astype(str), counts)
    plt.xlabel("Class label")
    plt.ylabel("Count (train)")
    plt.title(f"Train label distribution (Nclasses={n_classes})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_per_group_mean_std(
    groups: np.ndarray, mean_acc: np.ndarray, std_acc: np.ndarray, n_classes: int, out_png: str, group_name: str
) -> None:
    plt.figure()
    plt.plot(groups, mean_acc, marker="o")
    plt.fill_between(groups, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    plt.xlabel(group_name)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"test {group_name} accuracy mean±std over seeds (Nclasses={n_classes})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Metrics
# =========================
def per_group_accuracy(group_vec: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute accuracy per unique group (trial/day)."""
    uniq = np.unique(group_vec)
    acc = []
    n = []
    for g in uniq:
        idx = (group_vec == g)
        n.append(int(idx.sum()))
        acc.append(float((y_pred[idx] == y_true[idx]).mean()))
    return uniq, np.asarray(acc, dtype=float), np.asarray(n, dtype=int)


# =========================
# Lag Feature Construction
# =========================
def make_lagged_features(
    X: np.ndarray,
    y: np.ndarray,
    group: Optional[np.ndarray],
    n_lags: int,
    lag_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    X: (N, D)
    y: (N,)
    group: (N,) grouping id to avoid crossing (e.g., day/trial). If None, treat as one group.
    Returns:
      X_lag: (N_eff, D*(n_lags+1)) = [x[t], x[t-lag_step], ..., x[t-n_lags*lag_step]]
      y_eff: (N_eff,) aligned with x[t]
      group_eff: (N_eff,) aligned with x[t]
    """
    if n_lags <= 0:
        return X, y, group
    if lag_step < 1:
        raise ValueError("lag_step must be >= 1")

    N, D = X.shape
    if group is None:
        group = np.zeros(N, dtype=int)

    X_out, y_out, g_out = [], [], []

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
            g_out.append(g)

    X_lag = np.stack(X_out, axis=0).astype(np.float32)
    y_eff = np.asarray(y_out, dtype=int)
    g_eff = np.asarray(g_out, dtype=int)
    return X_lag, y_eff, g_eff


# =========================
# Model
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


# =========================
# Data Pipeline (load -> mask -> slice -> trial ids)
# =========================
@dataclass
class LoadedData:
    sbp: np.ndarray
    labels: np.ndarray
    day_info: np.ndarray
    trial_bin: np.ndarray
    target_style: Optional[np.ndarray]


def load_all_arrays(args) -> LoadedData:
    sbp = npy_loader(args.sbp_path)
    labels = npy_loader(args.label_path)
    day_info = npy_loader(args.day_info_path)
    trial_bin = npy_loader(args.trial_bin_path)

    if trial_bin.shape[0] != sbp.shape[0]:
        raise ValueError("Length mismatch: trial_bin vs sbp")
    if sbp.shape[0] != labels.shape[0] or sbp.shape[0] != day_info.shape[0]:
        raise ValueError("Length mismatch among sbp/labels/day_info")

    target_style = None
    if args.target_style_path is not None:
        target_style = npy_loader(args.target_style_path)
        if target_style.shape[0] != sbp.shape[0]:
            raise ValueError("Length mismatch: target_style vs sbp")
        target_style = np.asarray(target_style).astype(bool)

    return LoadedData(sbp=sbp, labels=labels, day_info=day_info, trial_bin=trial_bin, target_style=target_style)


def apply_task_masks(data: LoadedData, target_type: Optional[str], allowed_labels: Optional[np.ndarray]) -> LoadedData:
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
    uniq_all_days = np.unique(day_info_i)
    if slicing_day_index_1based < 1 or slicing_day_index_1based > len(uniq_all_days):
        raise ValueError(f"--slicing_day must be in [1, {len(uniq_all_days)}]")
    return int(uniq_all_days[slicing_day_index_1based - 1])


def build_trial_ids(trial_bin_used: np.ndarray) -> Tuple[np.ndarray, int]:
    trial_starts = (trial_bin_used == 0.0)
    if len(trial_bin_used) > 0 and (trial_bin_used[0] != 0.0):
        trial_starts[0] = True
    trial_id = np.cumsum(trial_starts).astype(int) - 1
    n_trials = int(trial_starts.sum())
    return trial_id, n_trials


def choose_lag_group(lag_group: str, day_vec: np.ndarray, trial_id_vec: np.ndarray) -> Optional[np.ndarray]:
    if lag_group == "none":
        return None
    if lag_group == "day":
        return day_vec
    if lag_group == "trial":
        return trial_id_vec
    raise ValueError("lag_group must be one of: none, day, trial")


# =========================
# Train/Eval Helpers
# =========================
def preprocess_train_val(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    scale: bool,
    log_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    scaler = None
    X_train = X_train_raw
    X_val = X_val_raw

    if scale:
        if log_scale:
            X_train = np.log1p(X_train)
            X_val = np.log1p(X_val)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)

    return X_train, X_val, scaler


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
    seed: int,
) -> None:
    
    # classes, counts = np.unique(y_train, return_counts=True)
    # total_samples = len(y_train)

    # weights = []
    # for count in counts:
    #     weight = total_samples / (len(classes) * count)
    #     weights.append(weight)

    # class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    for ep in range(1, epochs + 1):
        print(f"[seed={seed}] Epoch {ep}/{epochs}")
        model.train()
        total_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            bs = int(xb.shape[0])
            total_loss += float(loss.item()) * bs
            n_seen += bs

        if ep == 1 or ep % print_every == 0 or ep == epochs:
            logger.info(f"[seed={seed}] epoch {ep:3d}/{epochs} loss={total_loss/max(n_seen,1):.6f}")


def predict_labels(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=batch_size,
        shuffle=False,
    )

    y_pred_list = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=False)
            logits = model(xb)
            y_pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())

    return np.concatenate(y_pred_list, axis=0).astype(int)


# =========================
# Saving
# =========================
def save_weights_npz(
    out_weights_dir: str,
    base_name: str,
    model: nn.Module,
    seed: int,
    args,
    n_classes: int,
    slicing_day_value: int,
    hidden: List[int],
    scaler: Optional[StandardScaler],
) -> str:
    w_path = os.path.join(out_weights_dir, f"{base_name}.npz")
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        w_path,
        seed=seed,
        state_dict=np.array(state, dtype=object),
        slicing_day=args.slicing_day,
        slicing_day_value=int(slicing_day_value),
        target_type=args.target_type,
        label_mask=args.label_mask,
        scale=args.scale,
        log_scale=args.log_scale,
        hidden_sizes=np.array(hidden, dtype=int),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        train_ratio=float(args.train_ratio),
        scaler_mean=(scaler.mean_ if scaler is not None else None),
        scaler_scale=(scaler.scale_ if scaler is not None else None),
        n_lags=int(args.n_lags),
        lag_step=int(args.lag_step),
        lag_group=str(args.lag_group),
        n_classes=int(n_classes),
        split_npz=(args.split_npz if args.split_npz is not None else ""),
    )
    return w_path


def save_test_outputs(
    out_results_dir: str,
    base_name: str,
    correct01: np.ndarray,
    pergroup_groups: np.ndarray,
    pergroup_acc: np.ndarray,
    pergroup_n: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    group_of_each_sample: np.ndarray,
    seed: int,
) -> Tuple[str, str]:
    perf_path = os.path.join(out_results_dir, f"{base_name}_test_correct01.npy")
    np.save(perf_path, correct01)

    group_path = os.path.join(out_results_dir, f"{base_name}_pertrial_testacc.npz")
    np.savez(
        group_path,
        seed=seed,
        groups=pergroup_groups.astype(int),
        acc=pergroup_acc.astype(float),
        n=pergroup_n.astype(int),
        y_pred=y_pred.astype(np.int16),
        y_true=y_true.astype(np.int16),
        group_of_each_sample=group_of_each_sample.astype(np.int32),
    )
    return perf_path, group_path


def save_aggregate(
    out_results_dir: str,
    out_plots_dir: str,
    prefix: str,
    n_classes: int,
    slicing_day: int,
    hs_tag: str,
    seeds: List[int],
    groups: np.ndarray,
    pergroup_acc_matrix: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    overall_acc: np.ndarray,
    train_ratio: float,
    group_name: str,
) -> Tuple[str, str]:
    seed_tag = f"seeds{min(seeds)}-{max(seeds)}"
    agg_base = f"{prefix}_{hs_tag}_{seed_tag}_per{group_name}_testmeanstd_Nclasses{n_classes}_day{slicing_day}"

    agg_path = os.path.join(out_results_dir, f"{agg_base}.npz")
    np.savez(
        agg_path,
        seeds=np.array(seeds, dtype=int),
        groups=groups.astype(int),
        pergroup_acc_matrix=pergroup_acc_matrix.astype(float),
        mean=mean_acc.astype(float),
        std=std_acc.astype(float),
        overall_acc=overall_acc.astype(float),
        train_ratio=float(train_ratio),
        group_name=str(group_name),
    )

    agg_png = os.path.join(out_plots_dir, f"{agg_base}.png")
    plot_per_group_mean_std(groups, mean_acc, std_acc, n_classes=n_classes, out_png=agg_png, group_name=group_name)
    return agg_path, agg_png


# =========================
# Main
# =========================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # data paths
    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--trial_bin_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)

    # slicing / masks
    ap.add_argument("--slicing_day", type=int, required=True)  # 1-indexed among unique days
    ap.add_argument("--target_type", type=str, default=None)   # center-out / random
    ap.add_argument("--target_style_path", type=str, default=None)
    ap.add_argument("--label_mask", type=str, default=None)

    # preprocessing
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--log_scale", action="store_true")

    # legacy shuffle split (only used if --split_npz is not given)
    ap.add_argument("--train_ratio", type=float, default=0.8)

    # torch MLP
    ap.add_argument("--hidden_sizes", type=str, default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=1)

    # output
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)

    # temporal stacking
    ap.add_argument("--n_lags", type=int, default=0)
    ap.add_argument("--lag_step", type=int, default=1)
    ap.add_argument("--lag_group", type=str, default="trial", choices=["none", "day", "trial"])

    # Optuna best params
    ap.add_argument("--best_param", type=str, default=None)

    # NEW: fixed split from mlp_hpo.py
    ap.add_argument(
        "--split_npz",
        type=str,
        default=None,
        help="Path to split_*.npz saved by mlp_hpo.py. "
             "If provided, train uses train_trials and test uses test_trials (Optuna-unseen).",
    )

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    # (0) Optional: load Optuna best params and override args
    if args.best_param is not None and str(args.best_param).strip() != "":
        with open(args.best_param, "r") as f:
            obj = json.load(f)
        bp = obj.get("best_params", obj)

        # --- hidden sizes from h1/h2/h3/h4 ---
        h1 = int(bp["h1"])
        h2 = int(bp.get("h2", 0))
        h3 = int(bp.get("h3", 0))
        h4 = int(bp.get("h4", 0))
        
        hidden = [h1]
        if h2 != 0: hidden.append(h2)
        if h3 != 0: hidden.append(h3)
        if h4 != 0: hidden.append(h4)
        
        args.hidden_sizes = ",".join(map(str, hidden))

        args.lr = float(bp["lr"])
        args.weight_decay = float(bp["weight_decay"])
        args.batch_size = int(bp["batch_size"])
        args.n_lags = int(bp["n_lags"])
        args.lag_step = int(bp["lag_step"])

        logger.info(f"[HPO] Loaded best params from: {args.best_param}")
        logger.info(
            f"[HPO] Override -> hidden_sizes={args.hidden_sizes}, lr={args.lr}, wd={args.weight_decay}, "
            f"batch={args.batch_size}, n_lags={args.n_lags}, lag_step={args.lag_step}"
        )

    if (args.best_param is None or str(args.best_param).strip() == "") and (args.hidden_sizes is None or args.hidden_sizes.strip() == ""):
        raise ValueError("Provide --hidden_sizes (e.g., '256,128') or --best_param path to best_params.json")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")

    outp = make_outdirs(args.out_dir)
    device = get_device()
    logger.info(f"Using device: {device}")

    hidden = parse_hidden_sizes(args.hidden_sizes)
    hs_tag = hidden_tag(hidden)

    # -------------------------
    # (1) Load all arrays
    # -------------------------
    data0 = load_all_arrays(args)

    # -------------------------
    # (2) Apply task masks
    # -------------------------
    allowed = parse_label_mask(args.label_mask)
    # data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=allowed)
    data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=None)

    # -------------------------
    # (3) Build X/y/day after masking
    # -------------------------
    X = data.sbp.astype(np.float32)
    y_raw = data.labels.astype(int)

    # NEW: Remap labels to be contiguous so PyTorch doesn't crash for the case labels are masked
    unique_labels = np.sort(np.unique(y_raw))
    remapper = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
    y = np.vectorize(remapper.get)(y_raw)
    day_info_i = np.asarray(data.day_info).astype(int)

    slicing_day_value = compute_slicing_day_value(day_info_i, args.slicing_day)
    print(f"Slicing day value: {slicing_day_value}  (args.slicing_day={args.slicing_day})")

    # -------------------------
    # (4) Use only data up to slicing day
    # -------------------------
    use_mask = (day_info_i <= slicing_day_value)
    X_use0 = X[use_mask]
    y_use0 = y[use_mask]
    day_use0 = day_info_i[use_mask]

    trial_bin_use = data.trial_bin[use_mask]
    trial_id_use0, n_trials = build_trial_ids(trial_bin_use)
    print("count_1 (trials within used data):", n_trials)

    # -------------------------
    # (4.5) Load fixed split (from mlp_hpo.py), if provided
    # -------------------------
    split = None
    train_trials = None
    test_trials = None

    if args.split_npz is not None and str(args.split_npz).strip() != "":
        split = load_split_npz(args.split_npz)
        train_trials = split["train_trials"].astype(int)
        test_trials = split["test_trials"].astype(int)

        # sanity check: slicing_day_value must match (strongly recommended)
        split_sdv = int(split["slicing_day_value"])
        if split_sdv != int(slicing_day_value):
            raise ValueError(
                f"split_npz slicing_day_value mismatch: split={split_sdv} vs current={int(slicing_day_value)}. "
                f"Make sure you use the SAME slicing_day and SAME masking (target_type/label_mask)."
            )

        # to guarantee trial-wise leakage-free split in lag, enforce lag_group=trial here
        if args.lag_group != "trial":
            raise ValueError("When using --split_npz, set --lag_group trial (required).")

        logger.info(f"[SPLIT] Loaded split_npz: {args.split_npz}")
        logger.info(f"[SPLIT] #train_trials={len(train_trials)}  #test_trials={len(test_trials)}")

    # -------------------------
    # (5) Lag stacking
    # -------------------------
    group0 = choose_lag_group(args.lag_group, day_vec=day_use0, trial_id_vec=trial_id_use0)
    X_lag, y_lag, group_eff = make_lagged_features( #(N_filtered, D * (n_lags + 1))
        X=X_use0,
        y=y_use0,
        group=group0,
        n_lags=args.n_lags,
        lag_step=args.lag_step,
    )
    
    print(f"After lagging: N={X_lag.shape[0]}  D={X_lag.shape[1]}  (n_lags={args.n_lags} step={args.lag_step} group={args.lag_group})")
    
    if X_lag.shape[0] == 0:
        raise ValueError("No samples after applying slicing_day & masks & lagging.")

    # In lag_group=trial, group_eff is trial_id per lagged sample (this is what we need for split and per-trial acc)
    if group_eff is None:
        group_eff = np.zeros(X_lag.shape[0], dtype=int)

    # NEW: FILTER CLASS AND REMAP AFTER LAGGING
    # ==========================================
    if allowed is not None:
        valid_idx = np.isin(y_lag, allowed)
        X_lag = X_lag[valid_idx]
        y_lag = y_lag[valid_idx]
        group_eff = group_eff[valid_idx]
        
    if X_lag.shape[0] == 0:
        raise ValueError("No samples left after removing masked classes.")

    # Remap labels to be contiguous (e.g., [0, 2] becomes [0, 1])
    unique_labels = np.sort(np.unique(y_lag))
    remapper = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
    y_lag = np.vectorize(remapper.get)(y_lag)
    # ==========================================

    n_classes = int(np.unique(y_lag).size)
    print(f"[Used+Lag] N={len(y_lag)}  Nclasses={n_classes}")

    # -------------------------
    # (6) Make train/test split
    # -------------------------
    if split is not None:
        tr_mask = np.isin(group_eff, train_trials)
        te_mask = np.isin(group_eff, test_trials)
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            raise ValueError("Empty train or test after applying split trials. Check split_npz vs current masks.")
        X_train_raw = X_lag[tr_mask].copy()
        y_train = y_lag[tr_mask].copy()
        X_test_raw = X_lag[te_mask].copy()
        y_test = y_lag[te_mask].copy()
        group_test = group_eff[te_mask].copy()
        split_name = "fixed_split_npz"
    else:
        # legacy behavior (not independent test): shuffle split => test is just a holdout
        N = X_lag.shape[0]
        perm = np.random.permutation(N)
        n_train = int(round(N * args.train_ratio))
        n_train = max(1, min(n_train, N - 1))
        tr_idx = perm[:n_train]
        te_idx = perm[n_train:]
        X_train_raw = X_lag[tr_idx].copy()
        y_train = y_lag[tr_idx].copy()
        X_test_raw = X_lag[te_idx].copy()
        y_test = y_lag[te_idx].copy()
        group_test = group_eff[te_idx].copy()
        split_name = "shuffle_holdout"

    # train label distribution (true train only)
    labeldist_png = os.path.join(
        outp.plots,
        f"{args.prefix}_{hs_tag}_labeldist_train_Nclasses{n_classes}_day{args.slicing_day}.png",
    )
    plot_label_distribution(y_train, labeldist_png, n_classes=n_classes)

    # -------------------------
    # (7) Seeds loop (kept simple: same as before)
    # -------------------------
    seeds = list(range(1))
    seed_to_groupacc: Dict[int, Dict[int, float]] = {}
    seed_to_overall_acc: Dict[int, float] = {}

    for seed in seeds:
        set_seed(seed)

        # preprocessing: fit scaler on TRAIN ONLY -> apply to TEST
        X_train, X_test, scaler = preprocess_train_val(
            X_train_raw=X_train_raw,
            X_val_raw=X_test_raw,
            scale=args.scale,
            log_scale=args.log_scale,
        )

        model = MLP(in_dim=X_train.shape[1], hidden=hidden, out_dim=n_classes).to(device)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=True,
        )

        train_mlp(
            model=model,
            train_loader=train_loader,
            y_train=y_train,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            print_every=args.print_every,
            seed=seed,
        )

        # eval on TEST (independent if split_npz is used)
        y_pred = predict_labels(model, X_val=X_test, y_val=y_test, device=device, batch_size=args.batch_size)
        correct01 = (y_pred == y_test).astype(np.int8)
        acc = float(correct01.mean())
        seed_to_overall_acc[seed] = acc
        print(f"[TEST][seed={seed}] N={len(y_test)} accuracy={acc:.4f}  (split={split_name})")

        # per-trial accuracy on test
        uniq_g, g_acc, g_n = per_group_accuracy(group_test, y_pred=y_pred, y_true=y_test)
        seed_to_groupacc[seed] = {int(g): float(a) for g, a in zip(uniq_g, g_acc)}

        base = f"{args.prefix}_{hs_tag}_seed{seed}_Nclasses{n_classes}_day{args.slicing_day}_{split_name}"

        w_path = save_weights_npz(
            out_weights_dir=outp.weights,
            base_name=f"{base}_weights",
            model=model,
            seed=seed,
            args=args,
            n_classes=n_classes,
            slicing_day_value=slicing_day_value,
            hidden=hidden,
            scaler=scaler,
        )

        perf_path, group_path = save_test_outputs(
            out_results_dir=outp.results,
            base_name=base,
            correct01=correct01,
            pergroup_groups=uniq_g,
            pergroup_acc=g_acc,
            pergroup_n=g_n,
            y_pred=y_pred,
            y_true=y_test,
            group_of_each_sample=group_test,
            seed=seed,
        )

        logger.info(f"Saved weights:  {w_path}")
        logger.info(f"Saved results:  {perf_path}")
        logger.info(f"Saved per-trial: {group_path}")

    # -------------------------
    # (8) Aggregate mean/std across seeds
    # -------------------------
    all_groups_sorted = sorted({g for seed in seeds for g in seed_to_groupacc[seed].keys()})
    all_groups = np.array(all_groups_sorted, dtype=int)

    A = np.full((len(seeds), len(all_groups)), np.nan, dtype=float)
    for si, seed in enumerate(seeds):
        d2a = seed_to_groupacc[seed]
        for gi, g in enumerate(all_groups):
            if g in d2a:
                A[si, gi] = d2a[g]

    mean_acc = np.nanmean(A, axis=0)
    std_acc = np.nanstd(A, axis=0)
    overall_acc = np.array([seed_to_overall_acc[s] for s in seeds], dtype=float)

    agg_path, agg_png = save_aggregate(
        out_results_dir=outp.results,
        out_plots_dir=outp.plots,
        prefix=args.prefix,
        n_classes=n_classes,
        slicing_day=args.slicing_day,
        hs_tag=hs_tag,
        seeds=seeds,
        groups=all_groups,
        pergroup_acc_matrix=A,
        mean_acc=mean_acc,
        std_acc=std_acc,
        overall_acc=overall_acc,
        train_ratio=float(args.train_ratio),
        group_name="trial",
    )

    print("Saved (aggregate):")
    print(" ", agg_path)
    print(" ", agg_png)
    print("Saved (plots):")
    print(" ", labeldist_png)


if __name__ == "__main__":
    main()