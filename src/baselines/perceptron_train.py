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


def plot_per_day_mean_std(days: np.ndarray, mean_acc: np.ndarray, std_acc: np.ndarray, n_classes: int, out_png: str) -> None:
    plt.figure()
    plt.plot(days, mean_acc, marker="o")
    plt.fill_between(days, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    plt.xlabel("trial")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"test trials accuracy mean±std over seeds (Nclasses={n_classes})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Metrics
# =========================
def per_day_accuracy(day_vec: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute accuracy per unique day."""
    uniq_days = np.unique(day_vec)
    day_acc = []
    day_n = []
    for d in uniq_days:
        idx = (day_vec == d)
        day_n.append(int(idx.sum()))
        day_acc.append(float((y_pred[idx] == y_true[idx]).mean()))
    return uniq_days, np.asarray(day_acc, dtype=float), np.asarray(day_n, dtype=int)


# =========================
# Lag Feature Construction (unchanged logic)
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
# Model (Perceptron = single Linear layer)
# =========================
class Perceptron(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# =========================
# Data Pipeline (load -> mask -> slice -> trial ids -> lag)
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
    IMPORTANT: logic identical to mlp_train.py:
      1) label_mask: keep labels in allowed_labels
      2) target_type:
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
    uniq_all_days = np.unique(day_info_i)
    if slicing_day_index_1based < 1 or slicing_day_index_1based > len(uniq_all_days):
        raise ValueError(f"--slicing_day must be in [1, {len(uniq_all_days)}]")
    return int(uniq_all_days[slicing_day_index_1based - 1])


def build_trial_ids(trial_bin_used: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Build trial_id from trial_bin (0.0 indicates trial start).
    Safety: if the first used sample isn't a trial start, force it as start.
    """
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
# Preprocess / Train / Eval
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


def train_perceptron(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
    seed: int,
) -> None:
    # Linear classifier trained with CE (softmax regression); architecture is still a Perceptron layer.
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

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
# Saving (same format; hs_tag fixed to "perceptron")
# =========================
def save_weights_npz(
    out_weights_dir: str,
    base_name: str,
    model: nn.Module,
    seed: int,
    args,
    n_classes: int,
    slicing_day_value: int,
    scaler: Optional[StandardScaler],
) -> str:
    w_path = os.path.join(out_weights_dir, f"{base_name}.npz")
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        w_path,
        seed=seed,
        state_dict=np.array(state, dtype=object),
        model_type="perceptron",
        slicing_day=args.slicing_day,
        slicing_day_value=int(slicing_day_value),
        target_type=args.target_type,
        label_mask=args.label_mask,
        scale=args.scale,
        log_scale=args.log_scale,
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
    )
    return w_path


def save_val_outputs(
    out_results_dir: str,
    base_name: str,
    correct01: np.ndarray,
    perday_days: np.ndarray,
    perday_acc: np.ndarray,
    perday_n: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    day_of_each_sample: np.ndarray,
    seed: int,
) -> Tuple[str, str]:
    perf_path = os.path.join(out_results_dir, f"{base_name}_val_correct01.npy")
    np.save(perf_path, correct01)

    day_path = os.path.join(out_results_dir, f"{base_name}_perday_valacc.npz")
    np.savez(
        day_path,
        seed=seed,
        days=perday_days.astype(int),
        acc=perday_acc.astype(float),
        n=perday_n.astype(int),
        y_pred=y_pred.astype(np.int16),
        y_true=y_true.astype(np.int16),
        day_of_each_sample=day_of_each_sample.astype(np.int16),
    )
    return perf_path, day_path


def save_aggregate(
    out_results_dir: str,
    out_plots_dir: str,
    prefix: str,
    n_classes: int,
    slicing_day: int,
    hs_tag: str,
    seeds: List[int],
    days: np.ndarray,
    perday_acc_matrix: np.ndarray,
    mean_acc: np.ndarray,
    std_acc: np.ndarray,
    overall_acc: np.ndarray,
    train_ratio: float,
) -> Tuple[str, str]:
    seed_tag = f"seeds{min(seeds)}-{max(seeds)}"
    agg_base = f"{prefix}_{hs_tag}_{seed_tag}_perday_valmeanstd_Nclasses{n_classes}_day{slicing_day}"

    agg_path = os.path.join(out_results_dir, f"{agg_base}.npz")
    np.savez(
        agg_path,
        seeds=np.array(seeds, dtype=int),
        days=days.astype(int),
        perday_acc_matrix=perday_acc_matrix.astype(float),
        mean=mean_acc.astype(float),
        std=std_acc.astype(float),
        overall_acc=overall_acc.astype(float),
        train_ratio=float(train_ratio),
    )

    agg_png = os.path.join(out_plots_dir, f"{agg_base}.png")
    plot_per_day_mean_std(days, mean_acc, std_acc, n_classes=n_classes, out_png=agg_png)
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

    # shuffle split
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Shuffle split ratio within sliced data")

    # torch training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)  # often larger OK for SGD-linear
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=1)

    # output
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)

    # temporal stacking
    ap.add_argument("--n_lags", type=int, default=0, help="Number of past lags to stack (0=off)")
    ap.add_argument("--lag_step", type=int, default=1, help="Step between lags in samples (1=adjacent)")
    ap.add_argument(
        "--lag_group",
        type=str,
        default="trial",
        choices=["none", "day", "trial"],
        help="Prevent lag crossing boundaries: trial recommended for concatenated-success data",
    )

    # optional: reuse Optuna best_params.json (ignores h1/h2; uses lr/wd/batch/n_lags/lag_step if present)
    ap.add_argument(
        "--best_param",
        type=str,
        default=None,
        help="Path to Optuna best_params.json; if provided, overrides lr/weight_decay/batch_size/n_lags/lag_step when present",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    # Optional: load Optuna best params (safe: ignores missing keys)
    if args.best_param is not None and str(args.best_param).strip() != "":
        with open(args.best_param, "r") as f:
            obj = json.load(f)
        bp = obj.get("best_params", obj)

        if "lr" in bp:
            args.lr = float(bp["lr"])
        if "weight_decay" in bp:
            args.weight_decay = float(bp["weight_decay"])
        if "batch_size" in bp:
            args.batch_size = int(bp["batch_size"])
        if "n_lags" in bp:
            args.n_lags = int(bp["n_lags"])
        if "lag_step" in bp:
            args.lag_step = int(bp["lag_step"])

        logger.info(f"[HPO] Loaded best params from: {args.best_param}")
        logger.info(
            f"[HPO] Override -> lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
            f"n_lags={args.n_lags}, lag_step={args.lag_step}"
        )

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")

    outp = make_outdirs(args.out_dir)
    device = get_device()
    logger.info(f"Using device: {device}")

    hs_tag = "perceptron"

    # -------------------------
    # (1) Load all arrays
    # -------------------------
    data0 = load_all_arrays(args)

    # -------------------------
    # (2) Apply task masks (logic unchanged)
    # -------------------------
    allowed = parse_label_mask(args.label_mask)
    data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=allowed)

    # -------------------------
    # (3) Build X/y/day after masking
    # -------------------------
    X = data.sbp.astype(np.float32)
    y = data.labels.astype(int)
    day_info_i = np.asarray(data.day_info).astype(int)

    slicing_day_value = compute_slicing_day_value(day_info_i, args.slicing_day)
    print(f"Slicing day value: {slicing_day_value}  (args.slicing_day={args.slicing_day})")

    # -------------------------
    # (4) Use only data up to slicing day (no future-day test)
    # -------------------------
    use_mask = (day_info_i <= slicing_day_value)

    X_use = X[use_mask]
    y_use = y[use_mask]
    day_use = day_info_i[use_mask]

    trial_bin_use = data.trial_bin[use_mask]
    trial_id_use, n_trials = build_trial_ids(trial_bin_use)
    print("count_1 (trials within used data):", n_trials)

    # -------------------------
    # (5) Lag stacking (logic unchanged)
    # -------------------------
    group = choose_lag_group(args.lag_group, day_vec=day_use, trial_id_vec=trial_id_use)
    X_use, y_use, _ = make_lagged_features(
        X=X_use,
        y=y_use,
        group=group,
        n_lags=args.n_lags,
        lag_step=args.lag_step,
    )

    # keep identical “day_use overwritten to group_eff” behavior when n_lags>0
    if args.n_lags > 0:
        _, _, day_use = make_lagged_features(
            X=X[use_mask].astype(np.float32),
            y=y[use_mask].astype(int),
            group=group,
            n_lags=args.n_lags,
            lag_step=args.lag_step,
        )

    if X_use.shape[0] == 0:
        raise ValueError("No samples after applying slicing_day & masks.")

    n_classes = int(np.unique(y_use).size)
    print(f"[Used] N={len(y_use)}  Nclasses={n_classes}")
    cls, cnt = np.unique(y_use, return_counts=True)
    print("[Used] label counts:", dict(zip(cls.tolist(), cnt.tolist())))

    labeldist_png = os.path.join(
        outp.plots,
        f"{args.prefix}_{hs_tag}_labeldist_Nclasses{n_classes}_day{args.slicing_day}.png",
    )
    plot_label_distribution(y_use, labeldist_png, n_classes=n_classes)

    # -------------------------
    # (6) Seeds loop (same as mlp_train.py)
    # -------------------------
    seeds = list(range(10))
    seed_to_dayacc: Dict[int, Dict[int, float]] = {}
    seed_to_overall_acc: Dict[int, float] = {}

    N = X_use.shape[0]

    for seed in seeds:
        set_seed(seed)

        perm = np.random.permutation(N)
        n_train = int(round(N * args.train_ratio))
        n_train = max(1, min(n_train, N - 1))
        tr_idx = perm[:n_train]
        va_idx = perm[n_train:]

        X_train_raw = X_use[tr_idx].copy()
        y_train = y_use[tr_idx].copy()
        X_val_raw = X_use[va_idx].copy()
        y_val = y_use[va_idx].copy()

        day_val = day_use[va_idx]

        X_train, X_val, scaler = preprocess_train_val(
            X_train_raw=X_train_raw,
            X_val_raw=X_val_raw,
            scale=args.scale,
            log_scale=args.log_scale,
        )

        model = Perceptron(in_dim=X_train.shape[1], out_dim=n_classes).to(device)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=True,
        )

        train_perceptron(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            print_every=args.print_every,
            seed=seed,
        )

        y_pred = predict_labels(model, X_val=X_val, y_val=y_val, device=device, batch_size=args.batch_size)
        correct01 = (y_pred == y_val).astype(np.int8)
        acc = float(correct01.mean())
        seed_to_overall_acc[seed] = acc
        print(f"[Val][seed={seed}] N={len(y_val)} accuracy={acc:.4f}")

        uniq_days, day_acc, day_n = per_day_accuracy(day_val, y_pred=y_pred, y_true=y_val)
        seed_to_dayacc[seed] = {int(d): float(a) for d, a in zip(uniq_days, day_acc)}

        base = f"{args.prefix}_{hs_tag}_seed{seed}_Nclasses{n_classes}_day{args.slicing_day}"

        w_path = save_weights_npz(
            out_weights_dir=outp.weights,
            base_name=f"{base}_weights",
            model=model,
            seed=seed,
            args=args,
            n_classes=n_classes,
            slicing_day_value=slicing_day_value,
            scaler=scaler,
        )

        perf_path, day_path = save_val_outputs(
            out_results_dir=outp.results,
            base_name=base,
            correct01=correct01,
            perday_days=uniq_days,
            perday_acc=day_acc,
            perday_n=day_n,
            y_pred=y_pred,
            y_true=y_val,
            day_of_each_sample=day_val,
            seed=seed,
        )

        logger.info(f"Saved weights:  {w_path}")
        logger.info(f"Saved results:  {perf_path}")
        logger.info(f"Saved per-day:  {day_path}")

    # -------------------------
    # (7) Aggregate mean/std across seeds
    # -------------------------
    all_days_sorted = sorted({d for seed in seeds for d in seed_to_dayacc[seed].keys()})
    all_days = np.array(all_days_sorted, dtype=int)

    A = np.full((len(seeds), len(all_days)), np.nan, dtype=float)
    for si, seed in enumerate(seeds):
        d2a = seed_to_dayacc[seed]
        for di, d in enumerate(all_days):
            if d in d2a:
                A[si, di] = d2a[d]

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
        days=all_days,
        perday_acc_matrix=A,
        mean_acc=mean_acc,
        std_acc=std_acc,
        overall_acc=overall_acc,
        train_ratio=float(args.train_ratio),
    )

    print("Saved (aggregate):")
    print(" ", agg_path)
    print(" ", agg_png)
    print("Saved (plots):")
    print(" ", labeldist_png)


if __name__ == "__main__":
    main()
