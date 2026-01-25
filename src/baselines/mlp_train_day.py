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
# Data Pipeline (load -> mask -> slice -> trial ids -> lag)
# =========================
@dataclass
class LoadedData:
    sbp: np.ndarray           # (N, D)
    labels: np.ndarray        # (N,)
    day_info: np.ndarray      # (N,)
    trial_bin: np.ndarray     # (N,) float, 0.0 at trial start
    target_style: Optional[np.ndarray]  # (N,) bool or None


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
    IMPORTANT: logic identical to the original script:
      1) label_mask: keep labels in allowed_labels
      2) target_type:
          center-out => keep ~target_style
          random     => keep target_style
    """
    keep = np.ones(data.sbp.shape[0], dtype=bool)

    # (1) label_mask
    if allowed_labels is not None:
        keep &= np.isin(data.labels.astype(int), allowed_labels)

    # (2) target_type masking
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

    # apply
    sbp = data.sbp[keep]
    labels = data.labels[keep]
    day_info = data.day_info[keep]
    trial_bin = data.trial_bin[keep]
    target_style = (np.asarray(data.target_style).astype(bool)[keep] if data.target_style is not None else None)

    return LoadedData(sbp=sbp, labels=labels, day_info=day_info, trial_bin=trial_bin, target_style=target_style)


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
    """Match original behavior: none/day/trial."""
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
@dataclass
class SplitData:
    X_train_raw: np.ndarray
    y_train: np.ndarray
    X_val_raw: np.ndarray
    y_val: np.ndarray
    day_val: np.ndarray


def shuffle_split(X: np.ndarray, y: np.ndarray, day: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle split identical logic to the original script."""
    N = X.shape[0]
    n_train = int(round(N * train_ratio))
    n_train = max(1, min(n_train, N - 1))  # ensure both non-empty

    perm = np.random.permutation(N)
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]
    return tr_idx, va_idx, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], day[va_idx]


def preprocess_train_val(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    scale: bool,
    log_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """Fit scaler on train only (unchanged logic)."""
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
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
    seed: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
# Saving
# =========================
def save_weights_npz(
    out_weights_dir: str,
    base_name: str,
    model: nn.Module,
    seed: int,
    args,
    n_classes: int,
    day_info: int,
    hidden: List[int],
    scaler: Optional[StandardScaler],
) -> str:
    w_path = os.path.join(out_weights_dir, f"{base_name}.npz")
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        w_path,
        seed=seed,
        state_dict=np.array(state, dtype=object),
        day_info=day_info,
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

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_type", type=str, default=None)   # center-out / random
    ap.add_argument("--target_style_path", type=str, default=None)
    ap.add_argument("--label_mask", type=str, default=None)

    # preprocessing
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--log_scale", action="store_true")

    # shuffle split
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Shuffle split ratio within sliced data")

    # torch MLP
    ap.add_argument("--hidden_sizes", type=str, default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    ap.add_argument(
        "--best_param",
        type=str,
        default=None,
        help="Path to Optuna best_params.json; if provided, overrides hidden_sizes/lr/weight_decay/batch_size/n_lags/lag_step",
    )
    return ap


def _train_eval_one_day(
    *,
    day_value: int,
    X_day: np.ndarray,
    y_day: np.ndarray,
    trial_bin_day: np.ndarray,
    args,
    hidden: List[int],
    hs_tag: str,
    outp,
    device,
):
    """Train exactly ONE model on this day, using shuffle split 80:20, return test acc."""
    # trial ids (for lag_group="trial")
    trial_id_day, n_trials = build_trial_ids(trial_bin_day)

    # lag stacking (keep args.n_lags etc.)
    group = choose_lag_group(args.lag_group, day_vec=np.full(len(y_day), day_value, dtype=int), trial_id_vec=trial_id_day)
    X_use, y_use, _ = make_lagged_features(
        X=X_day.astype(np.float32),
        y=y_day.astype(int),
        group=group,
        n_lags=args.n_lags,
        lag_step=args.lag_step,
    )

    if X_use.shape[0] < 2:
        print(f"[Day {day_value}] skipped (too few samples after lagging): N={X_use.shape[0]}")
        return None

    # label dist (optional print)
    n_classes = int(np.unique(y_use).size)
    cls, cnt = np.unique(y_use, return_counts=True)
    print(f"[Day {day_value}] N={len(y_use)}  Nclasses={n_classes}  label_counts={dict(zip(cls.tolist(), cnt.tolist()))}")

    # shuffle split within this day
    N = X_use.shape[0]
    perm = np.random.permutation(N)
    n_train = int(round(N * 0.8))  # FORCE 80:20
    n_train = max(1, min(n_train, N - 1))
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:]

    X_train_raw = X_use[tr_idx].copy()
    y_train = y_use[tr_idx].copy()
    X_test_raw = X_use[te_idx].copy()
    y_test = y_use[te_idx].copy()

    # preprocessing: fit scaler on TRAIN ONLY
    X_train, X_test, scaler = preprocess_train_val(
        X_train_raw=X_train_raw,
        X_val_raw=X_test_raw,
        scale=args.scale,
        log_scale=args.log_scale,
    )

    # model
    model = MLP(in_dim=X_train.shape[1], hidden=hidden, out_dim=n_classes).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # train
    train_mlp(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        print_every=args.print_every,
        seed=int(args.seed),
    )

    # eval on test
    y_pred = predict_labels(model, X_val=X_test, y_val=y_test, device=device, batch_size=args.batch_size)
    correct01 = (y_pred == y_test).astype(np.int8)
    acc = float(correct01.mean())
    print(f"[Test][Day {day_value}][seed={args.seed}] N={len(y_test)} accuracy={acc:.4f}")

    # --- save (per day) ---
    base = f"{args.prefix}_{hs_tag}_seed{int(args.seed)}_Nclasses{n_classes}_day{int(day_value)}"

    # weights
    w_path = save_weights_npz(
        out_weights_dir=outp.weights,
        base_name=f"{base}_weights",
        model=model,
        seed=int(args.seed),
        args=args,
        n_classes=n_classes,
        day_info=int(day_value),  # keep field name for backward compat in saver
        hidden=hidden,
        scaler=scaler,
    )

    # results (store y_pred/y_true and correct01)
    perf_path, day_path = save_val_outputs(
        out_results_dir=outp.results,
        base_name=f"{base}_test",
        correct01=correct01,
        perday_days=np.array([day_value], dtype=int),
        perday_acc=np.array([acc], dtype=float),
        perday_n=np.array([len(y_test)], dtype=int),
        y_pred=y_pred,
        y_true=y_test,
        day_of_each_sample=np.full(len(y_test), day_value, dtype=np.int16),
        seed=int(args.seed),
    )

    return acc


def main() -> None:
    args = build_argparser().parse_args()

    # (0) Optional: load Optuna best params and override args (existing logic)
    if args.best_param is not None and str(args.best_param).strip() != "":
        with open(args.best_param, "r") as f:
            obj = json.load(f)
        bp = obj.get("best_params", obj)

        h1 = int(bp["h1"])
        h2 = int(bp.get("h2", 0))
        hidden = [h1] if (h2 == 0) else [h1, h2]
        args.hidden_sizes = ",".join(map(str, hidden))

        args.lr = float(bp["lr"])
        args.weight_decay = float(bp["weight_decay"])
        args.batch_size = int(bp["batch_size"])
        args.n_lags = int(bp["n_lags"])
        args.lag_step = int(bp["lag_step"])

        logger.info(f"[HPO] Loaded best params from: {args.best_param}")

    if (args.best_param is None or str(args.best_param).strip() == "") and (args.hidden_sizes is None or args.hidden_sizes.strip() == ""):
        raise ValueError("Provide --hidden_sizes (e.g., '256,128') or --best_param path to best_params.json")

    outp = make_outdirs(args.out_dir)
    device = get_device()
    logger.info(f"Using device: {device}")

    # FORCE 80:20
    args.train_ratio = 0.8

    # set seed once (one model per day, deterministic split/training per day if desired)
    set_seed(int(args.seed))

    hidden = parse_hidden_sizes(args.hidden_sizes)
    hs_tag = hidden_tag(hidden)

    # -------------------------
    # (1) Load all arrays
    # -------------------------
    data0 = load_all_arrays(args)

    # -------------------------
    # (2) Apply task masks (unchanged)
    # -------------------------
    allowed = parse_label_mask(args.label_mask)
    data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=allowed)

    # -------------------------
    # (3) Build X/y/day after masking
    # -------------------------
    X = data.sbp.astype(np.float32)
    y = data.labels.astype(int)
    day_info_i = np.asarray(data.day_info).astype(int)
    trial_bin = np.asarray(data.trial_bin)

    uniq_days = np.unique(day_info_i)
    uniq_days_sorted = np.sort(uniq_days)
    print(f"[All days] n_days={len(uniq_days_sorted)}  days={uniq_days_sorted.tolist()}")

    # -------------------------
    # (4) Train 1 model per day, store all test acc in a list
    # -------------------------
    test_acc_list = []
    day_list = []

    for d in uniq_days_sorted:
        day_mask = (day_info_i == d)
        X_day = X[day_mask]
        y_day = y[day_mask]
        trial_bin_day = trial_bin[day_mask]

        acc = _train_eval_one_day(
            day_value=int(d),
            X_day=X_day,
            y_day=y_day,
            trial_bin_day=trial_bin_day,
            args=args,
            hidden=hidden,
            hs_tag=hs_tag,
            outp=outp,
            device=device,
        )
        if acc is None:
            continue

        day_list.append(int(d))
        test_acc_list.append(float(acc))

    # -------------------------
    # (5) Save aggregated per-day test performance (list)
    # -------------------------
    day_arr = np.array(day_list, dtype=int)
    acc_arr = np.array(test_acc_list, dtype=float)

    agg_path = os.path.join(
        outp.results,
        f"{args.prefix}_{hs_tag}_seed{int(args.seed)}_perday_TESTACC.npz"
    )
    np.savez(
        agg_path,
        seed=int(args.seed),
        days=day_arr,
        test_acc=acc_arr,
        train_ratio=0.8,
        n_lags=int(args.n_lags),
        lag_step=int(args.lag_step),
        lag_group=str(args.lag_group),
        hidden_sizes=str(args.hidden_sizes),
    )
    print("[Saved] per-day test acc list ->", agg_path)
    print("[Summary]")
    for d, a in zip(day_arr.tolist(), acc_arr.tolist()):
        print(f"  day={d}  test_acc={a:.4f}")
        
if __name__ == "__main__":
    main()