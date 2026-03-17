#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from utils.trial_splits import split_group_ids
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
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
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([int(p) for p in parts], dtype=int)


def parse_hidden_sizes(s: str) -> List[int]:
    s = (s or "").strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def hidden_tag(hidden: List[int]) -> str:
    return "hsnone" if len(hidden) == 0 else "hs" + "-".join(map(str, hidden))


# =========================
# Output dirs
# =========================
@dataclass(frozen=True)
class OutPaths:
    root: str
    weights: str
    results: str


def make_outdirs(out_dir: str) -> OutPaths:
    root = out_dir
    weights = os.path.join(root, "weights")
    results = os.path.join(root, "results")
    os.makedirs(weights, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    return OutPaths(root=root, weights=weights, results=results)


# =========================
# Metrics (macro)
# =========================
def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray) -> Dict[int, Tuple[int, int, int]]:
    """
    returns dict[c] = (tp, fp, fn)
    """
    out = {}
    for c in class_ids:
        c = int(c)
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        out[c] = (tp, fp, fn)
    return out


def balanced_accuracy_macro(y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray) -> float:
    # mean recall over classes
    rec = []
    for c in class_ids:
        c = int(c)
        denom = int(np.sum(y_true == c))
        if denom == 0:
            continue
        rec.append(float(np.sum((y_true == c) & (y_pred == c))) / float(denom))
    return float(np.mean(rec)) if len(rec) > 0 else float("nan")


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray) -> float:
    d = _confusion_counts(y_true, y_pred, class_ids)
    f1s = []
    for c in class_ids:
        tp, fp, fn = d[int(c)]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(float(f1))
    return float(np.mean(f1s)) if len(f1s) > 0 else float("nan")


# =========================
# Lag feature construction (trial-safe)
# =========================
def make_lagged_features(
    X: np.ndarray,
    y: np.ndarray,
    group: Optional[np.ndarray],
    n_lags: int,
    lag_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if n_lags <= 0:
        return X.astype(np.float32), y.astype(int), group
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
            y_out.append(int(y[t]))
            g_out.append(int(g))

    if len(X_out) == 0:
        return np.zeros((0, D * (n_lags + 1)), dtype=np.float32), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    X_lag = np.stack(X_out, axis=0).astype(np.float32)
    y_eff = np.asarray(y_out, dtype=int)
    g_eff = np.asarray(g_out, dtype=int)
    return X_lag, y_eff, g_eff


def build_trial_ids(trial_bin_used: np.ndarray) -> Tuple[np.ndarray, int]:
    trial_starts = (trial_bin_used == 0.0)
    if len(trial_bin_used) > 0 and (trial_bin_used[0] != 0.0):
        trial_starts[0] = True
    trial_id = np.cumsum(trial_starts).astype(int) - 1
    n_trials = int(trial_starts.sum())
    return trial_id, n_trials


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


def preprocess_train_test(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    scale: bool,
    log_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    scaler = None
    Xtr = X_train_raw
    Xte = X_test_raw

    if scale:
        if log_scale:
            Xtr = np.log1p(Xtr)
            Xte = np.log1p(Xte)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr).astype(np.float32)
        Xte = scaler.transform(Xte).astype(np.float32)
    else:
        Xtr = Xtr.astype(np.float32)
        Xte = Xte.astype(np.float32)

    return Xtr, Xte, scaler


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
    seed: int,
    day_value: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        if ep == 1 or ep % print_every == 0 or ep == epochs:
            logger.info(f"[day={day_value} seed={seed}] epoch {ep}/{epochs}")

        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.zeros((X.shape[0],), dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        out.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(int)


# =========================
# Masking (same semantics as your other scripts)
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
    day_info = npy_loader(args.day_info_path).astype(int)
    trial_bin = npy_loader(args.trial_bin_path)

    if trial_bin.shape[0] != sbp.shape[0]:
        raise ValueError("Length mismatch: trial_bin vs sbp")
    if sbp.shape[0] != labels.shape[0] or sbp.shape[0] != day_info.shape[0]:
        raise ValueError("Length mismatch among sbp/labels/day_info")

    target_style = None
    if args.target_style_path is not None:
        target_style = npy_loader(args.target_style_path).astype(bool)
        if target_style.shape[0] != sbp.shape[0]:
            raise ValueError("Length mismatch: target_style vs sbp")

    return LoadedData(sbp=sbp, labels=labels, day_info=day_info, trial_bin=trial_bin, target_style=target_style)


def apply_task_masks(data: LoadedData, target_type: Optional[str], allowed_labels: Optional[np.ndarray]) -> LoadedData:
    keep = np.ones(data.sbp.shape[0], dtype=bool)

    if allowed_labels is not None:
        keep &= np.isin(data.labels.astype(int), allowed_labels)

    if target_type is not None:
        if data.target_style is None:
            raise ValueError("--target_type provided but --target_style_path is None.")
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
        target_style=(data.target_style[keep] if data.target_style is not None else None),
    )

# =========================
# Saving
# =========================
def save_day_split_npz(out_results_dir: str, base: str, day_value: int, seed: int, train_trials: np.ndarray, test_trials: np.ndarray, args) -> str:
    p = os.path.join(out_results_dir, f"{base}_split_trials_day{day_value}_{args.prefix}.npz")
    np.savez(
        p,
        seed=int(seed),
        day_value=int(day_value),
        train_trials=train_trials.astype(int),
        test_trials=test_trials.astype(int),
        train_ratio=float(args.train_ratio),
        n_lags=int(args.n_lags),
        lag_step=int(args.lag_step),
        lag_group=str(args.lag_group),
        target_type=(args.target_type if args.target_type is not None else ""),
        label_mask=(args.label_mask if args.label_mask is not None else ""),
    )
    return p


def save_day_weights_npz(out_weights_dir: str, base: str, day_value: int, seed: int, model: nn.Module, hidden: List[int], n_classes: int, scaler: Optional[StandardScaler], args) -> str:
    p = os.path.join(out_weights_dir, f"{base}_weights_day{day_value}_{args.prefix}.npz")
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        p,
        seed=int(seed),
        day_value=int(day_value),
        state_dict=np.array(state, dtype=object),
        hidden_sizes=np.array(hidden, dtype=int),
        n_classes=int(n_classes),
        scale=bool(args.scale),
        log_scale=bool(args.log_scale),
        scaler_mean=(scaler.mean_ if scaler is not None else None),
        scaler_scale=(scaler.scale_ if scaler is not None else None),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        train_ratio=float(args.train_ratio),
        n_lags=int(args.n_lags),
        lag_step=int(args.lag_step),
        lag_group=str(args.lag_group),
        target_type=(args.target_type if args.target_type is not None else ""),
        label_mask=(args.label_mask if args.label_mask is not None else ""),
    )
    return p


def save_day_outputs_npz(out_results_dir: str, base: str, day_value: int, seed: int, y_true: np.ndarray, y_pred: np.ndarray, group_test: np.ndarray, metrics: Dict[str, float], args) -> str:
    p = os.path.join(out_results_dir, f"{base}_outputs_day{day_value}_{args.prefix}.npz")
    np.savez(
        p,
        seed=int(seed),
        day_value=int(day_value),
        y_true=y_true.astype(np.int16),
        y_pred=y_pred.astype(np.int16),
        trial_id=group_test.astype(np.int32),
        **{k: float(v) for k, v in metrics.items()},
    )
    return p


# =========================
# Args
# =========================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--trial_bin_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--target_type", type=str, default=None)  # center-out/random
    ap.add_argument("--target_style_path", type=str, default=None)
    ap.add_argument("--label_mask", type=str, default=None)

    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--log_scale", action="store_true")

    ap.add_argument("--train_ratio", type=float, default=0.8)

    ap.add_argument("--hidden_sizes", type=str, default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=10)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, required=True)

    ap.add_argument("--n_lags", type=int, default=0)
    ap.add_argument("--lag_step", type=int, default=1)
    ap.add_argument("--lag_group", type=str, default="trial", choices=["trial"])

    ap.add_argument(
        "--best_param",
        type=str,
        default=None,
        help="Optuna best_params.json (optional). If given, overrides hidden/lr/wd/batch/n_lags/lag_step",
    )
    return ap


# =========================
# Main
# =========================
def main() -> None:
    args = build_argparser().parse_args()

    # optional: override by HPO best params (same style as your scripts)
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

    if args.hidden_sizes is None or args.hidden_sizes.strip() == "":
        raise ValueError("Provide --hidden_sizes like '256,128' or --best_param JSON.")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")

    # enforce trial-safe lag behavior
    if args.lag_group != "trial":
        raise ValueError("This per-day trainer enforces trial-safe lag: set --lag_group trial.")

    outp = make_outdirs(args.out_dir)
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(int(args.seed))
    hidden = parse_hidden_sizes(args.hidden_sizes)
    hs_tag = hidden_tag(hidden)

    # load + mask
    data0 = load_all_arrays(args)
    allowed = parse_label_mask(args.label_mask)
    data = apply_task_masks(data0, target_type=args.target_type, allowed_labels=None)

    X_all = data.sbp.astype(np.float32)
    y_all = data.labels.astype(int)
    day_all = data.day_info.astype(int)
    trial_bin_all = data.trial_bin

    uniq_days = np.unique(day_all)
    uniq_days = np.sort(uniq_days)
    
    #limit day for small run
    uniq_days = uniq_days[:50]
    logger.info(f"[All days] n_days={len(uniq_days)}  days={uniq_days.tolist()}")

    base_common = f"{args.prefix}_{hs_tag}_seed{int(args.seed)}"

    # collect per-day metrics
    day_list = []
    acc_list = []
    bacc_list = []
    f1_list = []

    for d in uniq_days:
        # ---- subset one day
        m = (day_all == int(d))
        X_day0 = X_all[m]
        y_day0 = y_all[m]
        tb_day0 = trial_bin_all[m]

        # ---- trial ids (within this day)
        trial_id_day0, n_trials_raw = build_trial_ids(tb_day0)
        if n_trials_raw < 2:
            logger.warning(f"[Day {int(d)}] too few trials ({n_trials_raw}); skip")
            continue

        # ---- lag (trial-safe)
        X_lag, y_lag, group_eff = make_lagged_features(
            X=X_day0,
            y=y_day0,
            group=trial_id_day0,   # trial-safe
            n_lags=int(args.n_lags),
            lag_step=int(args.lag_step),
        )
        if X_lag.shape[0] < 2:
            logger.warning(f"[Day {int(d)}] too few samples after lag; skip")
            continue

        group_eff = np.asarray(group_eff).astype(int)

        # ==========================================
        # NEW: FILTER CLASS AND REMAP AFTER LAGGING
        # ==========================================
        if allowed is not None:
            valid_idx = np.isin(y_lag, allowed)
            X_lag = X_lag[valid_idx]
            y_lag = y_lag[valid_idx]
            group_eff = group_eff[valid_idx]

        if X_lag.shape[0] < 2:
            logger.warning(f"[Day {int(d)}] too few samples after masking; skip")
            continue

        # Remap labels to be contiguous (e.g., [0, 2] becomes [0, 1])
        unique_labels = np.sort(np.unique(y_lag))
        remapper = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
        y_lag = np.vectorize(remapper.get)(y_lag)
        # ==========================================

        valid_trials = np.unique(group_eff)
        if valid_trials.size < 2:
            logger.warning(f"[Day {int(d)}] too few valid trials after lag/mask; skip")
            continue

        n_classes = int(np.unique(y_lag).size)
        cls, cnt = np.unique(y_lag, return_counts=True)
        logger.info(f"[Day {int(d)}] N={len(y_lag)} Nclasses={n_classes} label_counts={dict(zip(cls.tolist(), cnt.tolist()))}")

        # ---- trial-wise split (deterministic by seed+day)
        train_trials, test_trials = split_group_ids(
            group_eff,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed) + int(d),
            y=y_lag,
            stratify=True,
        )

        tr_mask = np.isin(group_eff, train_trials)
        te_mask = np.isin(group_eff, test_trials)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            logger.warning(f"[Day {int(d)}] empty train/test after split; skip")
            continue

        X_train_raw = X_lag[tr_mask].copy()
        y_train = y_lag[tr_mask].copy()
        X_test_raw = X_lag[te_mask].copy()
        y_test = y_lag[te_mask].copy()
        group_test = group_eff[te_mask].copy()

        # ---- scaler fit on TRAIN only
        X_train, X_test, scaler = preprocess_train_test(
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            scale=bool(args.scale),
            log_scale=bool(args.log_scale),
        )

        # ---- model train
        model = MLP(in_dim=X_train.shape[1], hidden=hidden, out_dim=n_classes).to(device)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
            batch_size=int(args.batch_size),
            shuffle=True,
        )
        train_mlp(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            print_every=int(args.print_every),
            seed=int(args.seed),
            day_value=int(d),
        )

        # ---- eval
        y_pred = predict_labels(model, X_test, device=device, batch_size=int(args.batch_size))
        class_ids = np.unique(y_test).astype(int)

        acc = float(np.mean(y_pred == y_test))
        bacc = float(balanced_accuracy_macro(y_test, y_pred, class_ids))
        f1 = float(f1_macro(y_test, y_pred, class_ids))

        logger.info(f"[Day {int(d)} TEST] N={len(y_test)} acc={acc:.4f} bAcc={bacc:.4f} f1={f1:.4f}  (#test_trials={len(test_trials)})")

        # ---- save artifacts
        base_day = f"{base_common}_Nclasses{n_classes}_day{int(d)}"

        split_path = save_day_split_npz(
            out_results_dir=outp.results,
            base=base_day,
            day_value=int(d),
            seed=int(args.seed),
            train_trials=train_trials,
            test_trials=test_trials,
            args=args,
        )
        w_path = save_day_weights_npz(
            out_weights_dir=outp.weights,
            base=base_day,
            day_value=int(d),
            seed=int(args.seed),
            model=model,
            hidden=hidden,
            n_classes=n_classes,
            scaler=scaler,
            args=args,
        )
        out_path = save_day_outputs_npz(
            out_results_dir=outp.results,
            base=base_day,
            day_value=int(d),
            seed=int(args.seed),
            y_true=y_test,
            y_pred=y_pred,
            group_test=group_test,
            metrics={"acc": acc, "bacc": bacc, "f1": f1},
            args=args,
        )

        logger.info(f"[Saved] split:   {split_path}")
        logger.info(f"[Saved] weights: {w_path}")
        logger.info(f"[Saved] outputs: {out_path}")

        # collect
        day_list.append(int(d))
        acc_list.append(acc)
        bacc_list.append(bacc)
        f1_list.append(f1)

    # ---- save per-day summary
    day_arr = np.array(day_list, dtype=int)
    acc_arr = np.array(acc_list, dtype=float)
    bacc_arr = np.array(bacc_list, dtype=float)
    f1_arr = np.array(f1_list, dtype=float)

    summ_path = os.path.join(outp.results, f"{base_common}_perday_TESTMETRICS.npz")
    np.savez(
        summ_path,
        seed=int(args.seed),
        days=day_arr,
        test_acc=acc_arr,
        test_bacc=bacc_arr,
        test_f1=f1_arr,
        train_ratio=float(args.train_ratio),
        n_lags=int(args.n_lags),
        lag_step=int(args.lag_step),
        lag_group=str(args.lag_group),
        hidden_sizes=np.array(hidden, dtype=int),
        target_type=(args.target_type if args.target_type is not None else ""),
        label_mask=(args.label_mask if args.label_mask is not None else ""),
    )
    print("[Saved] per-day metrics ->", summ_path)
    for d, a, b, f in zip(day_arr.tolist(), acc_arr.tolist(), bacc_arr.tolist(), f1_arr.tolist()):
        print(f"  day={d}  acc={a:.4f}  bAcc={b:.4f}  f1={f:.4f}")


if __name__ == "__main__":
    main()