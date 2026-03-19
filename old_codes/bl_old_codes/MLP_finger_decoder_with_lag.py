#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def npy_loader(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def parse_label_mask(s: str | None) -> np.ndarray | None:
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([int(p) for p in parts], dtype=int)


def apply_masks(sbp, labels, day_info, target_style, target_type, label_mask_allowed):
    # 1) label_mask
    if label_mask_allowed is not None:
        keep = np.isin(labels.astype(int), label_mask_allowed)
        sbp = sbp[keep]
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
        sbp = sbp[keep]
        labels = labels[keep]
        day_info = day_info[keep]
        target_style = t[keep]

    return sbp, labels, day_info, target_style


def plot_label_distribution(y_train: np.ndarray, out_png: str, n_classes: int) -> None:
    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    plt.figure()
    plt.bar(classes.astype(str), counts)
    plt.xlabel("Class label")
    plt.ylabel("Count (train)")
    plt.title(f"Train label distribution (Nclasses={n_classes})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def per_day_accuracy(day_info_i, mask, y_pred, y_true):
    # mask is the boolean mask applied to the global arrays (same length as day_info_i)
    days = day_info_i[mask]
    uniq_days = np.unique(days)
    day_acc = []
    day_n = []
    for d in uniq_days:
        idx = (days == d)
        day_n.append(int(idx.sum()))
        day_acc.append(float((y_pred[idx] == y_true[idx]).mean()))
    return uniq_days, np.array(day_acc, float), np.array(day_n, int)


def plot_per_day_mean_std(days_sorted, mean_acc, std_acc, n_classes, out_png):
    plt.figure()
    plt.plot(days_sorted, mean_acc, marker="o")
    plt.fill_between(days_sorted, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    plt.xlabel("Day")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Per-day accuracy mean±std over seeds (Nclasses={n_classes})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def parse_hidden_sizes(s: str) -> list[int]:
    s = s.strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]



def make_lagged_features(X: np.ndarray,
                         y: np.ndarray,
                         group: np.ndarray | None,
                         n_lags: int,
                         lag_step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    X: (N, D)
    y: (N,)
    group: (N,) grouping id to avoid crossing (e.g., day). If None, treat as one group.
    Returns:
      X_lag: (N_eff, D*(n_lags+1)) = [x[t], x[t-lag_step], ..., x[t-n_lags*lag_step]]
      y_eff: (N_eff,) aligned with x[t]
      group_eff: (N_eff,) aligned
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

        # idx は元配列の連番である前提（dayで切っているので通常OK）
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--trial_bin_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)

    ap.add_argument("--slicing_day", type=int, required=True)  # 1-indexed among unique days (same as original)
    ap.add_argument("--target_type", type=str, default=None)  # center-out / random
    ap.add_argument("--target_style_path", type=str, default=None)

    ap.add_argument("--label_mask", type=str, default=None)

    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--log_scale", action="store_true")

    # NEW: within-train split
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Shuffle split ratio within sliced data")

    # torch MLP training
    ap.add_argument("--hidden_sizes", type=str)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=1)

    ap.add_argument("--out_dir", type=str)
    ap.add_argument("--prefix", type=str)

    ap.add_argument("--n_lags", type=int, default=0, help="Number of past lags to stack (0=off)")
    ap.add_argument("--lag_step", type=int, default=1, help="Step between lags in samples (1=adjacent)")
    ap.add_argument("--lag_group", type=str, default="trial", choices=["none", "day", "trial"],
                help="Prevent lag crossing boundaries: trial recommended for concatenated-success data")
    

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")

    device = get_device()
    logger.info(f"Using device: {device}")

    # ---- load ----
    sbp = npy_loader(args.sbp_path)            # (N, D)
    labels = npy_loader(args.label_path)       # (N,)
    day_info = npy_loader(args.day_info_path)  # (N,)
    time_within_trial = npy_loader(args.trial_bin_path)  # (N,) float, 0.0 at trial start
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

    # ---- masks (same logic) ----
    keep = np.ones(sbp.shape[0], dtype=bool)

    # 1) label_mask
    if allowed is not None:
        keep &= np.isin(labels.astype(int), allowed)

    # 2) target_type masking (center-out => ~t, random => t)
    if args.target_type is not None:
        if target_style is None:
            raise ValueError("--target_type was provided but --target_style_path is None.")
        t = np.asarray(target_style).astype(bool)
        if args.target_type == "center-out":
            keep &= ~t
        elif args.target_type == "random":
            keep &= t
        else:
            raise ValueError("--target_type must be one of: center-out, random")

    # apply
    sbp = sbp[keep]
    labels = labels[keep]
    day_info = day_info[keep]
    time_within_trial = time_within_trial[keep]
    if target_style is not None:
        target_style = np.asarray(target_style).astype(bool)[keep]

    # ---- X/y ----
    X = sbp.astype(np.float32)
    y = labels.astype(int)
    day_info_i = np.asarray(day_info).astype(int)

    uniq_all_days = np.unique(day_info_i)
    if args.slicing_day < 1 or args.slicing_day > len(uniq_all_days):
        raise ValueError(f"--slicing_day must be in [1, {len(uniq_all_days)}]")

    slicing_day_value = uniq_all_days[args.slicing_day - 1]
    print(f"Slicing day value: {slicing_day_value}  (args.slicing_day={args.slicing_day})")

    # (1) use only data up to slicing_day_value (NO future-day test)
    use_mask = day_info_i <= slicing_day_value
    time_use = time_within_trial[use_mask]

    trial_starts = (time_use == 0.0)
    if len(time_use) > 0 and (time_use[0] != 0.0):
        trial_starts[0] = True

    trial_id_use = np.cumsum(trial_starts).astype(int) - 1
    print("count_1 (trials within used data):", int(trial_starts.sum()))

    X_use = X[use_mask]
    y_use = y[use_mask]
    day_use = day_info_i[use_mask]

    # ---- lag stacking (adds temporal context) ----
    group = None
    if args.lag_group == "day":
        group = day_use
    elif args.lag_group == "trial":
        group = trial_id_use

    X_use, y_use, day_use = make_lagged_features(
        X=X_use, y=y_use, group=group, n_lags=args.n_lags, lag_step=args.lag_step
    )

    if X_use.shape[0] == 0:
        raise ValueError("No samples after applying slicing_day & masks.")

    # label dist (within used data)
    n_classes = int(np.unique(y_use).size)
    print(f"[Used] N={len(y_use)}  Nclasses={n_classes}")
    cls, cnt = np.unique(y_use, return_counts=True)
    print("[Used] label counts:", dict(zip(cls.tolist(), cnt.tolist())))
    plot_label_distribution(
        y_use,
        os.path.join(args.out_dir, f"{args.prefix}_labeldist_Nclasses{n_classes}_day{args.slicing_day}.png"),
        n_classes=n_classes,
    )

    # preprocessing: fit scaler on TRAIN ONLY (after shuffle split)
    hidden = parse_hidden_sizes(args.hidden_sizes)
    seeds = list(range(1)) #0

    seed_to_dayacc = {}
    seed_to_overall_acc = {}

    N = X_use.shape[0]
    n_train = int(round(N * args.train_ratio))
    n_train = max(1, min(n_train, N - 1))  # ensure both non-empty

    for seed in seeds:
        set_seed(seed)

        # (2) shuffle split within used data
        perm = np.random.permutation(N)
        tr_idx = perm[:n_train]
        va_idx = perm[n_train:]

        X_train_raw = X_use[tr_idx].copy()
        y_train = y_use[tr_idx].copy()
        X_val_raw = X_use[va_idx].copy()
        y_val = y_use[va_idx].copy()

        scaler = None
        X_train = X_train_raw
        X_val = X_val_raw
        if args.scale:
            if args.log_scale:
                X_train = np.log1p(X_train)
                X_val = np.log1p(X_val)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_val = scaler.transform(X_val).astype(np.float32)
        else:
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)

        model = MLP(in_dim=X_train.shape[1], hidden=hidden, out_dim=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=True,
        )

        for ep in range(1, args.epochs + 1):
            print(f"[seed={seed}] Epoch {ep}/{args.epochs}")
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

            if ep == 1 or ep % args.print_every == 0 or ep == args.epochs:
                logger.info(f"[seed={seed}] epoch {ep:3d}/{args.epochs} loss={total_loss/max(n_seen,1):.6f}")

        # (3) evaluate on val split only, then exit after saving
        model.eval()
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=False,
        )

        y_pred_list = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=False)
                logits = model(xb)
                y_pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())

        y_pred = np.concatenate(y_pred_list, axis=0).astype(int)
        correct01 = (y_pred == y_val).astype(np.int8)
        acc = float(correct01.mean())
        seed_to_overall_acc[seed] = acc
        print(f"[Val][seed={seed}] N={len(y_val)} accuracy={acc:.4f}")

        # per-day accuracy within the used range (on val subset only)
        day_val = day_use[va_idx]
        uniq_days, day_acc, day_n = per_day_accuracy(day_val, np.ones_like(day_val, dtype=bool), y_pred, y_val)
        seed_to_dayacc[seed] = {int(d): float(a) for d, a in zip(uniq_days, day_acc)}

        # save weights (same style)
        w_path = os.path.join(
            args.out_dir,
            f"{args.prefix}_seed{seed}_weights_Nclasses{n_classes}_day{args.slicing_day}.npz"
        )
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
        )

        perf_path = os.path.join(
            args.out_dir,
            f"{args.prefix}_seed{seed}_val_correct01_Nclasses{n_classes}_day{args.slicing_day}.npy"
        )
        np.save(perf_path, correct01)

        day_path = os.path.join(
            args.out_dir,
            f"{args.prefix}_seed{seed}_perday_valacc_Nclasses{n_classes}_day{args.slicing_day}.npz"
        )
        np.savez(
            day_path,
            seed=seed,
            days=uniq_days.astype(int),
            acc=day_acc.astype(float),
            n=day_n.astype(int),
            y_pred=y_pred.astype(np.int16),
            y_true=y_val.astype(np.int16),
            day_of_each_sample=day_val.astype(np.int16),
        )

    # aggregate mean/std across seeds (over val-days within used range)
    all_days = sorted({d for seed in seeds for d in seed_to_dayacc[seed].keys()})
    all_days = np.array(all_days, dtype=int)

    A = np.full((len(seeds), len(all_days)), np.nan, dtype=float)
    for si, seed in enumerate(seeds):
        d2a = seed_to_dayacc[seed]
        for di, d in enumerate(all_days):
            if d in d2a:
                A[si, di] = d2a[d]

    mean_acc = np.nanmean(A, axis=0)
    std_acc = np.nanstd(A, axis=0)

    agg_path = os.path.join(
        args.out_dir,
        f"{args.prefix}_seeds0-9_perday_valmeanstd_Nclasses{n_classes}_day{args.slicing_day}.npz"
    )
    np.savez(
        agg_path,
        seeds=np.array(seeds, dtype=int),
        days=all_days,
        perday_acc_matrix=A,
        mean=mean_acc,
        std=std_acc,
        overall_acc=np.array([seed_to_overall_acc[s] for s in seeds], dtype=float),
        train_ratio=float(args.train_ratio),
    )

    agg_png = os.path.join(
        args.out_dir,
        f"{args.prefix}_seeds0-9_perday_valmeanstd_Nclasses{n_classes}_day{args.slicing_day}.png"
    )
    plot_per_day_mean_std(all_days, mean_acc, std_acc, n_classes=n_classes, out_png=agg_png)

    print("Saved (aggregate):")
    print(" ", agg_path)
    print(" ", agg_png)


if __name__ == "__main__":
    main()
