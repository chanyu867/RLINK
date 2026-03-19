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

    # 2) target_type masking (MATCHES your current code: center-out => ~t, random => t)
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


def per_day_accuracy(day_info_i, test_mask, y_pred, y_test):
    test_days = day_info_i[test_mask]
    uniq_days = np.unique(test_days)
    day_acc = []
    day_n = []
    for d in uniq_days:
        idx = (test_days == d)
        day_n.append(int(idx.sum()))
        day_acc.append(float((y_pred[idx] == y_test[idx]).mean()))
    return uniq_days, np.array(day_acc, float), np.array(day_n, int)


def plot_per_day_mean_std(days_sorted, mean_acc, std_acc, n_classes, out_png):
    plt.figure()
    plt.plot(days_sorted, mean_acc, marker="o")
    plt.fill_between(days_sorted, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    plt.xlabel("Day")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Per-day test accuracy mean±std over seeds (Nclasses={n_classes})")
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)

    ap.add_argument("--slicing_day", type=int, required=True)
    ap.add_argument("--target_type", type=str, default=None)  # center-out / random
    ap.add_argument("--target_style_path", type=str, default=None)

    ap.add_argument("--label_mask", type=str, default=None)

    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--log_scale", action="store_true")

    # torch MLP training
    ap.add_argument("--hidden_sizes", type=str, default="256,128")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=1)

    ap.add_argument("--out_dir", type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP")
    ap.add_argument("--prefix", type=str, default="mlp_torch")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    logger.info(f"Using device: {device}")

    # ---- load ----
    sbp = npy_loader(args.sbp_path)            # (N, D)
    labels = npy_loader(args.label_path)       # (N,)
    day_info = npy_loader(args.day_info_path)  # (N,)

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
    sbp, labels, day_info, target_style = apply_masks(
        sbp=sbp,
        labels=labels,
        day_info=day_info,
        target_style=target_style,
        target_type=args.target_type,
        label_mask_allowed=allowed,
    )

    # ---- X/y ----
    X = sbp.astype(np.float32)
    y = labels.astype(int)
    day_info_i = np.asarray(day_info).astype(int)

    slicing_day = np.unique(day_info_i)[args.slicing_day-1]
    print(f"Slicing day: {slicing_day}, {args.slicing_day} th day slicing")

    # ---- split (same logic as your code) ----
    train_mask = day_info_i <= slicing_day
    test_mask = day_info_i > slicing_day

    X_train_raw, y_train = X[train_mask], y[train_mask]
    X_test_raw, y_test = X[test_mask], y[test_mask]

    if X_train_raw.shape[0] == 0 or X_test_raw.shape[0] == 0:
        raise ValueError("Empty train or test split. Check slicing_day.")

    # label dist
    n_classes = int(np.unique(y_train).size)
    print(f"[Train] N={len(y_train)}  Nclasses={n_classes}")
    cls, cnt = np.unique(y_train, return_counts=True)
    print("[Train] label counts:", dict(zip(cls.tolist(), cnt.tolist())))
    plot_label_distribution(
        y_train,
        os.path.join(args.out_dir, f"{args.prefix}_labeldist_Nclasses{n_classes}_day{args.slicing_day}.png"),
        n_classes=n_classes,
    )

    # preprocessing (computed once; same for all seeds)
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    scaler = None
    if args.scale:
        if args.log_scale:
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)
        scaler = StandardScaler()
        #do transform separately on purpose
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    # ---- multi-seed ----
    hidden = parse_hidden_sizes(args.hidden_sizes)
    seeds = list(range(10))
    seed_to_dayacc = {}
    seed_to_overall_acc = {}

    for seed in seeds:
        set_seed(seed)

        model = MLP(in_dim=X_train.shape[1], hidden=hidden, out_dim=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=True,
        )

        # train epochs with progress
        for ep in range(1, args.epochs + 1):
            print(f"Epoch {ep}/{args.epochs}")
            model.train()
            total_loss = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device) #loading data on mps
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

        # test using mini-batches for safety
        model.eval()
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test.astype(np.int64))),
            batch_size=args.batch_size,
            shuffle=False,
        )

        y_pred_list = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device, non_blocking=False)
                logits = model(xb)
                y_pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())

        y_pred = np.concatenate(y_pred_list, axis=0).astype(int)

        correct01 = (y_pred == y_test).astype(np.int8)
        acc = float(correct01.mean())
        seed_to_overall_acc[seed] = acc
        print(f"[Test][seed={seed}] day={args.slicing_day}, N={len(y_test)} accuracy={acc:.4f}")

        uniq_days, day_acc, day_n = per_day_accuracy(day_info_i, test_mask, y_pred, y_test)
        seed_to_dayacc[seed] = {int(d): float(a) for d, a in zip(uniq_days, day_acc)}

        # save per-seed weights + perf (same naming style)
        w_path = os.path.join(args.out_dir, f"seed{seed}_weights_Nclasses{n_classes}_day{args.slicing_day}.npz")
        state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez(
            w_path,
            seed=seed,
            state_dict=np.array(state, dtype=object),
            slicing_day=args.slicing_day,
            target_type=args.target_type,
            label_mask=args.label_mask,
            scale=args.scale,
            log_scale=args.log_scale,
            hidden_sizes=np.array(hidden, dtype=int),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            scaler_mean=(scaler.mean_ if scaler is not None else None),
            scaler_scale=(scaler.scale_ if scaler is not None else None),
        )

        perf_path = os.path.join(args.out_dir, f"{args.prefix}_seed{seed}_test_correct01_Nclasses{n_classes}_day{args.slicing_day}.npy")
        np.save(perf_path, correct01)

        day_path = os.path.join(args.out_dir, f"{args.prefix}_seed{seed}_perday_acc_Nclasses{n_classes}_day{args.slicing_day}.npz")
        np.savez(day_path, seed=seed, days=uniq_days.astype(int), acc=day_acc.astype(float), n=day_n.astype(int), y_pred=y_pred.astype(np.int16), y_true=y_test.astype(np.int16))
    # aggregate mean/std across seeds
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

    agg_path = os.path.join(args.out_dir, f"{args.prefix}_seeds0-9_perday_meanstd_Nclasses{n_classes}_day{args.slicing_day}.npz")
    np.savez(
        agg_path,
        seeds=np.array(seeds, dtype=int),
        days=all_days,
        perday_acc_matrix=A,
        mean=mean_acc,
        std=std_acc,
        overall_acc=np.array([seed_to_overall_acc[s] for s in seeds], dtype=float),
    )

    agg_png = os.path.join(args.out_dir, f"{args.prefix}_seeds0-9_perday_meanstd_Nclasses{n_classes}_day{args.slicing_day}.png")
    plot_per_day_mean_std(all_days, mean_acc, std_acc, n_classes=n_classes, out_png=agg_png)

    print("Saved (aggregate):")
    print(" ", agg_path)
    print(" ", agg_png)


if __name__ == "__main__":
    main()
