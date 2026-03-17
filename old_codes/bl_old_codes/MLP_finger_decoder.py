#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def npy_loader(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def assemble_features(neural_data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if neural_data.shape[0] != labels.shape[0]:
        raise ValueError(f"Length mismatch: sbp={neural_data.shape[0]} labels={labels.shape[0]}")
    return np.concatenate([neural_data, labels], axis=1)


def parse_label_mask(s: str | None) -> np.ndarray | None:
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([int(p) for p in parts], dtype=int)


def apply_masks(
    sbp: np.ndarray,
    labels: np.ndarray,
    day_info: np.ndarray,
    target_style: np.ndarray | None,
    target_type: str | None,
    label_mask_allowed: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    # 1) label_mask
    if label_mask_allowed is not None:
        keep = np.isin(labels.astype(int), label_mask_allowed)
        sbp = sbp[keep]
        labels = labels[keep]
        day_info = day_info[keep]
        if target_style is not None:
            target_style = target_style[keep]

    # 2) target_type masking (MATCHES your perceptron code)
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
    y_train = y_train.astype(int)
    classes, counts = np.unique(y_train, return_counts=True)

    plt.figure()
    plt.bar(classes.astype(str), counts)
    plt.xlabel("Class label")
    plt.ylabel("Count (train)")
    plt.title(f"Train label distribution (Nclasses={n_classes})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def per_day_accuracy(day_info_i: np.ndarray, test_mask: np.ndarray, y_pred: np.ndarray, y_test: np.ndarray):
    test_days = day_info_i[test_mask]
    uniq_days = np.unique(test_days)

    day_acc = []
    day_n = []
    for d in uniq_days:
        idx = (test_days == d)
        day_n.append(int(idx.sum()))
        day_acc.append(float((y_pred[idx] == y_test[idx]).mean()))
    return uniq_days, np.array(day_acc, dtype=float), np.array(day_n, dtype=int)


def plot_per_day_mean_std(days_sorted: np.ndarray, mean_acc: np.ndarray, std_acc: np.ndarray,
                          n_classes: int, out_png: str):
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


def parse_hidden_sizes(s: str) -> tuple[int, ...]:
    s = s.strip()
    if s == "":
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def main() -> None:
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)

    ap.add_argument("--slicing_day", type=int, required=True,
                    help="Train uses day_info <= slicing_day; test uses day_info > slicing_day.")
    ap.add_argument("--target_type", type=str, default=None, help="Options: center-out, random")
    ap.add_argument("--target_style_path", type=str, default=None,
                    help="Path to target_style.npy (bool-like). Required if --target_type is set.")

    ap.add_argument("--label_mask", type=str, default=None,
                    help='Comma-separated allowed labels, e.g. "0,1,2". If omitted, keep all.')

    # preprocessing (kept compatible)
    ap.add_argument("--scale", action="store_true",
                    help="If set, standardize X using StandardScaler fit on train only.")
    ap.add_argument("--log_scale", action="store_true")

    # MLP settings (simple)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--hidden_sizes", type=str, default="256,128")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)

    # outputs
    ap.add_argument("--out_dir", type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP")
    ap.add_argument("--prefix", type=str, default="mlp")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load ----
    sbp = npy_loader(args.sbp_path)
    labels = npy_loader(args.label_path)
    day_info = npy_loader(args.day_info_path)

    if sbp.shape[0] != labels.shape[0] or sbp.shape[0] != day_info.shape[0]:
        raise ValueError(
            f"Length mismatch: sbp={sbp.shape[0]} labels={labels.shape[0]} day_info={day_info.shape[0]}"
        )

    target_style = None
    if args.target_style_path is not None:
        target_style = npy_loader(args.target_style_path)
        if target_style.shape[0] != sbp.shape[0]:
            raise ValueError(f"Length mismatch: target_style={target_style.shape[0]} vs sbp={sbp.shape[0]}")
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

    # ---- X/y (same logic) ----
    labeled_sbp = assemble_features(sbp, labels)
    X = labeled_sbp[:, :-1]
    y = labeled_sbp[:, -1].astype(int)

    # ---- split (same logic) ----
    day_info_i = np.asarray(day_info).astype(int)
    train_mask = day_info_i <= args.slicing_day
    test_mask = day_info_i > args.slicing_day

    X_train_raw, y_train = X[train_mask], y[train_mask]
    X_test_raw, y_test = X[test_mask], y[test_mask]

    if X_train_raw.shape[0] == 0 or X_test_raw.shape[0] == 0:
        raise ValueError(
            f"Empty split: train={X_train_raw.shape[0]} test={X_test_raw.shape[0]}. "
            f"Check --slicing_day={args.slicing_day} vs day_info range."
        )

    # ---- label distribution (once) ----
    n_classes = int(np.unique(y_train).size)
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"[Train] N={len(y_train)}  Nclasses={n_classes}")
    print("[Train] label counts:", dict(zip(classes.tolist(), counts.tolist())))

    dist_png = os.path.join(args.out_dir, f"{args.prefix}_labeldist_Nclasses{n_classes}_day{args.slicing_day}.png")
    plot_label_distribution(y_train, dist_png, n_classes=n_classes)

    # ---- preprocessing computed ONCE (same as your final perceptron script) ----
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    scaler = None
    if args.scale:
        if args.log_scale:
            logger.info("Applying log1p scaling to features.")
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)

        scaler = StandardScaler()
        logger.info(f"before scaling: {float(X_train.min())} {float(X_train.mean())} {float(X_train.max())}")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info(f"after scaling: {float(X_train.min())} {float(X_train.mean())} {float(X_train.max())}")

    # ---- multi-seed ----
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    seeds = list(range(10))
    seed_to_dayacc = {}
    seed_to_overall_acc = {}

    for seed in seeds:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu", #need to change according to AGREL or HRL
            solver="adam",
            alpha=args.alpha,
            learning_rate_init=args.lr,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            random_state=seed,
            early_stopping=True,  # keep simple & deterministic
            verbose=True
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test).astype(int)
        correct01 = (y_pred == y_test).astype(np.int8)
        acc = float(correct01.mean())
        seed_to_overall_acc[seed] = acc
        print(f"[Test][seed={seed}] days={args.slicing_day}, N={len(y_test)} accuracy={acc:.4f}")

        uniq_days, day_acc, day_n = per_day_accuracy(day_info_i, test_mask, y_pred, y_test)
        seed_to_dayacc[seed] = {int(d): float(a) for d, a in zip(uniq_days, day_acc)}

        # ---- save per-seed outputs (same naming style; just 'mlp') ----
        weights_path = os.path.join(
            args.out_dir,
            f"{args.prefix}_seed{seed}_weights_Nclasses{n_classes}_day{args.slicing_day}.npz"
        )
        # MLP weights are lists of arrays (coefs_ / intercepts_)
        np.savez(
            weights_path,
            seed=seed,
            coefs=np.array(clf.coefs_, dtype=object),
            intercepts=np.array(clf.intercepts_, dtype=object),
            classes_=clf.classes_,
            slicing_day=args.slicing_day,
            target_type=args.target_type,
            label_mask=args.label_mask,
            scale=args.scale,
            log_scale=args.log_scale,
            hidden_sizes=np.array(hidden_sizes, dtype=int),
            alpha=float(args.alpha),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            scaler_mean=(scaler.mean_ if scaler is not None else None),
            scaler_scale=(scaler.scale_ if scaler is not None else None),
        )

        perf_path = os.path.join(
            args.out_dir,
            f"{args.prefix}_seed{seed}_test_correct01_Nclasses{n_classes}_day{args.slicing_day}.npy"
        )
        np.save(perf_path, correct01)

        day_path = os.path.join(
            args.out_dir,
            f"{args.prefix}_seed{seed}_perday_acc_Nclasses{n_classes}_day{args.slicing_day}.npz"
        )
        np.savez(
            day_path,
            seed=seed,
            days=uniq_days.astype(int),
            acc=day_acc.astype(float),
            n=day_n.astype(int),
        )

    # ---- aggregate mean/std across seeds ----
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
        f"{args.prefix}_seeds0-9_perday_meanstd_Nclasses{n_classes}_day{args.slicing_day}.npz"
    )
    np.savez(
        agg_path,
        seeds=np.array(seeds, dtype=int),
        days=all_days,
        perday_acc_matrix=A,
        mean=mean_acc,
        std=std_acc,
        overall_acc=np.array([seed_to_overall_acc[s] for s in seeds], dtype=float),
    )

    agg_png = os.path.join(
        args.out_dir,
        f"{args.prefix}_seeds0-9_perday_meanstd_Nclasses{n_classes}_day{args.slicing_day}.png"
    )
    plot_per_day_mean_std(all_days, mean_acc, std_acc, n_classes=n_classes, out_png=agg_png)

    print("Saved (aggregate):")
    print(" ", agg_path)
    print(" ", agg_png)


if __name__ == "__main__":
    main()
