#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlp_train as mt  # uses your attached training code :contentReference[oaicite:1]{index=1}


# -------------------------
# Search ranges (reasonable defaults)
# -------------------------
H1_CHOICES = [64, 128, 256, 512, 1024]
H2_CHOICES = [0, 64, 128, 256, 512]  # 0 => no 2nd layer (i.e., 1 hidden layer)
BATCH_CHOICES = [64, 256]

LR_MIN, LR_MAX = 1e-5, 3e-3
WD_MIN, WD_MAX = 1e-8, 1e-2
NLAGS_MIN, NLAGS_MAX = 0, 16
LAGSTEP_MIN, LAGSTEP_MAX = 1, 4


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Required data inputs (same meaning as in mlp_train.py)
    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--trial_bin_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)

    # slicing (same meaning as mlp_train.py: 1-indexed among unique days)
    ap.add_argument("--slicing_day", type=int, required=True)

    # evaluation / budget
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=20, help="Max epochs per trial (pruning may stop earlier).")
    ap.add_argument("--eval_every", type=int, default=1, help="Validate every k epochs for pruning.")

    # Optuna
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--study_name", type=str, default="mlp_hpo")
    ap.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///mlp_hpo.db (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_style_path", type=str, default=None,
                help="target_style.npy path required when using --target_type (random/center-out)")


    # output
    ap.add_argument("--save_best_json", type=str, default="best_params.json")
    return ap


def suggest_hidden_sizes(trial):
    h1 = trial.suggest_categorical("h1", H1_CHOICES)
    h2 = trial.suggest_categorical("h2", H2_CHOICES)  # ← choicesは固定にする（重要）

    if h2 != 0 and h2 > h1:
        raise optuna.TrialPruned(f"Invalid hidden sizes: h2({h2}) > h1({h1})")

    hidden = [h1] if h2 == 0 else [h1, h2]
    return hidden

def make_train_val_split(
    X_use: np.ndarray,
    y_use: np.ndarray,
    day_vec_for_split: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    
    
    Shuffle split identical spirit to mlp_train.py but controlled by seed.
    Returns X_tr_raw, y_tr, X_va_raw, y_va, day_va
    """
    N = X_use.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)

    n_train = int(round(N * train_ratio))
    n_train = max(1, min(n_train, N - 1))

    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]
    return X_use[tr_idx], y_use[tr_idx], X_use[va_idx], y_use[va_idx], day_vec_for_split[va_idx]


def objective_factory(cli_args):
    # ---------- fixed (as you requested) ----------
    FIX_LAG_GROUP = "trial"
    FIX_SCALE = True
    FIX_LOGSCALE = False
    # data selection all fixed => no label_mask / no target_type
    FIX_TARGET_TYPE = "random"
    FIX_ALLOWED_LABELS = None

    # ---------- load + prepare ONCE ----------
    # Create an args-like object compatible with mt.load_all_arrays
    class A:  # simple namespace
        pass

    a = A()
    a.sbp_path = cli_args.sbp_path
    a.trial_bin_path = cli_args.trial_bin_path
    a.label_path = cli_args.label_path
    a.day_info_path = cli_args.day_info_path
    a.target_style_path = cli_args.target_style_path

    data0 = mt.load_all_arrays(a)
    data = mt.apply_task_masks(data0, target_type=FIX_TARGET_TYPE, allowed_labels=FIX_ALLOWED_LABELS)

    X = data.sbp.astype(np.float32)
    y = data.labels.astype(int)
    day_info_i = np.asarray(data.day_info).astype(int)

    slicing_day_value = mt.compute_slicing_day_value(day_info_i, cli_args.slicing_day)

    use_mask = (day_info_i <= slicing_day_value)
    X0_use = X[use_mask]
    y0_use = y[use_mask]
    day0_use = day_info_i[use_mask]

    trial_bin_use = data.trial_bin[use_mask]
    trial_id_use, _n_trials = mt.build_trial_ids(trial_bin_use)

    device = mt.get_device()

    def objective(trial: optuna.Trial) -> float:
        # deterministic-ish per trial
        trial_seed = int(cli_args.seed) + int(trial.number)
        mt.set_seed(trial_seed)

        # --- sample hyperparams ---
        hidden = suggest_hidden_sizes(trial)
        lr = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
        weight_decay = trial.suggest_float("weight_decay", WD_MIN, WD_MAX, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
        n_lags = trial.suggest_int("n_lags", NLAGS_MIN, NLAGS_MAX)
        lag_step = trial.suggest_int("lag_step", LAGSTEP_MIN, LAGSTEP_MAX)

        # --- lag stacking (group fixed = trial) ---
        group = mt.choose_lag_group(FIX_LAG_GROUP, day_vec=day0_use, trial_id_vec=trial_id_use)
        X_use, y_use, group_eff = mt.make_lagged_features(
            X=X0_use,
            y=y0_use,
            group=group,
            n_lags=n_lags,
            lag_step=lag_step,
        )

        if X_use.shape[0] < 50:
            # too few samples after lagging => return poor score
            return 0.0

        # IMPORTANT (matches your original behavior): per-day metric later uses "day_use" after overwrite.
        # Here we keep day-vector aligned with lagged samples by using group_eff (as in your script).
        day_use_for_split = group_eff

        n_classes = int(np.unique(y_use).size)
        if n_classes < 2:
            return 0.0

        # --- split ---
        X_tr_raw, y_tr, X_va_raw, y_va, _day_va = make_train_val_split(
            X_use=X_use,
            y_use=y_use,
            day_vec_for_split=day_use_for_split,
            train_ratio=float(cli_args.train_ratio),
            seed=trial_seed,
        )

        # --- preprocessing (fixed) ---
        X_tr, X_va, _scaler = mt.preprocess_train_val(
            X_train_raw=X_tr_raw,
            X_val_raw=X_va_raw,
            scale=FIX_SCALE,
            log_scale=FIX_LOGSCALE,
        )

        # --- data loaders ---
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
        )

        # --- model ---
        model = mt.MLP(in_dim=X_tr.shape[1], hidden=hidden, out_dim=n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = nn.CrossEntropyLoss()

        # --- train with pruning ---
        max_epochs = int(cli_args.epochs)
        eval_every = max(1, int(cli_args.eval_every))

        for ep in range(1, max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            if ep % eval_every == 0 or ep == max_epochs:
                y_pred = mt.predict_labels(model, X_val=X_va, y_val=y_va, device=device, batch_size=int(batch_size))
                val_acc = float((y_pred == y_va).mean())

                trial.report(val_acc, step=ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return val_acc

    return objective


def main():
    cli = build_cli().parse_args()
    if not (0.0 < cli.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")

    sampler = optuna.samplers.TPESampler(seed=int(cli.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

    study = optuna.create_study(
        direction="maximize",
        study_name=cli.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=cli.storage,
        load_if_exists=(cli.storage is not None),
    )

    objective = objective_factory(cli)
    study.optimize(objective, n_trials=int(cli.n_trials))

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "note": "hidden_sizes = [h1] if h2==0 else [h1,h2]",
    }
    print("BEST:", json.dumps(best, indent=2))

    with open(cli.save_best_json, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved best params to: {cli.save_best_json}")


if __name__ == "__main__":
    main()
