#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.trial_splits import split_group_ids, group_kfold_ids
import mlp_train as mt  # your existing utilities (load, masks, preprocess, MLP, predict_labels, etc.)


# -------------------------
# Search ranges (reasonable defaults)
# -------------------------
H1_CHOICES = [64, 128, 256, 512, 1024]
H2_CHOICES = [0, 64, 128, 256, 512]  # 0 => no 2nd layer (i.e., 1 hidden layer)
H3_CHOICES = [0, 64, 128, 256, 512]  # 0 => no 2nd layer (i.e., 1 hidden layer)
H4_CHOICES = [0, 64, 128, 256, 512]  # 0 => no 2nd layer (i.e., 1 hidden layer)
BATCH_CHOICES = [64, 256]

LR_MIN, LR_MAX = 1e-5, 3e-3
WD_MIN, WD_MAX = 1e-8, 1e-2
NLAGS_MIN, NLAGS_MAX = 0, 16
LAGSTEP_MIN, LAGSTEP_MAX = 1, 4


# -------------------------
# Fixed settings (to keep behavior aligned with your current script)
# -------------------------
FIX_LAG_GROUP = "trial"
FIX_SCALE = True
FIX_LOGSCALE = False
FIX_TARGET_TYPE = "random"
FIX_ALLOWED_LABELS = None


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Required data inputs (same meaning as in mlp_train.py)
    ap.add_argument("--sbp_path", type=str, required=True)
    ap.add_argument("--trial_bin_path", type=str, required=True)
    ap.add_argument("--label_path", type=str, required=True)
    ap.add_argument("--day_info_path", type=str, required=True)
    ap.add_argument(
        "--target_style_path",
        type=str,
        default=None,
        help="target_style.npy path required when using --target_type (random/center-out)",
    )

    # slicing (same meaning as mlp_train.py: 1-indexed among unique days)
    ap.add_argument("--slicing_day", type=int, required=True)

    # split / CV
    ap.add_argument("--train_ratio", type=float, default=0.8, help="train:test split ratio (trial-group split)")
    ap.add_argument("--cv_folds", type=int, default=3, help="Group K-fold CV on TRAIN trials only")

    # training budget
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # Optuna
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--study_name", type=str, default="mlp_hpo")
    ap.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///mlp_hpo.db (optional)")
    
    ap.add_argument("--metric", type=str, default="bacc", choices=["acc", "bacc", "f1_macro"], help="Optimize for balanced metrics")

    # outputs
    ap.add_argument("--save_best_json", type=str, default="best_params.json")
    ap.add_argument(
        "--save_split_npz",
        type=str,
        default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/hpo/mlp_cv/split_indices.npz",
        help="Saves train/test trial split and corresponding global indices (so you can extract non-Optuna data later).",
    )
    ap.add_argument(
        "--save_final_eval_json",
        type=str,
        default=f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/hpo/mlp_cv/final_eval_{xxx}.json",
        help="Saves final train-on-train-trials + eval-on-test-trials results.",
    )
    return ap

def suggest_hidden_sizes(trial: optuna.Trial) -> List[int]:
    # 1. Suggest all raw choices statically to prevent the ValueError
    h1 = trial.suggest_categorical("h1", H1_CHOICES)
    h2_raw = trial.suggest_categorical("h2", H2_CHOICES)
    h3_raw = trial.suggest_categorical("h3", H3_CHOICES)
    h4_raw = trial.suggest_categorical("h4", H4_CHOICES)
    
    # 2. Waterfall capping (Funnel Rule: h1 >= h2 >= h3 >= h4)
    # Cap h2 at h1
    h2 = h2_raw if (h2_raw <= h1 or h2_raw == 0) else h1
    
    # Cap h3 at h2 (If h2 is 0, h3 MUST be 0)
    if h2 == 0:
        h3 = 0
    else:
        h3 = h3_raw if (h3_raw <= h2 or h3_raw == 0) else h2
        
    # Cap h4 at h3 (If h3 is 0, h4 MUST be 0)
    if h3 == 0:
        h4 = 0
    else:
        h4 = h4_raw if (h4_raw <= h3 or h4_raw == 0) else h3

    # Build the final valid layer list
    layers = [h1]
    if h2 != 0: layers.append(h2)
    if h3 != 0: layers.append(h3)
    if h4 != 0: layers.append(h4)
    
    return layers


def make_lagged_by_trials(
    X: np.ndarray,
    y: np.ndarray,
    day: np.ndarray,
    trial_id: np.ndarray,
    use_trial_ids: np.ndarray,
    n_lags: int,
    lag_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build lagged features using ONLY samples within given trials (prevents leakage across train/val/test).

    Returns:
      X_lag: (M, D*(n_lags+1))
      y_lag: (M,)
      day_lag: (M,)        day at current time t
      idx_global: (M,)     global index (w.r.t X, y, day arrays after slicing mask)
    """
    use_set = set(int(t) for t in np.asarray(use_trial_ids).tolist())
    D = X.shape[1]
    if n_lags < 0 or lag_step < 1:
        raise ValueError("n_lags must be >=0 and lag_step must be >=1")

    X_out = []
    y_out = []
    day_out = []
    idx_out = []

    # Process trial by trial to keep temporal structure inside a trial
    for tid in np.unique(trial_id):
        if int(tid) not in use_set:
            continue

        idx = np.where(trial_id == tid)[0]
        if idx.size == 0:
            continue

        # idx is already in time order if original arrays are in time order; keep safe:
        idx = np.sort(idx)

        # need at least (n_lags*lag_step + 1) points
        start = n_lags * lag_step
        if idx.size <= start:
            continue

        for p in range(start, idx.size):
            # global indices for window: [t - n_lags*lag_step, ..., t]
            window = [idx[p - k * lag_step] for k in range(n_lags, -1, -1)]  # oldest -> current
            feat = X[window].reshape(-1)  # (D*(n_lags+1),)

            t_idx = idx[p]
            X_out.append(feat)
            y_out.append(int(y[t_idx]))
            day_out.append(int(day[t_idx]))
            idx_out.append(int(t_idx))

    if len(X_out) == 0:
        return (
            np.zeros((0, D * (n_lags + 1)), dtype=np.float32),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    return (
        np.asarray(X_out, dtype=np.float32),
        np.asarray(y_out, dtype=int),
        np.asarray(day_out, dtype=int),
        np.asarray(idx_out, dtype=int),
    )


@dataclass(frozen=True)
class PreparedData:
    X0: np.ndarray
    y0: np.ndarray
    day0: np.ndarray
    trial_id0: np.ndarray
    # mapping to original indices BEFORE slicing mask (so you can recover full-array indices)
    orig_idx0: np.ndarray
    slicing_day_value: int


def prepare_data(cli_args) -> PreparedData:
    # Create args-like object compatible with mt.load_all_arrays
    class A:
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
    orig_idx0 = np.where(use_mask)[0].astype(int)

    X0 = X[use_mask]
    y0 = y[use_mask]
    day0 = day_info_i[use_mask]

    trial_bin_use = data.trial_bin[use_mask]
    trial_id0, _n_trials = mt.build_trial_ids(trial_bin_use)

    return PreparedData(
        X0=X0,
        y0=y0,
        day0=day0,
        trial_id0=np.asarray(trial_id0).astype(int),
        orig_idx0=orig_idx0,
        slicing_day_value=int(slicing_day_value),
    )


def train_one_fold(
    *,
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_va_raw: np.ndarray,
    y_va: np.ndarray,
    hidden: List[int],
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
    metric: str, # NEW: Accept the metric string
) -> float:
    if X_tr_raw.shape[0] < 10 or X_va_raw.shape[0] < 10:
        return 0.0

    # preprocess (fit scaler on train only)
    X_tr, X_va, _scaler = mt.preprocess_train_val(
        X_train_raw=X_tr_raw,
        X_val_raw=X_va_raw,
        scale=FIX_SCALE,
        log_scale=FIX_LOGSCALE,
    )

    n_classes = int(np.unique(y_tr).size)
    if n_classes < 2:
        return 0.0

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    model = mt.MLP(in_dim=X_tr.shape[1], hidden=hidden, out_dim=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    for _ep in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    y_pred = mt.predict_labels(model, X_val=X_va, y_val=y_va, device=device, batch_size=int(batch_size))

    if metric == "bacc":
        val_score = float(balanced_accuracy_score(y_va, y_pred))
    elif metric == "f1_macro":
        val_score = float(f1_score(y_va, y_pred, average="macro"))
    else:
        val_score = float((y_pred == y_va).mean())
        
    return val_score

def objective_factory(cli_args, pdata: PreparedData, train_trials: np.ndarray) -> optuna.Trial:
    device = mt.get_device()

    # CV folds: list of val trial-id arrays
    # val_folds = _group_kfold_ids(train_trials, k=int(cli_args.cv_folds), seed=int(cli_args.seed))
    val_folds = group_kfold_ids(train_trials, k=int(cli_args.cv_folds), seed=int(cli_args.seed))

    def objective(trial: optuna.Trial) -> float:
        trial_seed = int(cli_args.seed) + int(trial.number)
        mt.set_seed(trial_seed)

        hidden = suggest_hidden_sizes(trial)
        lr = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
        weight_decay = trial.suggest_float("weight_decay", WD_MIN, WD_MAX, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
        n_lags = trial.suggest_int("n_lags", NLAGS_MIN, NLAGS_MAX)
        lag_step = trial.suggest_int("lag_step", LAGSTEP_MIN, LAGSTEP_MAX)

        fold_scores: List[float] = []

        for fold_i, val_trials in enumerate(val_folds, start=1):
            tr_trials = np.setdiff1d(train_trials, val_trials, assume_unique=False)

            # make train data
            X_tr_raw, y_tr, _day_tr, _idx_tr = make_lagged_by_trials(
                X=pdata.X0,
                y=pdata.y0,
                day=pdata.day0,
                trial_id=pdata.trial_id0,
                use_trial_ids=tr_trials,
                n_lags=n_lags,
                lag_step=lag_step,
            )
            # make validation data, which is the trial block not selected in this fold run
            X_va_raw, y_va, _day_va, _idx_va = make_lagged_by_trials(
                X=pdata.X0,
                y=pdata.y0,
                day=pdata.day0,
                trial_id=pdata.trial_id0,
                use_trial_ids=val_trials,
                n_lags=n_lags,
                lag_step=lag_step,
            )

            # execute training for this fold and get accuracy
            score = train_one_fold(
                X_tr_raw=X_tr_raw,
                y_tr=y_tr,
                X_va_raw=X_va_raw,
                y_va=y_va,
                hidden=hidden,
                lr=float(lr),
                weight_decay=float(weight_decay),
                batch_size=int(batch_size),
                epochs=int(cli_args.epochs),
                device=device,
                metric=cli_args.metric, # NEW: Pass metric to fold
            )
            fold_scores.append(float(score))

            # prune based on mean score so far
            mean_so_far = float(np.mean(fold_scores)) if len(fold_scores) > 0 else 0.0
            trial.report(mean_so_far, step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def final_train_and_test(
    *,
    cli_args,
    pdata: PreparedData,
    train_trials: np.ndarray,
    test_trials: np.ndarray,
    best_params: Dict,
) -> Dict:
    device = mt.get_device()

    # unpack best
    # unpack best WITH the waterfall capping rule
    h1 = int(best_params["h1"])
    h2_raw = int(best_params.get("h2", 0))
    h3_raw = int(best_params.get("h3", 0))
    h4_raw = int(best_params.get("h4", 0))

    h2 = h2_raw if (h2_raw <= h1 or h2_raw == 0) else h1
    h3 = 0 if h2 == 0 else (h3_raw if (h3_raw <= h2 or h3_raw == 0) else h2)
    h4 = 0 if h3 == 0 else (h4_raw if (h4_raw <= h3 or h4_raw == 0) else h3)

    hidden = [h1]
    if h2 != 0: hidden.append(h2)
    if h3 != 0: hidden.append(h3)
    if h4 != 0: hidden.append(h4)

    lr = float(best_params["lr"])
    weight_decay = float(best_params["weight_decay"])
    batch_size = int(best_params["batch_size"])
    n_lags = int(best_params["n_lags"])
    lag_step = int(best_params["lag_step"])

    # build lagged train/test from fixed trial split
    X_tr_raw, y_tr, _day_tr, idx_tr = make_lagged_by_trials(
        X=pdata.X0,
        y=pdata.y0,
        day=pdata.day0,
        trial_id=pdata.trial_id0,
        use_trial_ids=train_trials,
        n_lags=n_lags,
        lag_step=lag_step,
    )
    X_te_raw, y_te, _day_te, idx_te = make_lagged_by_trials(
        X=pdata.X0,
        y=pdata.y0,
        day=pdata.day0,
        trial_id=pdata.trial_id0,
        use_trial_ids=test_trials,
        n_lags=n_lags,
        lag_step=lag_step,
    )

    # preprocess train->test
    X_tr, X_te, _scaler = mt.preprocess_train_val(
        X_train_raw=X_tr_raw,
        X_val_raw=X_te_raw,
        scale=FIX_SCALE,
        log_scale=FIX_LOGSCALE,
    )

    n_classes = int(np.unique(y_tr).size)
    model = mt.MLP(in_dim=X_tr.shape[1], hidden=hidden, out_dim=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    for _ep in range(int(cli_args.epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    y_pred = mt.predict_labels(model, X_val=X_te, y_val=y_te, device=device, batch_size=int(batch_size))
    test_acc = float((y_pred == y_te).mean())
    test_bacc = float(balanced_accuracy_score(y_te, y_pred))
    test_f1 = float(f1_score(y_te, y_pred, average="macro"))

    # convert split indices to ORIGINAL (pre-slicing) indices for easy extraction later
    orig_idx_tr = pdata.orig_idx0[idx_tr].astype(int) if idx_tr.size > 0 else np.zeros((0,), dtype=int)
    orig_idx_te = pdata.orig_idx0[idx_te].astype(int) if idx_te.size > 0 else np.zeros((0,), dtype=int)

    return {
        "metric": "acc",
        "test_acc": test_acc,
        "test_bacc": test_bacc,     # NEW
        "test_f1_macro": test_f1,   # NEW
        "best_params": dict(best_params),
        "n_train_samples_lagged": int(X_tr.shape[0]),
        "n_test_samples_lagged": int(X_te.shape[0]),
        "train_trials": train_trials.astype(int).tolist(),
        "test_trials": test_trials.astype(int).tolist(),
        "train_indices_orig": orig_idx_tr.tolist(),
        "test_indices_orig": orig_idx_te.tolist(),
    }


def main():
    cli = build_cli().parse_args()
    if not (0.0 < cli.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")
    if int(cli.cv_folds) < 2:
        raise ValueError("--cv_folds must be >= 2")

    # ---- load once ----
    pdata = prepare_data(cli) #slicing_day = 1, meaning only first day is used for HPO by Optuna

    # ---- fixed train/test split at TRIAL level ---- #-> take an account about the trial group, so lag does not go over different trials
    # train_trials, test_trials = _split_unique_ids(pdata.trial_id0, train_ratio=float(cli.train_ratio), seed=int(cli.seed))
    train_trials, test_trials = split_group_ids(
        pdata.trial_id0,
        train_ratio=float(cli.train_ratio),
        seed=int(cli.seed),
        stratify=False,
    )
    # print("unique train trials:", np.unique(train_trials).shape, np.unique(test_trials).shape) -> (300,) (75,)
    # print("label info?: ", np.unique(pdata.trial_id0).shape, test_trials.shape) -> (375,) (75,)

    # Save split info so you can extract "Optuna-unseen" test data later
    train_mask0 = np.isin(pdata.trial_id0, train_trials)
    test_mask0 = np.isin(pdata.trial_id0, test_trials)

    split_payload = dict(
        slicing_day=int(cli.slicing_day),
        slicing_day_value=int(pdata.slicing_day_value),
        seed=int(cli.seed),
        train_ratio=float(cli.train_ratio),
        cv_folds=int(cli.cv_folds),
        # trial ids (group-level split)
        train_trials=train_trials.astype(int),
        test_trials=test_trials.astype(int),
        # indices w.r.t arrays AFTER slicing mask
        train_indices_in_sliced=np.where(train_mask0)[0].astype(int),
        test_indices_in_sliced=np.where(test_mask0)[0].astype(int),
        # indices w.r.t ORIGINAL arrays (before slicing mask) -> most useful for later extraction
        train_indices_orig=pdata.orig_idx0[train_mask0].astype(int),
        test_indices_orig=pdata.orig_idx0[test_mask0].astype(int),
    )
    np.savez(cli.save_split_npz, **split_payload)
    print(f"[OK] Saved split file: {cli.save_split_npz}")

    # ---- Optuna on TRAIN only, with group K-fold CV ----
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

    objective = objective_factory(cli, pdata, train_trials=train_trials)
    study.optimize(objective, n_trials=int(cli.n_trials))

    # Re-apply the funnel rule to the saved parameters so mlp_train.py reads it correctly
    best_params_dict = dict(study.best_params)
    
    h1 = best_params_dict["h1"]
    h2_raw = best_params_dict.get("h2", 0)
    h3_raw = best_params_dict.get("h3", 0)
    h4_raw = best_params_dict.get("h4", 0)
    
    h2 = h2_raw if (h2_raw <= h1 or h2_raw == 0) else h1
    h3 = 0 if h2 == 0 else (h3_raw if (h3_raw <= h2 or h3_raw == 0) else h2)
    h4 = 0 if h3 == 0 else (h4_raw if (h4_raw <= h3 or h4_raw == 0) else h3)
    
    best_params_dict["h1"] = h1
    best_params_dict["h2"] = h2
    best_params_dict["h3"] = h3
    best_params_dict["h4"] = h4

    best = {
        "best_cv_value": float(study.best_value),
        "best_params": best_params_dict,
        "note": "Enforced h1 >= h2 >= h3 >= h4 funnel rule.",
        "split_npz": cli.save_split_npz,
    }
    print("BEST (CV):", json.dumps(best, indent=2))

    with open(cli.save_best_json, "w") as f:
        json.dump(best, f, indent=2)
    # print(f"[OK] Saved best params to: {cli.save_best_json}")

    # ---- Final independent TEST evaluation (train on all train trials, eval on held-out test trials) ----
    final_eval = final_train_and_test(
        cli_args=cli,
        pdata=pdata,
        train_trials=train_trials,
        test_trials=test_trials,
        best_params=study.best_params,
    )
    # print("FINAL (TEST):", json.dumps(final_eval, indent=2))

    with open(cli.save_final_eval_json, "w") as f:
        json.dump(final_eval, f, indent=2)
    print(f"[OK] Saved final eval to: {cli.save_final_eval_json}")


if __name__ == "__main__":
    main()