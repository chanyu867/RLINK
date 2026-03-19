# trial_splits.py
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np


def split_group_ids(
    group_ids: np.ndarray,
    *,
    train_ratio: float,
    seed: int,
    y: Optional[np.ndarray] = None,
    stratify: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trial-aware split: split unique group IDs (e.g., trial_id) into train/test.

    - stratify=False (default): random split of unique ids (mlp_hpo.py behavior)
    - stratify=True and y provided: try stratified split using per-group majority label
      (mlp_train_day.py behavior), else fallback to random.

    Returns:
      train_ids, test_ids  (both are arrays of unique group IDs)
    """
    rng = np.random.RandomState(int(seed))
    uniq = np.unique(group_ids).astype(int)

    if uniq.size < 2:
        return uniq, np.array([], dtype=int)

    # ---------- random unique-id split (HPO behavior) ----------
    def _random_split() -> Tuple[np.ndarray, np.ndarray]:
        perm = rng.permutation(len(uniq))
        n_train = int(round(len(uniq) * float(train_ratio)))
        n_train = max(1, min(n_train, len(uniq) - 1))
        train_ids = uniq[perm[:n_train]]
        test_ids = uniq[perm[n_train:]]
        return train_ids.astype(int), test_ids.astype(int)

    if (not stratify) or (y is None):
        return _random_split()

    # ---------- stratified by per-trial majority label (per-day behavior) ----------
    # Compute per-group majority label
    maj = []
    for gid in uniq:
        idx = (group_ids == gid)
        y_g = y[idx]
        if y_g.size == 0:
            maj.append(-1)
            continue
        cls, cnt = np.unique(y_g, return_counts=True)
        maj.append(int(cls[np.argmax(cnt)]))
    maj = np.asarray(maj, dtype=int)

    # test size (same idea as per-day script)
    n_test = int(np.round(uniq.size * (1.0 - float(train_ratio))))
    n_test = max(1, min(n_test, uniq.size - 1))

    # check feasibility: each class must have >=2 trials
    ok = True
    for c in np.unique(maj[maj >= 0]):
        if np.sum(maj == c) < 2:
            ok = False
            break
    if not ok:
        return _random_split()

    # Stratified sample test trials within each stratum
    test_ids_list = []
    for c in np.unique(maj):
        idx = np.where(maj == c)[0]
        if idx.size == 0:
            continue
        k = int(np.round(n_test * (idx.size / uniq.size)))
        k = max(0, min(k, idx.size))
        if k > 0:
            pick = rng.choice(idx, size=k, replace=False)
            test_ids_list.extend(uniq[pick].tolist())

    test_ids = np.asarray(sorted(set(test_ids_list)), dtype=int)

    # Adjust rounding drift to match n_test
    if test_ids.size < n_test:
        remain = np.setdiff1d(uniq, test_ids, assume_unique=False)
        add = rng.choice(remain, size=(n_test - test_ids.size), replace=False)
        test_ids = np.asarray(sorted(set(test_ids.tolist() + add.tolist())), dtype=int)
    elif test_ids.size > n_test:
        drop = rng.choice(np.arange(test_ids.size), size=(test_ids.size - n_test), replace=False)
        keep = np.ones(test_ids.size, dtype=bool)
        keep[drop] = False
        test_ids = np.asarray(sorted(test_ids[keep].tolist()), dtype=int)

    train_ids = np.setdiff1d(uniq, test_ids, assume_unique=False).astype(int)
    return train_ids, test_ids


def group_kfold_ids(train_ids: np.ndarray, *, k: int, seed: int) -> List[np.ndarray]:
    """
    Group K-fold on unique group IDs (trial-aware).
    Returns list of val-id arrays.
    (Same behavior as mlp_hpo.py _group_kfold_ids)
    """
    uniq = np.array(train_ids, copy=True)
    rng = np.random.RandomState(int(seed))
    uniq = uniq[rng.permutation(len(uniq))]

    k = max(2, min(int(k), len(uniq)))
    folds = np.array_split(uniq, k)
    return [f.astype(int) for f in folds if len(f) > 0]
