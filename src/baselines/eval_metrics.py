# eval_metrics.py
from __future__ import annotations
from typing import Tuple
import numpy as np


def _confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (tp, fp, fn, support) for each class in class_ids.
    support = (#true samples of that class) = tp + fn
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    class_ids = np.asarray(class_ids).astype(int)

    tp = np.zeros(len(class_ids), dtype=float)
    fp = np.zeros(len(class_ids), dtype=float)
    fn = np.zeros(len(class_ids), dtype=float)
    support = np.zeros(len(class_ids), dtype=float)

    for k, c in enumerate(class_ids):
        yt = (y_true == c)
        yp = (y_pred == c)
        tp[k] = float(np.sum(yt & yp))
        fp[k] = float(np.sum((~yt) & yp))
        fn[k] = float(np.sum(yt & (~yp)))
        support[k] = float(np.sum(yt))

    return tp, fp, fn, support


def balanced_accuracy_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: np.ndarray,
) -> float:
    """
    Multiclass balanced accuracy = mean recall across classes.
    Classes with support==0 are ignored (nanmean).
    """
    tp, _fp, fn, support = _confusion_counts(y_true, y_pred, class_ids)
    denom = tp + fn
    recall = np.full_like(denom, np.nan, dtype=float)
    m = denom > 0
    recall[m] = tp[m] / denom[m]
    return float(np.nanmean(recall))


def f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: np.ndarray,
) -> float:
    """
    Macro-F1 across classes. Classes with support==0 are ignored (nanmean).
    """
    tp, fp, fn, support = _confusion_counts(y_true, y_pred, class_ids)

    precision = np.full_like(tp, np.nan, dtype=float)
    recall = np.full_like(tp, np.nan, dtype=float)
    f1 = np.full_like(tp, np.nan, dtype=float)

    # precision defined when (tp+fp)>0
    mp = (tp + fp) > 0
    precision[mp] = tp[mp] / (tp[mp] + fp[mp])

    # recall defined when (tp+fn)>0 (same as support>0)
    mr = (tp + fn) > 0
    recall[mr] = tp[mr] / (tp[mr] + fn[mr])

    # f1 defined when precision+recall>0
    mf = (precision + recall) > 0
    f1[mf] = 2.0 * precision[mf] * recall[mf] / (precision[mf] + recall[mf])

    # ignore classes not appearing in y_true (support==0)
    f1[support == 0] = np.nan
    return float(np.nanmean(f1))


def per_day_balanced_accuracy(
    day_arr: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq_days = np.unique(day_arr.astype(int))
    out = np.full(len(uniq_days), np.nan, dtype=float)
    n = np.zeros(len(uniq_days), dtype=int)
    for i, d in enumerate(uniq_days):
        m = (day_arr == d)
        n[i] = int(np.sum(m))
        if n[i] > 0:
            out[i] = balanced_accuracy_macro(y_true[m], y_pred[m], class_ids)
    return uniq_days.astype(int), out, n


def per_day_f1_macro(
    day_arr: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq_days = np.unique(day_arr.astype(int))
    out = np.full(len(uniq_days), np.nan, dtype=float)
    n = np.zeros(len(uniq_days), dtype=int)
    for i, d in enumerate(uniq_days):
        m = (day_arr == d)
        n[i] = int(np.sum(m))
        if n[i] > 0:
            out[i] = f1_macro(y_true[m], y_pred[m], class_ids)
    return uniq_days.astype(int), out, n
