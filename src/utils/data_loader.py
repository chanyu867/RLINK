import os
import re
import numpy as np
import pandas as pd

def load_full_data(data_dir):
    """
    Returns:
        sbp: (N, 96)
        idx_pos: (N,)
        mrs_pos: (N,)
    """
    sbp = np.load(os.path.join(data_dir, "sbp_all.npy"))
    idx_pos = np.load(os.path.join(data_dir, "idx_position_all.npy")).squeeze()
    mrs_pos = np.load(os.path.join(data_dir, "mrs_position_all.npy")).squeeze()
    # possible to add more

    assert sbp.shape[0] == idx_pos.shape[0] == mrs_pos.shape[0]

    return sbp, idx_pos, mrs_pos


def make_intervals(bins, max_val=1.0):
    edges = [0.0] + list(bins) + [max_val]
    return [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]


def discretize_position(values, bins, low=0.0, high=1.0):
    labels = np.digitize(values, bins)
    intervals = make_intervals(bins)

    meta = {
        "bins": bins,
        "intervals": intervals,
        "n_classes": len(intervals),
    }
    return labels, meta


def combine_finger_labels(idx_labels, mrs_labels, bins):
    intervals = make_intervals(bins)
    n_classes = len(intervals)

    combined = idx_labels * n_classes + mrs_labels

    class_map = {}
    class_id = 0

    for i in range(n_classes):
        for j in range(n_classes):
            class_map[class_id] = {
                "idx": intervals[i],
                "mrs": intervals[j],
            }
            class_id += 1

    return combined, class_map


def apply_class_mask(data, labels, allowed_classes):
    """
    Keeps the labels in allowed_classes.

    data: (N, D)
    labels: (N,)
    allowed_classes: iterable of class ids

    Returns:
        data_masked, labels_masked
    """
    mask = np.isin(labels, allowed_classes)
    return data[mask], labels[mask]


def assemble_features(neural_data, labels):
    """
    Adds labels as the last column to neural_data.

    neural_data: (N, 96)
    labels: (N,)

    Returns:
        X: (N, 97)
    """
    return np.concatenate([neural_data, labels[:, None]], axis=1)


def prepare_dataset_both_fingers(
    data_dir,
    bins,
    allowed_classes=None
):
  """
  Final full-cycle data processing function.
  Returns:
    X = sbp + label
    y = combined labels
    class_map = dict[int, dict] (description of y)
  """
  sbp, idx_pos, mrs_pos = load_full_data(data_dir)

  idx_labels, idx_meta = discretize_position(idx_pos, bins)
  del idx_pos

  mrs_labels, mrs_meta = discretize_position(mrs_pos, bins)
  del mrs_pos

  y, class_map = combine_finger_labels(
      idx_labels, mrs_labels, bins
  )
  del idx_labels, mrs_labels, idx_meta, mrs_meta

  if allowed_classes is not None:
      sbp, y = apply_class_mask(sbp, y, allowed_classes)

  X = assemble_features(sbp, y)
  del sbp

  return X, y, class_map


def prepare_dataset_one_finger(
    data_dir,
    finger,              # "idx" or "mrs"
    bins,
    allowed_classes=None
):
    """
    Final full-cycle data processing function (single finger).

    Returns:
        X: (N, 97)  -> sbp + label
        y: (N,)     -> finger labels
        class_map: dict[int, list] -> {class_id: [low, high]}
    """

    if finger == "idx":
        finger_pos = np.load(os.path.join(data_dir, "idx_position_all.npy")).squeeze()
    elif finger == "mrs":
        finger_pos = np.load(os.path.join(data_dir, "mrs_position_all.npy")).squeeze()
    else:
        raise ValueError("finger must be 'idx' or 'mrs'")

    y, meta = discretize_position(
        finger_pos,
        bins
    )
    del finger_pos

    class_map = {
        i: interval for i, interval in enumerate(meta["intervals"])
    }

    if allowed_classes is not None:
        mask = np.isin(y, allowed_classes)
        y = y[mask]
    else:
        mask = None

    sbp = np.load(os.path.join(data_dir, "sbp_all.npy"))

    if mask is not None:
        sbp = sbp[mask]

    X = assemble_features(sbp, y)
    del sbp

    return X, y, class_map


def make_mask(y, allowed_classes):
    """
    y: (N,) array of class labels
    allowed_classes: list or array of class IDs to keep

    Returns:
        mask: boolean array of shape (N,)
    """
    mask = np.isin(y, allowed_classes)
    return mask

if __name__ == "__main__":

    # --- Examples ---

    data_dir = "path_to_the_dir_w_full_npy_data"

    # Example 1: one finger, keeps all the classes
    X, y, class_map = prepare_dataset_one_finger(
        data_dir = data_dir,
        finger = 'idx',
        bins = [0.15, 0.85]
    )
    # class_map output:
    # {0: [0.0, 0.15], 1: [0.15, 0.85], 2: [0.85, 1.0]}


    # Example 2: one finger, filter classes from the previous example (leaves only flex and deflex)
    X, y, class_map = prepare_dataset_one_finger(
        data_dir = data_dir,
        finger = 'idx',
        bins = [0.15, 0.85],
        allowed_classes = [0,2]
    )


    # Example 3: apply the boundaries to additonal data (here - stimulus onset)
    # suppose we have y from the example 1
    stimulus_onset = np.load(f"{data_dir}/start_time_stop_time_all.npy")

    mask = make_mask(y, [0,2])

    X_masked = X[mask]
    y_masked = y[mask]
    stimulus_onset_masked = stimulus_onset[mask]

    print(X.shape, y.shape, stimulus_onset.shape)
    print(X_masked.shape, y_masked.shape, stimulus_onset_masked.shape)

    # output:
    # (7711816, 97) (7711816,) (7711816, 2)
    # (1095899, 97) (1095899,) (1095899, 2)

    