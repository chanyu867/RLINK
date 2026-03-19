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

def discretize_position(values, bins, shift=0, trial_ID=None, low=0.0, high=1.0):
    # labels = np.digitize(values, bins)
    # intervals = make_intervals(bins)

    # meta = {
    #     "bins": bins,
    #     "intervals": intervals,
    #     "n_classes": len(intervals),
    # }
    # return labels, meta

    v = np.asarray(values)
    N = v.shape[0]

    labels = np.full(N, -1, dtype=int)  # invalid by default
    mask = np.zeros(N, dtype=int)       # 1 keep, 0 drop
    if shift == 0:
        labels[:] = np.digitize(v, bins)
        mask[:] = 1
    else:
        tid = np.asarray(trial_ID)

        valid = np.zeros(N, dtype=bool) #initial array
        valid[:-shift] = (tid[:-shift] == tid[shift:]) #get only the same trial part after shifting

        idx = np.where(valid)[0] #get True index, meaning the safe index to shift
        labels[idx] = np.digitize(v[idx + shift], bins) #actually do shifting

        mask[valid] = 1 #insert 1 at the safe index for shifting
        print(f"shift: {shift} removed {N - np.sum(mask)} samples due to trial boundary")

    intervals = make_intervals(bins)
    meta = {
        "bins": bins,
        "intervals": intervals,
        "n_classes": len(intervals),
        "shift": shift,
    }

    return labels, meta, mask



def discretize_next_direction(values, shift=1):
    v = np.asarray(values)
    
    # labels = np.empty_like(v, dtype=int)
    # labels[:-1] = (np.diff(v) >= 0).astype(int)  # 1: forward, 0: backward
    # labels[-1] = -1  # no "next" at the last sample

    # meta = {
    #     "n_classes": 2,
    #     "class_names": {0: "backward", 1: "forward"},
    # }
    # return labels, meta
    labels = np.full(v.shape, -1, dtype=int) #full fill -1, so the edge part will be -1 regardless

    delta = v[shift:] - v[:-shift]         # shifted diff
    labels[:-shift] = (delta >= 0).astype(int)


    meta = {
        "n_classes": 2,
        "class_names": {0: "backward", 1: "forward"}, #-> no stayやんけ
        "shift": shift,
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
    # print("shape: ", np.array(neural_data).shape, np.array(labels).shape)
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



def load_or_create_classes(finger_type, mode, boundaries, data_dir, classes_dir, shift, trial_ID):
    
    """
    Loads or creates digitized classes for given finger(s) and mode, with metadata.

    Args:
        finger_type: "idx", "mrs", or "together"
        mode: "position" or "velocity"
        boundaries: list or array of bin edges
        data_dir: directory with raw npy data
        classes_dir: directory to save/load class arrays and metadata

    Returns:
        labels: (N,) array of class IDs
        class_map: dict with {finger_type: {class_id: [low, high], ...}} 
                   for single finger it's still wrapped in a dict
    """
    os.makedirs(classes_dir, exist_ok=True)
    meta_dir = os.path.join(classes_dir, "metadata")
    mask_dir = os.path.join(classes_dir, "masks")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if mode == "position":
        bin_str = "_".join([str(b) for b in boundaries])
        labels_file = os.path.join(classes_dir, f"{finger_type}_{mode}_labels_{bin_str}_shift{shift}.npy")
        meta_file = os.path.join(meta_dir, f"{finger_type}_{mode}_meta_{bin_str}_shift{shift}.npy")
        mask_file = os.path.join(mask_dir, f"{finger_type}_{mode}_mask_{bin_str}_shift{shift}.npy")
    else:
        labels_file = os.path.join(classes_dir, f"{finger_type}_{mode}_{shift}_labels.npy")
        meta_file = os.path.join(meta_dir, f"{finger_type}_{mode}_{shift}_meta.npy")

    # if os.path.exists(labels_file) and os.path.exists(meta_file):
    #     labels = np.load(labels_file)
    #     class_map = np.load(meta_file, allow_pickle=True).item()
    #     return labels, class_map

    # --- create labels and metadata ---
    if finger_type in ["idx", "mrs"]:
        values = np.load(os.path.join(data_dir, f"{finger_type}_position_all.npy")).squeeze()
        
        if mode == "position":
            labels, meta, mask = discretize_position(values, boundaries, shift=shift, trial_ID=trial_ID) # convert intervals to class_map format
            class_map = {finger_type: {i: interval for i, interval in enumerate(meta["intervals"])}}
        elif mode == "direction":
            labels, meta = discretize_next_direction(values, shift=shift)
            class_map = meta["class_names"]
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented yet")

    elif finger_type == "together":
        # load both fingers
        idx_values = np.load(os.path.join(data_dir, "idx_position_all.npy")).squeeze()
        mrs_values = np.load(os.path.join(data_dir, "mrs_position_all.npy")).squeeze()

        if mode != "position":
            raise NotImplementedError(f"Mode '{mode}' not implemented yet")

        idx_labels, idx_meta = discretize_position(idx_values, boundaries)
        mrs_labels, mrs_meta = discretize_position(mrs_values, boundaries)

        labels, _ = combine_finger_labels(idx_labels, mrs_labels, boundaries)

        # class_map for each finger
        class_map = {
            "idx": {i: interval for i, interval in enumerate(idx_meta["intervals"])},
            "mrs": {i: interval for i, interval in enumerate(mrs_meta["intervals"])}
        }

    else:
        raise ValueError("finger_type must be 'idx', 'mrs', or 'together'")

    # save for future use
    np.save(labels_file, labels)
    np.save(meta_file, class_map)
    np.save(mask_file, mask)

    print("types: ", type(labels), type(class_map), type(mask))

    return labels, class_map, mask


if __name__ == "__main__":

    # --- Ultimate guide to loading the data ---

    data_dir = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all" # raw_npy folder in Google Drive
    classes_dir = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir" # classes folder in Google Drive - will create metadata folder inside

    finger_type = "idx" # "idx", "mrs", or "together"
    mode = "position" # "position" or "velocity"
    # boundaries = [0.2, 0.4, 0.6, 0.8]
    # boundaries = [0.25, 0.5, 0.75]
    boundaries = [0.5]

    #load trial ID data
    trial_ID = np.load("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_number.npy", allow_pickle=True)
    print("data: ", np.unique(trial_ID).size, trial_ID[:100])

    # 1. load array with classes
    # labels is (N,) array of classes,
    # class_map is a class description like {0: [0.0, 0.15], 1: [0.15, 0.85], 2: [0.85, 1.0]}
    for shift in [0, 1, 3, 5, 7, 9, 11, 20, 50]:
        labels, class_map, mask = load_or_create_classes(
            finger_type = finger_type,
            mode = mode,
            boundaries = boundaries,
            data_dir = data_dir,
            classes_dir = classes_dir,
            shift = shift,
            trial_ID = trial_ID,
            )
        
        # # 2. create mask and apply to all data necessary
        # mask = make_mask(labels, [0, 2]) # keep [0., 0.15] and [0.85, 1.]

        # sbp = np.load(f"{data_dir}/sbp_all.npy")
        # labeled_sbp = assemble_features(neural_data=sbp, labels=labels) # add class as the last dimension
        # del sbp # be merciful to RAM, save it from keeping useless 5 GB

        # masked_sbp = labeled_sbp[mask] # apply mask
        # # del labeled_sbp

        # # same works for the additional data (here - stimulus onset)
        # stimulus_onset = np.load(f"{data_dir}/start_time_stop_time_all.npy")
        # stimulus_onset_masked = stimulus_onset[mask]
        # # del stimulus_onset

        # print(labeled_sbp.shape, masked_sbp.shape, stimulus_onset.shape)

    # ta-da! that's it, the data is ready

    # --- Examples for manual data handling ---

    # Example 1: one finger, keeps all the classes
    # X, y, class_map = prepare_dataset_one_finger(
    #     data_dir = data_dir,
    #     finger = 'idx',
    #     bins = [0.15, 0.85]
    # )
    # class_map output:
    # {0: [0.0, 0.15], 1: [0.15, 0.85], 2: [0.85, 1.0]}


    # Example 2: one finger, filter classes from the previous example (leaves only flex and deflex)
    # X, y, class_map = prepare_dataset_one_finger(
    #     data_dir = data_dir,
    #     finger = 'idx',
    #     bins = [0.15, 0.85],
    #     allowed_classes = [0,2]
    # )


    # Example 3: apply the boundaries to additonal data (here - stimulus onset)
    # suppose we have y from the example 1
    # stimulus_onset = np.load(f"{data_dir}/start_time_stop_time_all.npy")

    # mask = make_mask(y, [0,2])

    # X_masked = X[mask]
    # y_masked = y[mask]
    # stimulus_onset_masked = stimulus_onset[mask]

    # print(X.shape, y.shape, stimulus_onset.shape)
    # print(X_masked.shape, y_masked.shape, stimulus_onset_masked.shape)
    # output:
    # (7711816, 97) (7711816,) (7711816, 2)
    # (1095899, 97) (1095899,) (1095899, 2)