import numpy as np

from MLP_finger_decoder_train_test import apply_masks

finger_ID = "idx"
mode = "position"
boundary = "0.33_0.66"   # choose one: "0.33_0.66" / "0.25_0.5_0.75" / "0.2_0.4_0.6_0.8"
task = "random"         # "random" or "center-out"
slicing_day_idx = 10
train_ratio = 0.8
sbp_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
target_style_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"
label_path = f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/{finger_ID}_{mode}_labels_{boundary}_shift0.npy"
target_type = task
allowed = None
seed = 0

# 1) load (same as training)
sbp = np.load(sbp_path, allow_pickle=False)
labels = np.load(label_path, allow_pickle=False).astype(int)
day_info = np.load(day_info_path, allow_pickle=False).astype(int)

target_style = None
if target_style_path is not None:
    target_style = np.load(target_style_path, allow_pickle=False).astype(bool)

# 2) same masks
sbp, labels, day_info, target_style = apply_masks(
    sbp=sbp, labels=labels, day_info=day_info,
    target_style=target_style, target_type=target_type,
    label_mask_allowed=allowed,
)

# 3) slicing -> y_use
uniq_all_days = np.unique(day_info)
slicing_day_value = uniq_all_days[slicing_day_idx - 1]
use_mask = day_info <= slicing_day_value
y_use = labels[use_mask]

# 4) reconstruct split (same as training)
N = y_use.shape[0]
n_train = int(round(N * train_ratio))
n_train = max(1, min(n_train, N - 1))

np.random.seed(seed)
perm = np.random.permutation(N)
tr_idx = perm[:n_train]
va_idx = perm[n_train:]

y_tr = y_use[tr_idx]
y_va = y_use[va_idx]

# 5) counts
def counts(y):
    cls, cnt = np.unique(y, return_counts=True)
    return dict(zip(cls.tolist(), cnt.tolist()))

print("USED (before split):", counts(y_use))
print("TRAIN split:", counts(y_tr))
print("VAL split:", counts(y_va))
