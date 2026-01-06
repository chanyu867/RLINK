import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def npy_loader(path):

    arr = np.load(path)  # for sbp/mrs this should be numeric; no pickle needed

    print(f"File:  {path}")
    print(f"dtype: {arr.dtype}")
    print(f"shape: {arr.shape}")
    print(f"ndim:  {arr.ndim}")
    print(f"size:  {arr.size:,}")
    print(f"nbytes:{arr.nbytes:,} ({arr.nbytes/1024/1024:.2f} MiB)")

    # preview only (don’t print everything)
    # print(arr[:1])

    return arr

def threshold_binary(x, thresh=0.85):
    x = np.asarray(x)
    return (x < thresh).astype(int)

def plot_label_distribution_blocks(labels, block_size=1000):
    """
    labels: 1D array-like of 0/1
    block_size: number of samples per block
    """
    labels = np.asarray(labels).astype(int).ravel()
    n = len(labels)

    n_blocks = n // block_size  # ignore leftover for clean blocks
    if n_blocks == 0:
        raise ValueError("Not enough data for one block.")

    labels = labels[: n_blocks * block_size]
    blocks = labels.reshape(n_blocks, block_size)

    # fraction of ones per block
    frac1 = blocks.mean(axis=1)            # in [0,1]
    frac0 = 1.0 - frac1

    x = np.arange(1, n_blocks + 1)

    plt.figure(figsize=(12, 4))
    plt.bar(x, frac0 * 100.0, label="0 (%)")                 # blue default
    plt.bar(x, frac1 * 100.0, bottom=frac0 * 100.0, label="1 (%)")  # red default

    plt.ylim(0, 100)
    plt.xlabel(f"Block index (block_size={block_size})")
    plt.ylabel("Percent (%)")
    plt.title("Label distribution per block (stacked %)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#How to run
# ------- Combined data ------- #
# data for finger IDX
# data = npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/sbp_with_idx_position_min0.15_max0.85.npy")
# labels = threshold_binary(data[:, -1], thresh=0.85)
# print("data: ", labels.shape, labels[230000:230500]) #data:  (282564,)
# plot_label_distribution_blocks(labels, block_size=1000)
# # data for finger MRS
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/sbp_with_mrs_position_min0.15_max0.85.npy")

# # ------- Each feature's data ------- #
# # data without position data
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/sbp_all_position_min0.15_max0.85.npy")
# # data only position data for finger IDX
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/idx_all_position_min0.15_max0.85.npy")
# # data only position data for finger MRS
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/mrs_all_position_min0.15_max0.85.npy")