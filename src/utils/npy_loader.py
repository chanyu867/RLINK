import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def npy_loader(path):

    arr = np.load(path)  # for sbp/mrs this should be numeric; no pickle needed

    #debug
    # print(f"dtype: {arr.dtype}")
    # print(f"shape: {arr.shape}")
    # print(f"ndim:  {arr.ndim}")
    # print(f"size:  {arr.size:,}")

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

def add_block_id(data):
    data = np.asarray(data)
    block_ids = np.zeros(len(data), dtype=int)

    block = 0
    for i in range(1, len(data)):
        if data[i, 0] != data[i-1, 0] + 1:
            block += 1
        block_ids[i] = block

    return np.column_stack([data, block_ids])

#How to run
# ------- Combined data ------- #
# data for finger IDX
# data = npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/idx/sbp_position_min0.15_max0.85.npy")
# print("data: ", data.shape)

# data = npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/timestamps_all_position_min0.15_max0.85.npy")
# data = add_block_id(data)
# print("data: ", data[0:200]) #[timestamp_index, day_order, exact_date_int] - [     963        1 20200127]
#task memo:
# 1. make a function to get each trial by checking if time stamp is continous or not
# 2. as slicing time point is just when the finger reached the target, so there is no info about motor behaviour planning in the data
# 3. 

# labels = threshold_binary(data[:, -1], thresh=0.85)
# print("data: ", labels.shape, labels[230000:230500]) #data:  (282564,)
# plot_label_distribution_blocks(labels, block_size=1000)
# # data for finger MRS
# data = npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/sbp_position_min0.15_max0.85.npy")
# print("data: ", data.shape)
# # ------- Each feature's data ------- #
# # data without position data
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/sbp_all_position_min0.15_max0.85.npy")
# # data only position data for finger IDX
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/idx_all_position_min0.15_max0.85.npy")
# # data only position data for finger MRS
# npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/mrs_all_position_min0.15_max0.85.npy")