import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def npy_loader(path):

    arr = np.load(path, allow_pickle=True)  # for sbp/mrs this should be numeric; no pickle needed

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

# def plot_position_per_day(accs, time_info, data, break_day=None, figsize=(12, 4)):
#     time_info = np.asarray(time_info)
#     y = np.asarray(data)[:, -1].astype(float)

#     days = time_info[:, 1].astype(int)
#     block_ids = time_info[:, 3].astype(int)

#     if y.shape[0] != time_info.shape[0]:
#         raise ValueError("data rows must match time_info rows")

#     unique_days = np.unique(days)
#     flag_days = []

#     for day in unique_days:
#         idx = (days == day)
#         y_day = y[idx]
#         b_day = block_ids[idx]

#         n = len(y_day)
#         if n == 0:
#             continue

#         # ブロック開始点
#         starts = np.flatnonzero(np.r_[True, b_day[1:] != b_day[:-1]])
#         ends = np.r_[starts[1:], n]  # end is exclusive
#         if any((y_day[s:e] <= 0.15).any() and (y_day[s:e] >= 0.85).any() for s, e in zip(starts, ends)): flag_days.append(day)

#         # ブロック間に空白を作るため、ブロックごとにxを連結しつつギャップを入れる
#         gap = 5  # 空白幅（サンプル数相当）。見た目用
#         x_segments = []
#         y_segments = []
#         start_points = []  # (x, y) for block start
#         end_points = []    # (x, y) for block end

#         x_cursor = 0
#         for s, e in zip(starts, ends):
#             L = e - s
#             xs = np.arange(x_cursor, x_cursor + L)
#             ys = y_day[s:e]

#             x_segments.append(xs)
#             y_segments.append(ys)

#             start_points.append((xs[0], ys[0]))
#             end_points.append((xs[-1], ys[-1]))

#             x_cursor = xs[-1] + 1 + gap  # 次ブロックへジャンプ（ここが空白になる）

#         # --- plot ---
#         fig, ax = plt.subplots(figsize=figsize)

#         # ブロックごとに別々に plot するので、ブロック間は繋がらない
#         for xs, ys in zip(x_segments, y_segments):
#             ax.plot(xs, ys, linewidth=1)

#         # ブロック開始・終了点（開始=赤, 終了=青）
#         if len(start_points) > 0:
#             ax.scatter([p[0] for p in start_points], [p[1] for p in start_points], color="red", s=10, label="block start")
#         if len(end_points) > 0:
#             ax.scatter([p[0] for p in end_points], [p[1] for p in end_points], color="blue", s=10, label="block end")

#         ax.set_title(f"Day {day}: position (data[:,-1]) per block (gaps show block boundaries)")
#         ax.set_xlabel("sample index (with gaps at block boundaries)")
#         ax.set_ylabel("position / label")
#         ax.grid(True)
#         ax.legend(loc="best")

#         plt.tight_layout()
#         plt.savefig(f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/day_to_day_position/{day}.png")
#         plt.close()

#         if break_day is not None and day > break_day:
#             break

#     return flag_days

def plot_position_per_day(acc_days, time_info, data, break_day=None, figsize=(12, 4)):
    time_info = np.asarray(time_info)
    y = np.asarray(data).astype(float)

    days_all = time_info[:, 1].astype(int)
    block_ids_all = time_info[:, 3].astype(int)

    if y.shape[0] != time_info.shape[0]:
        raise ValueError("data rows must match time_info rows")

    unique_days = np.unique(days_all)  # accuracy_over_time と同じ並び（昇順）になる想定

    # --- day -> acc_01 の対応づけ ---
    if len(acc_days) != len(unique_days):
        raise ValueError(
            f"len(acc_days)={len(acc_days)} must match number of unique days={len(unique_days)}. "
            "Make sure you pass day_to_accs from accuracy_over_time."
        )
    day_to_acc01 = {int(d): np.asarray(a).astype(int) for d, a in zip(unique_days, acc_days)}

    flag_days = []

    for day in unique_days:
        day = int(day)
        idx = (days_all == day)

        y_day = y[idx]
        b_day = block_ids_all[idx]
        a_day = day_to_acc01[day]   # ← その日の 0/1 正誤系列

        n = len(y_day)
        if n == 0:
            continue

        if a_day.shape[0] != n:
            raise ValueError(f"Day {day}: acc length {a_day.shape[0]} != samples {n}. "
                             "Check that y_true/y_pred indexing matches temp_info day indexing.")

        # ブロック開始点/終了点
        starts = np.flatnonzero(np.r_[True, b_day[1:] != b_day[:-1]])
        ends = np.r_[starts[1:], n]  # end is exclusive

        # 0-0.15 と 0.85-1.0 が同一ブロック内に混在する日を記録
        if any((y_day[s:e] <= 0.15).any() and (y_day[s:e] >= 0.85).any() for s, e in zip(starts, ends)):
            flag_days.append(day)

        # ブロック間ギャップを作る
        gap = 5
        x_segments, y_segments, a_segments = [], [], []
        start_points, end_points = [], []

        x_cursor = 0
        for s, e in zip(starts, ends):
            L = e - s
            xs = np.arange(x_cursor, x_cursor + L)
            ys = y_day[s:e]
            aa = a_day[s:e]

            x_segments.append(xs)
            y_segments.append(ys)
            a_segments.append(aa)

            start_points.append((xs[0], ys[0]))
            end_points.append((xs[-1], ys[-1]))

            x_cursor = xs[-1] + 1 + gap

        # --- plot ---
        fig, ax = plt.subplots(figsize=figsize)

        # 正解=赤 / 不正解=青（点で表示）
        for xs, ys, aa in zip(x_segments, y_segments, a_segments):
            colors = np.where(aa == 1, "red", "blue")
            ax.scatter(xs, ys, c=colors, s=3, linewidths=0)

        # 始点・終点は灰色に統一
        if len(start_points) > 0:
            ax.scatter([p[0] for p in start_points], [p[1] for p in start_points],
                       color="gray", s=20, label="block start", zorder=5)
        if len(end_points) > 0:
            ax.scatter([p[0] for p in end_points], [p[1] for p in end_points],
                       color="gray", s=20, label="block end", zorder=5)

        ax.set_title(f"Day {day}: position (red=correct, blue=wrong; gray=start/end)")
        ax.set_xlabel("sample index (with gaps at block boundaries)")
        ax.set_ylabel("position / label")
        ax.grid(True)
        ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/day_to_day_position/{day}.png")
        plt.close()

        if break_day is not None and day > break_day:
            break

    return flag_days


#How to run
# ------- Combined data ------- #
# data for finger IDX
# sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/sbp_with_mrs_position_min0.15_max0.85.npy"
# data = npy_loader(sbp_path)
# # print("data: ", data[:500, -1])

# data = npy_loader("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/time_data/day_number.npy")
# # mask = npy_loader("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/masks/idx_position_mask_0.2_0.4_0.6_0.8_shift20.npy")
# data = data[data>90] #(5369900,)
# print("data: ", np.unique(data), data.shape)
# count_1 = data[data==0.0]
# masked_data = data[mask==1]
# print("data: ", data.shape, masked_data.shape, data.shape[0]-(117000*20)) #(7711816,) (7360816,)
# print("trial_bin: ", len(count_1), 117000)

# unique, counts = np.unique(data, return_counts=True)
# print("unique counts: ", len(unique), unique)
# all = 0
# for val, count in zip(unique, counts):
#     all+=(count-3)
#     # print(f"Value {val}: {count} occurrences")
# print("data: ", data[:100])

# time_info = npy_loader(path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/timestamps_position_min0.15_max0.85.npy")
# time_info = add_block_id(time_info)
# # print("data: ", data[:100])

# flag_days = plot_position_per_day(time_info, data, break_day=None)

# print("flag_days: ", flag_days)


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