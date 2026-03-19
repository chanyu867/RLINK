import os
import numpy as np
import matplotlib.pyplot as plt

def compute_block_switch_performance(xs, day_to_acc, temp_info):
    temp_info = np.asarray(temp_info)
    days_all = temp_info[:, 1].astype(int)
    block_ids_all = temp_info[:, 3].astype(int)

    results = {}

    for day, acc01 in zip(xs, day_to_acc):
        day = int(day)
        acc01 = np.asarray(acc01).astype(int)

        idx = (days_all == day)
        blocks_day = block_ids_all[idx]

        if len(acc01) != len(blocks_day):
            raise ValueError(f"Day {day}: acc01 and block_ids length mismatch")

        n = len(acc01)

        # --- (1) 全体平均正答率 ---
        mean_acc = acc01.mean()

        # --- (2) block切り替わり時点の正答率 ---
        switch_mask = np.zeros(n, dtype=bool)
        switch_mask[1:] = (blocks_day[1:] != blocks_day[:-1])

        if switch_mask.any():
            switch_acc = acc01[switch_mask].mean()
            n_switch = switch_mask.sum()
        else:
            switch_acc = np.nan
            n_switch = 0

        # print
        print(f"\n[Day {day}]")
        print(f"  overall accuracy           : {mean_acc:.4f} (n={n})")
        if n_switch > 0:
            print(f"  block-switch accuracy      : {switch_acc:.4f} (n={n_switch})")
        else:
            print(f"  block-switch accuracy      : N/A (no block switch)")

        results[day] = {
            "overall_accuracy": mean_acc,
            "block_switch_accuracy": switch_acc,
            "n_samples": n,
            "n_block_switch": n_switch,
        }

    plot_block_switch_accuracy(results)

    return results



def plot_block_switch_accuracy(results, figsize=(6, 4)):
    """
    results: compute_block_switch_performance が返す dict
             results[day]["block_switch_accuracy"]
    """

    days = sorted(results.keys())
    block_acc = np.array([results[d]["block_switch_accuracy"] for d in days], dtype=float)

    # NaN 除外
    valid = ~np.isnan(block_acc)
    days = np.array(days)[valid]
    block_acc = block_acc[valid]

    plt.figure(figsize=figsize)
    plt.plot(days, block_acc, marker="o")
    plt.xlabel("day")
    plt.ylabel("block-switch accuracy")
    plt.title("Performance at block switch points")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_block_switch_vs_overall_fair(
    results, xs, day_to_acc, temp_info,
    seed=0, figsize=(6, 4)
):
    """
    block-switch 点と、
    同数の非 switch 点をランダム抽出して比較する（dayごと）

    Parameters
    ----------
    results : dict
        compute_block_switch_performance の返り値
    xs : list[int]
        day番号
    day_to_acc : list[np.ndarray]
        各dayの 0/1 正誤系列
    temp_info : np.ndarray
        temp_info[:,1]=day, [:,3]=block_id
    """

    rng = np.random.default_rng(seed)

    temp_info = np.asarray(temp_info)
    days_all = temp_info[:, 1].astype(int)
    block_ids_all = temp_info[:, 3].astype(int)

    days_plot = []
    acc_switch = []
    acc_random = []

    for day, acc01 in zip(xs, day_to_acc):
        day = int(day)
        acc01 = np.asarray(acc01).astype(int)

        idx = (days_all == day)
        blocks_day = block_ids_all[idx]

        if len(acc01) != len(blocks_day):
            raise ValueError(f"Day {day}: length mismatch")

        n = len(acc01)
        if n < 2:
            continue

        # block switch mask
        switch_mask = np.zeros(n, dtype=bool)
        switch_mask[1:] = (blocks_day[1:] != blocks_day[:-1])

        n_switch = switch_mask.sum()
        if n_switch == 0:
            continue

        # 非 switch 点から同数ランダム抽出
        nonswitch_idx = np.flatnonzero(~switch_mask)
        if len(nonswitch_idx) < n_switch:
            continue  # 念のため

        sampled_idx = rng.choice(nonswitch_idx, size=n_switch, replace=False)

        acc_switch.append(acc01[switch_mask].mean())
        acc_random.append(acc01[sampled_idx].mean())
        days_plot.append(day)

    # --- plot ---
    plt.figure(figsize=figsize)
    plt.plot(days_plot, acc_switch, marker="o", label="block-switch (actual)")
    plt.plot(days_plot, acc_random, marker="s", label="non-switch (random-matched)")

    plt.xlabel("day")
    plt.ylabel("accuracy")
    plt.title("Block-switch vs matched non-switch accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_continuous_vs_blockchange_same_vs_flip(
    xs, day_to_acc, temp_info, positions,
    seed=0, figsize=(8, 4),
    low_rng=(0.0, 0.15), high_rng=(0.85, 1.0)
):
    """
    1つのプロットに3本線:
      - continuous samples (random-matched non-switch)
      - block change (same: low->low or high->high)
      - block change (flip: low->high or high->low)

    Parameters
    ----------
    xs : list[int]
        day番号（accuracy_over_time と同順）
    day_to_acc : list[np.ndarray]
        各dayの 0/1 正誤系列（xsと同順）
    temp_info : np.ndarray
        全サンプル情報。temp_info[:,1]=day, temp_info[:,3]=block_id
    positions : np.ndarray
        全サンプルの position (例: data[:,-1])
    """

    rng = np.random.default_rng(seed)

    temp_info = np.asarray(temp_info)
    positions = np.asarray(positions)

    days_all = temp_info[:, 1].astype(int)
    block_all = temp_info[:, 3].astype(int)

    lo0, lo1 = low_rng
    hi0, hi1 = high_rng

    def to_state(p):
        """low=0, high=1, その他は-1(無視)"""
        if lo0 <= p <= lo1:
            return 0
        if hi0 <= p <= hi1:
            return 1
        return -1

    days_plot = []
    cont_line = []
    same_line = []
    flip_line = []

    for day, acc01 in zip(xs, day_to_acc):
        day = int(day)
        acc01 = np.asarray(acc01).astype(int)

        idx = (days_all == day)
        blocks = block_all[idx]
        pos = positions[idx]

        if len(acc01) != len(blocks) or len(pos) != len(blocks):
            raise ValueError(f"Day {day}: length mismatch among acc01/blocks/pos")

        n = len(acc01)
        if n < 2:
            continue

        # --- block switch mask (t>=1) ---
        is_switch = np.zeros(n, dtype=bool)
        is_switch[1:] = (blocks[1:] != blocks[:-1])
        is_nonswitch = ~is_switch

        # --- position state transition (t>=1 uses pos[t-1] -> pos[t]) ---
        prev = np.array([to_state(p) for p in pos[:-1]], dtype=int)
        curr = np.array([to_state(p) for p in pos[1:]], dtype=int)

        valid_1 = (prev != -1) & (curr != -1)  # length n-1

        same_1 = valid_1 & (prev == curr)      # low->low or high->high
        flip_1 = valid_1 & (prev != curr)      # low<->high

        same = np.zeros(n, dtype=bool); same[1:] = same_1
        flip = np.zeros(n, dtype=bool); flip[1:] = flip_1
        valid = np.zeros(n, dtype=bool); valid[1:] = valid_1

        # --- block-change groups ---
        same_switch = is_switch & same
        flip_switch = is_switch & flip

        n_same = int(same_switch.sum())
        n_flip = int(flip_switch.sum())
        n_change = n_same + n_flip

        # その日に block-change が無いならスキップ
        if n_change == 0:
            continue

        a_same = float(acc01[same_switch].mean()) if n_same > 0 else np.nan
        a_flip = float(acc01[flip_switch].mean()) if n_flip > 0 else np.nan

        # --- continuous baseline (random-matched from non-switch & valid) ---
        pool = np.flatnonzero(is_nonswitch & valid)
        a_cont = np.nan
        if len(pool) >= n_change:
            sampled = rng.choice(pool, size=n_change, replace=False)
            a_cont = float(acc01[sampled].mean())

        days_plot.append(day)
        cont_line.append(a_cont)
        same_line.append(a_same)
        flip_line.append(a_flip)

    if len(days_plot) == 0:
        print("No days to plot: no valid block-change events found under the given thresholds.")
        return

    # day順にソート
    order = np.argsort(days_plot)
    days_plot = np.array(days_plot)[order]
    cont_line = np.array(cont_line, dtype=float)[order]
    same_line = np.array(same_line, dtype=float)[order]
    flip_line = np.array(flip_line, dtype=float)[order]

    # --- plot (single figure, 3 lines) ---
    # plt.figure(figsize=figsize)
    # plt.plot(days_plot, cont_line, marker="o", label="continuous samples (random-matched non-switch)")
    # plt.plot(days_plot, same_line, marker="s", label="block change (same: low→low / high→high)")
    # plt.plot(days_plot, flip_line, marker="^", label="block change (flip: low↔high)")

    # plt.xlabel("day")
    # plt.ylabel("accuracy")
    # plt.title("Continuous vs block-change performance (matched sample size)")
    # plt.ylim(0, 1)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1] * 1.6), sharex=True, sharey=True)

    axes[0].plot(days_plot, cont_line, marker="o")
    axes[0].set_title("continuous samples (random-matched non-switch)")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(True)

    axes[1].plot(days_plot, same_line, marker="s")
    axes[1].set_title("block change (same: low→low / high→high)")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(True)

    axes[2].plot(days_plot, flip_line, marker="^")
    axes[2].set_title("block change (flip: low↔high)")
    axes[2].set_xlabel("day")
    axes[2].set_ylabel("accuracy")
    axes[2].grid(True)

    for ax in axes:
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def plot_block_performance(acc, block_size=500, title=None):
    y_true, y_pred = acc
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}")

    n = y_true.shape[0]
    n_blocks = (n + block_size - 1) // block_size

    xs = np.arange(n_blocks) * block_size  # block start indices
    accs = np.empty(n_blocks, dtype=float)

    for b in range(n_blocks):
        s = b * block_size
        e = min((b + 1) * block_size, n)
        accs[b] = (y_true[s:e] == y_pred[s:e]).mean() * 100.0

    overall = (y_true == y_pred).mean() * 100.0
    print(f"Overall accuracy: {overall:.2f}%")
    print(f"Blocks: {n_blocks} (block_size={block_size}), last block size={n - (n_blocks-1)*block_size}")

    plt.figure(figsize=(10, 4))
    plt.plot(xs, accs, marker="o", linewidth=1)
    plt.ylim(0, 100)
    plt.xlabel("Sample index (block start)")
    plt.ylabel("Accuracy per block (%)")
    plt.title(title or f"Performance per {block_size}-sample block (overall={overall:.2f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_performance_compare(expected_fig_path, title, xs, accs_list,labels, ylim=(0, 1)):
    plt.figure(figsize=(10, 5))
    for accs, lab in zip(accs_list, labels):
        plt.plot(xs, accs, marker="o", label=lab)

    plt.grid(True)
    plt.ylim(ylim)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Time (block index)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    #make sure directory is existing
    plt.savefig(expected_fig_path)
    plt.close()
    # plt.show()
