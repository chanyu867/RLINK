import numpy as np

import matplotlib.pyplot as plt



def dates_from_days(bad_days, temp_info):
    temp_info = np.asarray(temp_info)
    bad_days = np.asarray(bad_days, dtype=int)

    day_vals = temp_info[:, 1].astype(int)
    date_vals = temp_info[:, 2].astype(int)

    # bad_days に対応する date を unique で返す
    return np.unique(date_vals[np.isin(day_vals, bad_days)]).tolist()

def plot_blocks_and_avg_samples_per_block(temp_info, day_col=1, block_col=3,
                                          title_top="#Blocks per day",
                                          title_bottom="Avg #samples per block (within day)"):
    temp_info = np.asarray(temp_info)
    days = temp_info[:, day_col].astype(int)
    blocks = temp_info[:, block_col].astype(int)

    uniq_days = np.unique(days)

    n_blocks_per_day = []
    avg_samples_per_block_per_day = []

    for d in uniq_days:
        b = blocks[days == d]
        uniq_b, counts = np.unique(b, return_counts=True)  # counts = samples per block
        n_blocks_per_day.append(len(uniq_b))
        avg_samples_per_block_per_day.append(float(np.mean(counts)) if len(counts) > 0 else 0.0)

    # plot (2-row subplot)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    axes[0].bar(uniq_days, n_blocks_per_day)
    axes[0].set_ylabel("#Blocks")
    axes[0].set_title(title_top)

    axes[1].bar(uniq_days, avg_samples_per_block_per_day)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Avg samples/block")
    axes[1].set_title(title_bottom)

    plt.tight_layout()
    plt.show()




def heatmap_sbp_mean_by_electrode_day(sbp, temp_info, day_col=1, title="SBP mean (electrode × day)"):
    sbp = np.asarray(sbp)                 # (N, 96)
    temp_info = np.asarray(temp_info)     # (N, ?)

    print("shapes: ", sbp.shape, temp_info.shape)

    days = temp_info[:, day_col].astype(int)
    uniq_days = np.unique(days)           # sorted
    n_elec = sbp.shape[1]

    # mean_sbp[e, d] = mean SBP of electrode e on day uniq_days[d]
    mean_sbp = np.full((n_elec, len(uniq_days)), np.nan, dtype=float)

    for j, d in enumerate(uniq_days):
        idx = (days == d)
        if np.any(idx):
            mean_sbp[:, j] = np.mean(sbp[idx, :], axis=0)

    # plot heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mean_sbp, aspect="auto", interpolation="nearest", vmax=10)
    plt.colorbar(im, label="Mean SBP")
    plt.xlabel("Day")
    plt.ylabel("Electrode")
    plt.title(title)

    # x ticks: show every ~3rd day label (adjust if you want)
    step = max(1, len(uniq_days)//10)  # keep labels readable
    xticks = np.arange(0, len(uniq_days), step)
    plt.xticks(xticks, uniq_days[xticks])

    plt.tight_layout()
    plt.show()

