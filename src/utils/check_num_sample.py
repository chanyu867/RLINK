import numpy as np
import matplotlib as plt

def plot_samples_per_day(temp_info, sort=True, title="Samples per day"):
    temp_info = np.asarray(temp_info)
    days = temp_info[:, 1].astype(int)

    uniq, counts = np.unique(days, return_counts=True)

    if sort:
        order = np.argsort(uniq)
        uniq, counts = uniq[order], counts[order]

    plt.figure()
    plt.bar(uniq, counts)
    plt.xlabel("Day")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.tight_layout()
    plt.show()

