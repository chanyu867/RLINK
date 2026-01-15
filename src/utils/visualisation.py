import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_finger_movement_diff_per_session(position_diff, days_data, sessions_data, trials_data, day_num, n_sessions = None, legend = False):
    day_mask = days_data == day_num

    if n_sessions is not None:
        plot_sessions = np.unique(sessions_data)[:n_sessions]
        session_mask = np.isin(sessions_data, plot_sessions)
        mask = day_mask & session_mask
    else:
        mask = day_mask

    day_sessions = sessions_data[mask]
    day_trials = trials_data[mask]
    day_diff = position_diff[mask]

    df = pd.DataFrame({
        "Trial": day_trials,
        "Position Diff": day_diff,
        "Session": day_sessions
    })

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df,
        x="Trial",
        y="Position Diff",
        hue="Session",
        palette="tab20",
        alpha=0.7,
        legend=legend
    )

    plt.title(f"Position Diff vs Trial (Day {day_num})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # load data
    raw_npy_dir = "path_to_npy_files"

    idx_true = np.load(f"{raw_npy_dir}/idx_position_all.npy")
    mrs_true = np.load(f"{raw_npy_dir}/mrs_position_all.npy")

    idx_target = np.load(f"{raw_npy_dir}/idx_target_position.npy")
    mrs_target = np.load(f"{raw_npy_dir}/mrs_target_position.npy")

    idx_diff = idx_target - idx_true
    mrs_diff = mrs_target - mrs_true

    idx_diff_abs = np.abs(idx_diff)
    mrs_diff_abs = np.abs(mrs_diff) 

    # load trial info
    # days --> sessions --> trials
    day_number = np.load(f"{raw_npy_dir}/day_number.npy")
    session = np.load(f"{raw_npy_dir}/trial_number.npy")
    trials = np.load(f"{raw_npy_dir}/trial_bin.npy")

    # plot
    day_num_to_plot = 100
    n_sessions_to_plot = 10
    plot_finger_movement_diff_per_session(
        position_diff = idx_diff_abs,
        days_data = day_number,
        sessions_data = session,
        trials_data = trials,
        day_num = day_num_to_plot,
        n_sessions = n_sessions_to_plot,
        legend = False                      # optional if you want desc what line represents what session, only for small n_sessions
    )