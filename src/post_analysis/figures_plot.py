import numpy as np
import matplotlib.pyplot as plt
import os, re, glob
import colorsys
from matplotlib.animation import FFMpegWriter
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

def load_results_dict(npy_path: str) -> dict:
    if isinstance(npy_path, (tuple, list)):
        npy_path = npy_path[0]
    npy_path = str(Path(npy_path))
    obj = np.load(npy_path, allow_pickle=True)
    return obj.item() if hasattr(obj, "item") else obj


def filter_result_paths_by_gamma(pattern, gamma):
    all_paths = sorted(glob.glob(pattern))
    out = []
    for p in all_paths:
        base = os.path.basename(p)
        if "_gamma" not in base:
            continue
        try:
            gstr = base.split("_gamma", 1)[1].rsplit(".npy", 1)[0]
        except IndexError:
            continue

        if gstr == "None" or gamma is None:
            out.append(p)
            continue
        try:
            gval = float(gstr)
        except Exception:
            continue

        if gamma == gval:
            out.append(p)

    return out


def plot_daywise_trial_blocks(res_paths, day_number, time_within_trial, block=10, 
                              figsize=(15, 6), max_trials_per_day=None, do_plot=False,
                              plot_transition_trials_only=False):

    day_number = np.asarray(day_number)
    time_within_trial = np.asarray(time_within_trial)
    unique_days = np.unique(day_number)

    def trial_color(t):
        phi = 0.618033988749895
        return colorsys.hsv_to_rgb((t * phi) % 1.0, 0.55, 0.80)

    collected_seeds = []

    for res_path in res_paths:
        collected = {}
        res = load_results_dict(res_path)
        model = res["meta"].get("model_type", "model")
        y_true = np.asarray(res["prediction"]["y_true"])
        y_pred = np.asarray(res["prediction"]["y_pred"])
        collected.setdefault(model, [])

        correct_01 = (y_true == y_pred).astype(int)
        num_target_classes = len(np.unique(y_true))
        w = 1.0 / num_target_classes

        if do_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharey=False)
            global_x_ax1 = 0

        trials_processed_total = 0

        for day in unique_days:
            idx = (day_number == day)
            c_day, twt_day = correct_01[idx], time_within_trial[idx]
            ytrue_day = y_true[idx]
            ytrue_pos_day = (ytrue_day.astype(float) + 0.5) * w

            starts = np.where(twt_day == 0.0)[0]
            if len(starts) == 0: continue
            ends = np.r_[starts[1:], len(twt_day)]

            y_concat, dpos, dacc, dpos_boundary, dacc_boundary = [], [], [], [], []
            prev_last_m, prev_last_yt = None, None
            trials_this_day = 0

            for s, e in zip(starts, ends):
                if max_trials_per_day and trials_this_day >= max_trials_per_day: break

                trial_y = ytrue_day[s:e]
                if plot_transition_trials_only and len(np.unique(trial_y)) < num_target_classes:
                    continue 

                trial_c, trial_pos = c_day[s:e], ytrue_pos_day[s:e]
                L = len(trial_c)
                if L < block: continue
                
                cut = (L // block) * block
                chunk = cut // block
                m = trial_c[:cut].reshape(block, chunk).mean(axis=1)
                yt = trial_pos[:cut].reshape(block, chunk).mean(axis=1)

                if do_plot:
                    x1 = np.arange(block) + global_x_ax1
                    ax1.plot(x1, m, color=trial_color(trials_processed_total), linewidth=1.2, alpha=0.9)
                    ax1.plot(x1, yt, linestyle="--", color="gray", linewidth=1.5, alpha=0.4)
                    
                    x2 = np.arange(block)
                    ax2.plot(x2, m, color=trial_color(trials_processed_total), linewidth=0.8, alpha=0.6)
                    global_x_ax1 += block

                y_concat.append(m)
                if prev_last_m is not None:
                    dpos_boundary.append(abs(yt[0] - prev_last_yt))
                    dacc_boundary.append(m[0] - prev_last_m)

                if block >= 2:
                    dpos.extend(np.abs(np.diff(yt)).tolist())
                    dacc.extend(np.diff(m).tolist())

                prev_last_m, prev_last_yt = m[-1], yt[-1]
                trials_this_day += 1
                trials_processed_total += 1

            if do_plot and trials_this_day > 0:
                ax1.axvline(x=global_x_ax1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax1.text(global_x_ax1, 1.05, f"Day {day}", color='red', fontsize=8, ha='center')

            if len(y_concat) > 0:
                collected[model].append({
                    "acc_block": np.array(y_concat),
                    "acc_block_mean": np.mean(np.concatenate(y_concat)),
                    "day": int(day),
                    "dpos": np.array(dpos, dtype=float),
                    "dacc": np.array(dacc, dtype=float),
                    "dpos_boundary": np.array(dpos_boundary, dtype=float),
                    "dacc_boundary": np.array(dacc_boundary, dtype=float),
                })

        if do_plot:
            mode_title = "Transition Trials ONLY" if plot_transition_trials_only else "All Trials"
            ax1.set_title(f"{model} | Continuous Timeline Across All Days | {mode_title}")
            ax1.set_xlabel("Concatenated block index")
            ax1.set_ylabel("Block mean accuracy")
            ax1.set_ylim(-0.05, 1.15)
            ax1.grid(True, alpha=0.25)

            ax2.set_title(f"{model} | Average shape of a trial (Overlay)")
            ax2.set_xlabel("Block index within trial")
            ax2.set_ylabel("Accuracy (mean)")
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.5)

            plt.tight_layout()
            plt.show()

        collected_seeds.append(collected)

    return collected_seeds


# ========================================================
# NEW: EVENT-ALIGNED RECOVERY CURVE (GOLD STANDARD)
# ========================================================

def plot_event_aligned_recovery(model_paths_dict, time_within_trial, pre_window=15, max_post_window=100):
    """
    ポジション変化直後の適応をプロットします。
    「一番短いトライアル」を見つけ出し、すべてのデータをその長さに切り揃えることで、
    データを捨てることなく、生存バイアス（N数の変動）を完全に防ぎます。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    mid_events = {m: [] for m in model_paths_dict.keys()}
    bound_events = {m: [] for m in model_paths_dict.keys()}
    
    # すべてのイベントの中で「最も短いトライアルの残り長さ」を記録する変数
    global_min_post_mid = max_post_window
    global_min_post_bound = max_post_window
    
    for model_name, res_paths in model_paths_dict.items():
        for path in res_paths:
            res = load_results_dict(path)
            y_true = np.asarray(res["prediction"]["y_true"])
            y_pred = np.asarray(res["prediction"]["y_pred"])
            
            transitions = np.where(y_true[1:] != y_true[:-1])[0] + 1
            
            for t in transitions:
                # ① 変化後のデータが「次のトライアル開始」までに何サンプルあるか(post_len)を計算
                max_fwd = min(max_post_window, len(y_true) - t)
                post_len = max_fwd
                for i in range(1, max_fwd):
                    if time_within_trial[t + i] == 0.0:
                        post_len = i  # 次のトライアルの直前までの長さを取得
                        break

                min_required_post = 5
                if post_len < min_required_post:
                    continue
                
                # ② 変化前のデータが「現在のトライアル開始」から何サンプルあるか(pre_len)を計算
                max_bwd = min(pre_window, t)
                pre_len = max_bwd
                if time_within_trial[t] != 0.0:
                    for i in range(1, max_bwd + 1):
                        if time_within_trial[t - i] == 0.0:
                            pre_len = i - 1
                            break
                
                # 変化前のベースライン(Hold)が確保できない極端に早すぎるイベントのみ除外
                if pre_len < pre_window:
                    continue
                    
                # 必要な長さだけ正確にスライスしてAccuracyを計算
                window_true = y_true[t - pre_window : t + post_len]
                window_pred = y_pred[t - pre_window : t + post_len]
                acc = (window_true == window_pred).astype(float)
                
                if time_within_trial[t] == 0.0:
                    bound_events[model_name].append(acc)
                    global_min_post_bound = min(global_min_post_bound, post_len)
                else:
                    # 壊れた時計（まぐれ当たり）を防ぐため、移動前の正解率が50%以上のものだけを採用
                    pre_transition_acc = np.mean(acc[:pre_window])
                    if pre_transition_acc > 0.5:
                        mid_events[model_name].append(acc)
                        global_min_post_mid = min(global_min_post_mid, post_len)

    # --- すべての配列を「最も短い長さに切り揃えて」平均化・プロット ---
    for ax, events_dict, min_post, title in zip(
        [ax1, ax2], 
        [mid_events, bound_events], 
        [global_min_post_mid, global_min_post_bound],
        ["Mid-Trial Adaptation", "Cross-Trial Reset Tolerance"]
    ):
        # 描画するx軸も「最小の長さ」に合わせる
        x_axis = np.arange(-pre_window, min_post)
        
        for model_name, ev_list in events_dict.items():
            if len(ev_list) > 0:
                trimmed = [arr[:pre_window + min_post] for arr in ev_list]
                avg_acc = np.mean(trimmed, axis=0)
                std_acc = np.std(trimmed, axis=0)
                
                # Plot the mean line and capture the color
                line, = ax.plot(x_axis, avg_acc, label=f"{model_name} (N={len(trimmed)})", linewidth=2)
                
                # Add the shaded variance region
                ax.fill_between(x_axis, 
                                np.clip(avg_acc - std_acc, 0, 1), 
                                np.clip(avg_acc + std_acc, 0, 1), 
                                color=line.get_color(), alpha=0.2)
                
        ax.axvline(x=0, color='black' if ax==ax1 else 'red', linestyle='--', linewidth=2)
        ax.set_xlim(-pre_window, min_post - 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Samples Relative to Transition")
        ax.set_ylabel("Average Accuracy")
        ax.set_title(f"{title}\n(Truncated to min valid length: {min_post} samples)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("RL Model Event-Aligned Recovery Curve (Constant-N, No Dropped Data)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_continuous_performance(model_paths_dict, day_number, resolution=10000, metric="f1"):
    """
    Plots the continuous performance of the models across the entire dataset.
    resolution: Number of data points (samples) to group together.
    metric: "f1" for Macro F1-Score, or "accuracy" for standard Accuracy.
    """
    plt.figure(figsize=(15, 6))
    
    day_number = np.asarray(day_number)
    day_changes = np.where(day_number[1:] != day_number[:-1])[0] + 1
    day_starts = np.insert(day_changes, 0, 0)
    unique_days = np.unique(day_number)
    
    for model_name, res_paths in model_paths_dict.items():
        if not res_paths:
            continue
            
        model_curves = []
        
        for path in res_paths:
            res = load_results_dict(path)
            y_true = np.asarray(res["prediction"]["y_true"])
            y_pred = np.asarray(res["prediction"]["y_pred"])
            
            n_samples = len(y_true)
            n_bins = int(np.ceil(n_samples / resolution))
            
            curve = []
            for i in range(n_bins):
                start = i * resolution
                end = min((i + 1) * resolution, n_samples)
                
                chunk_true = y_true[start:end]
                chunk_pred = y_pred[start:end]
                
                # --- METRIC SWITCHER ---
                if metric.lower() == "accuracy":
                    score = accuracy_score(chunk_true, chunk_pred)
                else:
                    score = f1_score(chunk_true, chunk_pred, average='macro', zero_division=0)
                    
                curve.append(score)
                
            model_curves.append(curve)
        
        avg_curve = np.mean(model_curves, axis=0)
        std_curve = np.std(model_curves, axis=0) 
        
        x_axis = np.arange(len(avg_curve)) * resolution 
        
        # Plot the mean line and capture the color
        line, = plt.plot(x_axis, avg_curve, label=model_name, linewidth=1.5, alpha=0.8)
        
        # Add the shaded variance region using the exact same color
        plt.fill_between(x_axis, 
                         np.clip(avg_curve - std_curve, 0, 1), # Clip at 0 and 1 so it doesn't spill off the graph
                         np.clip(avg_curve + std_curve, 0, 1), 
                         color=line.get_color(), alpha=0.2)
    
    # --- X-AXIS FORMATTING ---
    tick_indices = np.arange(0, len(unique_days), 10)
    tick_positions = day_starts[tick_indices]
    tick_labels = unique_days[tick_indices].astype(int)
    
    plt.xticks(tick_positions, tick_labels)

    # --- DYNAMIC TITLES ---
    metric_label = "Accuracy" if metric.lower() == "accuracy" else "Macro F1-Score"
    
    plt.title(f"Continuous {metric_label} over Time (Resolution: {resolution} samples/point)", fontsize=14, fontweight='bold')
    plt.xlabel("Recorded Days")
    plt.ylabel(metric_label)
    plt.ylim(0, 1.05) 
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()