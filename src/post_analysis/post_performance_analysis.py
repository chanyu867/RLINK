import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import colorsys

import numpy as np
from scipy.stats import spearmanr


def spearman_top_quantile(collected, model, delta_thr=0.1, min_n=20):

    rows = collected.get(model, [])
    if len(rows) == 0:
        return None

    dpos = np.concatenate([r["dpos"] for r in rows])
    dacc = np.concatenate([r["dacc"] for r in rows])

    mask = dpos >= delta_thr
    if mask.sum() < min_n:
        print(f"Not enough samples above delta_thr={delta_thr}: n={mask.sum()}")
        return {"model": model, "delta_thr": delta_thr, "rho": np.nan, "p": np.nan, "n": int(mask.sum())}

    rho, p = spearmanr(dpos[mask], dacc[mask])
    
    return {"model": model, "delta_thr": float(delta_thr), "rho": float(rho), "p": float(p), "n": int(mask.sum())}


from pathlib import Path
def load_results_dict(npy_path: str) -> dict:
    if isinstance(npy_path, (tuple, list)):
        npy_path = npy_path[0]

    npy_path = str(Path(npy_path))  # normalize
    obj = np.load(npy_path, allow_pickle=True)
    return obj.item() if hasattr(obj, "item") else obj

def trial_block_means(correct_01: np.ndarray, time_within_trial: np.ndarray, block: int = 10):
    correct_01 = np.asarray(correct_01).astype(float)
    time_within_trial = np.asarray(time_within_trial)

    starts = np.where(time_within_trial == 0.0)[0]
    if len(starts) == 0:
        raise ValueError("No trial start found: time_within_trial==0.0 is empty.")
    ends = np.r_[starts[1:], len(time_within_trial)]

    out = []
    for s, e in zip(starts, ends):
        trial = correct_01[s:e]
        L = len(trial)
        if L < block:
            continue
        cut = (L // block) * block
        trial = trial[:cut]  # drop tail
        chunk = cut // block
        out.append(trial.reshape(block, chunk).mean(axis=1))
    return out

def plot_daywise_trial_blocks(res_path,
        day_number,
        time_within_trial,
        block: int = 10,
        figsize=(8, 5),
        max_trials_per_day=None,
        do_plot=False
    ):

    day_number = np.asarray(day_number)
    time_within_trial = np.asarray(time_within_trial)
    unique_days = np.unique(day_number)

    # --- hue spacing that avoids adjacent similar colors (golden ratio step) ---
    def trial_color(t):
        phi = 0.618033988749895
        h = (t * phi) % 1.0
        s, v = 0.55, 0.80  # softer
        return colorsys.hsv_to_rgb(h, s, v)

    # --- block means for arbitrary values (not only correct_01) ---
    def values_trial_block_means(values: np.ndarray, twt: np.ndarray, block: int):
        values = np.asarray(values).astype(float)
        twt = np.asarray(twt)

        starts = np.where(twt == 0.0)[0]
        if len(starts) == 0:
            raise ValueError("No trial start found: time_within_trial==0.0 is empty.")
        ends = np.r_[starts[1:], len(twt)]

        out = []
        for s, e in zip(starts, ends):
            trial = values[s:e]
            L = len(trial)
            if L < block:
                continue
            cut = (L // block) * block
            trial = trial[:cut]
            chunk = cut // block
            out.append(trial.reshape(block, chunk).mean(axis=1))
        return out

    collected = {}

    res = load_results_dict(res_path)
    model = res["meta"].get("model_type", "model")
    y_true = np.asarray(res["prediction"]["y_true"])
    y_pred = np.asarray(res["prediction"]["y_pred"])
    collected.setdefault(model, [])

    if not (len(y_true) == len(y_pred) == len(day_number) == len(time_within_trial)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"day_number={len(day_number)}, time_within_trial={len(time_within_trial)}"
        )

    correct_01 = (y_true == y_pred).astype(int)

    # map class-id -> coarse position in [0,1] using number of predicted classes
    classes = np.unique(y_pred)
    K = len(classes)
    w = 1.0 / K

    for day in unique_days:
        idx = (day_number == day)
        c_day = correct_01[idx]
        twt_day = time_within_trial[idx]
        ytrue_day = y_true[idx]
        ytrue_pos_day = (ytrue_day.astype(float) + 0.5) * w

        # per-trial block means
        trial_means = trial_block_means(c_day, twt_day, block=block)                # accuracy(0/1) block mean
        ytrue_means = values_trial_block_means(ytrue_pos_day, twt_day, block=block) # y_true position block mean

        if max_trials_per_day is not None:
            trial_means = trial_means[:max_trials_per_day]
            ytrue_means = ytrue_means[:max_trials_per_day]

        n_trials = min(len(trial_means), len(ytrue_means))
        # print(f"day: {day}, number of trials: {n_trials}")
        if n_trials == 0:
            continue
        
        if do_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharey=False)

        # ---------- subplot 1: concatenated across trials ----------
        x_offset = 0
        y_concat = []
        yt_concat = []

        # save for the post performance correlation analysis (exclude trial boundary)
        dpos = []  # within-trial position change
        dacc = []  # within-trial accuracy change

        # save for the post performance correlation analysis (trial boundary only)
        dpos_boundary = []  # boundary position change (end of trial -> start of next trial)
        dacc_boundary = []  # boundary accuracy change (end of trial -> start of next trial)

        prev_last_m = None
        prev_last_yt = None

        for t in range(n_trials):
            m = np.asarray(trial_means[t], dtype=float)  # shape: (n_blocks,), block-averaged performance for trial t
            yt = np.asarray(ytrue_means[t], dtype=float) # shape: (n_blocks,), block-averaged true label/positions for trial t
            L = min(len(m), len(yt))
            m = m[:L]
            yt = yt[:L]
            if L == 0:
                continue

            # plot colored trial segment (accuracy)
            x = np.arange(L) + x_offset
            if do_plot:
                ax1.plot(x, m, color=trial_color(t), linewidth=1.2, alpha=0.9)

            y_concat.append(m)
            yt_concat.append(yt)
            x_offset += L

            # --- trial boundary effect (end of previous trial -> start of current trial) ---
            if prev_last_m is not None:
                dpos_boundary.append(abs(yt[0] - prev_last_yt)) #meaning the positon at first value in current trial
                dacc_boundary.append(m[0] - prev_last_m) #take gap between last block mean of previous trial and first block mean of current trial

            # --- within-trial deltas (no boundary) ---
            if L >= 2:
                dpos.extend(np.abs(np.diff(yt)).tolist())
                dacc.extend(np.diff(m).tolist())

            # store last point for next boundary calculation
            prev_last_m = m[-1] #save extracted trial's last block mean
            prev_last_yt = yt[-1]

        if len(y_concat) == 0:
            plt.close(fig)
            continue

        # add gray y_true line (same block-averaged & concatenated)
        yt_concat = np.concatenate(yt_concat) # flatten all -> all values are consequensed

        # within-trial (no boundary)
        dpos = np.asarray(dpos, dtype=float)  # position change
        dacc = np.asarray(dacc, dtype=float)  # accuracy change

        # trial-boundary only
        dpos_boundary = np.asarray(dpos_boundary, dtype=float)
        dacc_boundary = np.asarray(dacc_boundary, dtype=float)

        collected[model].append({
            "acc_block": np.array(y_concat),
            "acc_block_mean": np.mean(np.concatenate(y_concat)),
            "day": int(day),
            "dpos": dpos, #withi-trial position change
            "dacc": dacc, #within-trial accuracy change
            "dpos_boundary": dpos_boundary, #trial-boundary position change
            "dacc_boundary": dacc_boundary, #trial-boundary accuracy change
        })

        if do_plot:
            # y_pos (block-averaged, concatenated) on the same axis
            ax1.plot(range(len(yt_concat)), yt_concat, linestyle="--", color="red", linewidth=1.2, alpha=0.85)

            ax1.set_title(f"{model} | Day {day}  (trials={n_trials}, blocks/trial={block})")
            ax1.set_xlabel("Concatenated block index across trials")
            ax1.set_ylabel("Block mean")
            ax1.set_ylim(-0.05, 1.05)
            ax1.grid(True, alpha=0.25)

            # ---------- subplot 2: per-trial vs block index ----------
            for t in range(n_trials):
                m = np.asarray(trial_means[t], dtype=float)
                x = np.arange(len(m))  # IMPORTANT: match actual length (prevents broken lines)
                ax2.plot(x, m, color=trial_color(t), linewidth=0.8, alpha=0.8)

            ax2.set_xlabel("Block index within trial")
            ax2.set_ylabel("Accuracy (mean)")
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.5)

            plt.tight_layout()
        
        if do_plot:
            plt.close()

    return collected


def daywise_first_second_half_means(collected, model, min_blocks=2, ratio=0.5):
    rows = collected.get(model, [])
    day_to_first = {}
    day_to_second = {}
    day_to_n = {}

    for r in rows:
        day = int(r["day"])
        trials = r.get("acc_block", None)
        if trials is None:
            continue

        for m in trials:
            m = np.asarray(m, dtype=float)
            L = len(m)
            if L < min_blocks:
                continue

            cut = int(np.floor(L * ratio))
            cut = max(1, min(cut, L - 1))  # ensure both parts non-empty

            day_to_first.setdefault(day, []).append(float(np.mean(m[:cut])))
            day_to_second.setdefault(day, []).append(float(np.mean(m[cut:])))
            day_to_n[day] = day_to_n.get(day, 0) + 1

    out = {}
    for day in sorted(day_to_n.keys()):
        out[day] = {
            "first_half_mean": float(np.mean(day_to_first[day])),
            "second_half_mean": float(np.mean(day_to_second[day])),
            "n_trials": int(day_to_n[day]),
        }

    return out

def stability_metrics_from_collected(collected, model, include_boundary=False, eps=0.0):

    rows = collected.get(model, [])
    if len(rows) == 0:
        raise ValueError(f"model '{model}' not found in collected.")

    xs = []
    for r in rows:
        if "dacc" in r and len(r["dacc"]) > 0:
            xs.append(np.asarray(r["dacc"], dtype=float))
        if include_boundary and "dacc_boundary" in r and len(r["dacc_boundary"]) > 0:
            xs.append(np.asarray(r["dacc_boundary"], dtype=float))

    if len(xs) == 0:
        return {"model": model, "n": 0}

    x = np.concatenate(xs)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"model": model, "n": 0}

    # overall variability (symmetric)
    var_all = float(np.var(x, ddof=1)) if n > 1 else np.nan

    # downside-only (drops)
    neg = x[x < 0]
    n_neg = int(neg.size)
    var_down = float(np.var(neg, ddof=1)) if n_neg > 1 else (0.0 if n_neg == 1 else np.nan)

    # downside risk (squared negative changes, includes magnitude)
    downside_risk = float(np.mean(np.minimum(x, 0.0) ** 2))

    # drop probability beyond eps
    p_drop = float(np.mean(x < -eps))

    return {
        "model": model,
        "n": n,
        "var_all": var_all,
        "downside_risk": downside_risk,
        "n_neg": n_neg,
        "var_down": var_down,
        "p_drop": p_drop,
        "eps": float(eps),
        "include_boundary": bool(include_boundary),
    }

# ------------------------- #
# Example usage
# ------------------------- #

path_to_prediction = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/pred_results/position"
path_to_day_num = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
path_to_trial_bin = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy"
path_to_mask = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/masks"

class_task_table = {
    3: "0.33_0.66",
    4: "0.25_0.5_0.75",
    5: "0.2_0.4_0.6_0.8"
}

from src.utils.npy_loader import npy_loader
# results_npy_paths: list of saved result dicts (one per decoder), e.g.
for shift in [1]: #need to done for 0 shifting
    for model in ["banditron", "banditronRP", "HRL", "AGREL"]:
        for num_class, task in class_task_table.items():
            res_path = f"{path_to_prediction}/{num_class}classes/results_idx_{model}_shift{shift}_seed1.npy",

            day_number = npy_loader(path_to_day_num)
            time_within_trial = npy_loader(path_to_trial_bin)

            #apply mask
            if shift != 0:
                shift_mask_path = f"{path_to_mask}/idx_position_mask_{task}_shift{shift}.npy"
                shift_mask = npy_loader(shift_mask_path).astype(bool)
                count_1 = time_within_trial[time_within_trial==0.0] #count how many trials are there
                day_number = day_number[shift_mask]
                time_within_trial = time_within_trial[shift_mask]

            collected = plot_daywise_trial_blocks(res_path, day_number, time_within_trial, block=20, do_plot=False) 
            #-> maybe "block"should be decided based on the distriution of trial lengths

            res = spearman_top_quantile(collected, model=model, delta_thr=0.01, per_day=False)
            out = stability_metrics_from_collected(collected, model=model, include_boundary=False)
            mean_perf = np.mean([d["acc_block_mean"] for d in collected[model]])
            perf_trend = daywise_first_second_half_means(collected, model, ratio=0.5)
            
            first_half_means = [v['first_half_mean'] for v in perf_trend.values()]
            second_half_means = [v['second_half_mean'] for v in perf_trend.values()]

            print(f"\n{'='*70}")
            print(f"Model: {model} | Temporal shift: {shift} | decoding task: {task}")
            print(f"{'='*70}")
            print(f"  Correlation (ρ):                                {res['rho']:>8.4f}")
            print(f"  Degrade Probability:                            {out['p_drop']:>8.4f}")
            print(f"  Mean Performance:                               {mean_perf:>8.4f}")
            print(f"  First Half perf. : Second Half perf.:           {np.mean(first_half_means):>8.4f} : {np.mean(second_half_means):>8.4f}")
            print(f"{'='*70}\n")