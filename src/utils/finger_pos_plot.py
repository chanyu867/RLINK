#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def npy_loader(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def build_trial_ids(trial_bin: np.ndarray) -> np.ndarray:
    """
    Same spirit as mlp_train.build_trial_ids():
    trial_bin==0.0 indicates trial start. If first sample isn't start, force it.
    Returns trial_id per sample (0..n_trials-1).
    """
    trial_bin = np.asarray(trial_bin)
    starts = (trial_bin == 0.0)
    if starts.size == 0:
        return np.zeros(0, dtype=int)
    if not starts[0]:
        starts = starts.copy()
        starts[0] = True

    trial_id = np.zeros(trial_bin.shape[0], dtype=int)
    cur = -1
    for i, s in enumerate(starts):
        if s:
            cur += 1
        trial_id[i] = cur
    return trial_id


def choose_one_trial_start(trial_bin: np.ndarray,
                           target_style: np.ndarray,
                           want_random: bool,
                           rng: np.random.RandomState) -> int:
    trial_bin = np.asarray(trial_bin)
    target_style = np.asarray(target_style).astype(bool)

    starts = np.flatnonzero(trial_bin == 0.0)
    if starts.size == 0:
        raise RuntimeError("No trial starts found: trial_bin has no 0.0 entries.")

    mask = target_style[starts] if want_random else ~target_style[starts]
    candidates = starts[mask]

    if candidates.size == 0:
        kind = "random" if want_random else "center-out"
        raise RuntimeError(f"No {kind} trial starts found. Check target_style values at trial starts.")

    return int(rng.choice(candidates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position_path", type=str, required=True,
                    help="Path to finger position npy (float, 0.0-1.0), shape (N,)")
    ap.add_argument("--trial_bin_path", type=str, required=True,
                    help="Path to trial_bin npy (0.0 indicates trial start), shape (N,)")
    ap.add_argument("--target_style_path", type=str, required=True,
                    help="Path to target_style npy (bool), True=random, False=center-out, shape (N,)")

    ap.add_argument("--seed", type=int)
    ap.add_argument("--out_png", type=str, default="random_vs_centerout_trial_position.png")
    ap.add_argument("--use_trial_bin_as_time", action="store_true",
                    help="If set, x-axis uses trial_bin values; otherwise uses sample index t=0..T-1.")
    args = ap.parse_args()

    pos = np.asarray(npy_loader(args.position_path)).astype(float).reshape(-1)
    trial_bin = np.asarray(npy_loader(args.trial_bin_path)).reshape(-1)
    target_style = np.asarray(npy_loader(args.target_style_path)).astype(bool).reshape(-1)

    starts = np.flatnonzero(trial_bin == 0.0)
    lens = np.diff(np.r_[starts, len(trial_bin)])
    print("n_trials:", len(lens))
    print("min/median/mean/max:", lens.min(), np.median(lens), lens.mean(), lens.max())
    print("p1,p5,p10:", np.percentile(lens, [1,5,10]))

    if not (pos.shape[0] == trial_bin.shape[0] == target_style.shape[0]):
        raise ValueError(
            f"Length mismatch: position={pos.shape[0]}, trial_bin={trial_bin.shape[0]}, target_style={target_style.shape[0]}"
        )

    trial_id = build_trial_ids(trial_bin)
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    rng = np.random.RandomState(args.seed)

    starts = np.flatnonzero(trial_bin == 0.0)
    if starts.size == 0:
        raise RuntimeError("No trial starts found: trial_bin has no 0.0 entries.")

    # pick start indices using the helper
    start_r = choose_one_trial_start(trial_bin, target_style, want_random=True, rng=rng)
    start_c = choose_one_trial_start(trial_bin, target_style, want_random=False, rng=rng)

    def slice_trial(start_idx: int):
        nxt = starts[starts > start_idx]
        end_idx = int(nxt[0]) if nxt.size > 0 else len(trial_bin)
        return end_idx, pos[start_idx:end_idx], trial_bin[start_idx:end_idx]

    end_r, pos_r, tb_r = slice_trial(start_r)
    end_c, pos_c, tb_c = slice_trial(start_c)

    if args.use_trial_bin_as_time:
        t_r = tb_r
        t_c = tb_c
        xlabel = "t (trial_bin)"
    else:
        t_r = np.arange(pos_r.shape[0])
        t_c = np.arange(pos_c.shape[0])
        xlabel = "t (sample index within trial)"

    plt.figure()
    plt.plot(t_r, pos_r, label=f"random start={start_r} (len={end_r-start_r})")
    plt.plot(t_c, pos_c, label=f"center-out start={start_c} (len={end_c-start_c})")
    plt.xlabel(xlabel)
    plt.ylabel("finger position (0.0–1.0)")
    plt.ylim(-0.2, 1.2)
    plt.title("Finger position: one random trial vs one center-out trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.show()
    plt.close()

    print(f"Saved: {os.path.abspath(args.out_png)}")
    print(f"Picked trials (start idx): random={start_r}, center-out={start_c}")



if __name__ == "__main__":
    main()
