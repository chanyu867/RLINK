import numpy as np
import os, re, glob
import colorsys
from scipy.stats import spearmanr

def spearman_top_quantile(collected_or_seeds, model, delta_thr=0.1, min_n=20):

    rows = []
    for c in collected_or_seeds:
        rows.extend(c.get(model, []))

    dpos = np.concatenate([r["dpos"] for r in rows])
    dacc = np.concatenate([r["dacc"] for r in rows])

    mask = dpos >= delta_thr
    if mask.sum() < min_n:
        print(f"Not enough samples above delta_thr={delta_thr}: n={mask.sum()}")
        return {"model": model, "delta_thr": delta_thr, "rho": np.nan, "p": np.nan, "n": int(mask.sum())}

    rho, p = spearmanr(dpos[mask], dacc[mask])

    return {"model": model, "delta_thr": float(delta_thr), "rho": float(rho), "p": float(p), "n": int(mask.sum())}




def daywise_first_second_half_means(collected_seeds, model, min_blocks=2, ratio=0.5):
    rows = []
    for c in collected_seeds:
        rows.extend(c.get(model, []))
        
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

def stability_metrics_from_collected(collected_seeds, model, include_boundary=False, eps=0.0):

    rows = []
    for c in collected_seeds:
        rows.extend(c.get(model, []))

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

    var_all = float(np.var(x, ddof=1)) if n > 1 else np.nan

    neg = x[x < 0]
    n_neg = int(neg.size)
    var_down = float(np.var(neg, ddof=1)) if n_neg > 1 else (0.0 if n_neg == 1 else np.nan)

    downside_risk = float(np.mean(np.minimum(x, 0.0) ** 2))
    p_drop = float(np.mean(x < -eps))

    return {
        "model": model,
        "n": n,
        "variance": var_all,
        "downside_risk": downside_risk,
        "n_neg": n_neg,
        "var_down": var_down,
        "p_drop": p_drop,
        "eps": float(eps),
        "include_boundary": bool(include_boundary),
    }


def models_daily_mean_std(results_dir, finger, shift, models=None, gamma_mode=False, ylim=(0,1), ax=None):
    if ax is None:
        ax = plt.gca()
    if models is None:
        models = ["banditron", "banditronRP", "HRL", "AGREL"]

    pat = re.compile(r"results_(?P<finger>[^_]+)_(?P<model>[^_]+)_shift(?P<shift>-?\d+)_seed(?P<seed>\d+)_gamma(?P<gamma>.+)\.npy$")
    files = sorted(glob.glob(os.path.join(results_dir, "results_*.npy")))

    # model -> list of (path, day_dict)
    runs = {m: [] for m in models}

    for f in files:
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        gd = m.groupdict()
        if gd["finger"] != finger or int(gd["shift"]) != int(shift):
            continue
        model = gd["model"]
        if model not in runs:
            continue

        # gamma filtering
        if model == "HRL":
            if gamma_mode is True:   # gamma!=0 mode => exclude HRL
                continue
            # gamma==0 mode (None) => include HRL
        else:
            try:
                gval = float(gd["gamma"])  # "0.00" or "0.0000" -> 0.0
            except Exception:
                continue
            if gamma_mode is True and gval == 0.0:
                continue
            if gamma_mode is None and gval != 0.0:
                continue

        res = np.load(f, allow_pickle=True).item()
        d = res["performance"]["day_to_accs"]

        day_dict = {}
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, v in items:
            v = float(np.nanmean(np.asarray(v, dtype=float))) if isinstance(v, (list, tuple, np.ndarray)) else float(v)
            day_dict[int(k)] = v

        runs[model].append((f, day_dict))

    # print used files (required)
    mode_str = "gamma!=0" if gamma_mode is True else "gamma==0 (+HRL)"
    print(f"\nUsed files | finger={finger}, shift={shift}, gamma_mode={mode_str}")
    for model in models:
        paths = [p for p, _ in runs[model]]
        print(f"{model}: {len(paths)} files")
        for p in paths:
            print("  -", p)

    # plot
    ax.clear()
    any_plotted = False

    for model in models:
        day_dicts = [dd for _, dd in runs[model]]
        if not day_dicts:
            continue

        days = sorted({day for dd in day_dicts for day in dd.keys()})
        means, stds = [], []

        for day in days:
            vals = [dd[day] for dd in day_dicts if day in dd and np.isfinite(dd[day])]
            if len(vals) == 0:
                means.append(np.nan); stds.append(np.nan)
            elif len(vals) == 1:
                means.append(float(vals[0])); stds.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)))

        x = np.array(days)
        y = np.array(means, float)
        s = np.array(stds, float)
        ok = np.isfinite(y)

        ax.plot(x[ok], y[ok], label=f"{model} (n={len(day_dicts)} files)")
        ax.fill_between(x[ok], y[ok]-s[ok], y[ok]+s[ok], alpha=0.2)
        any_plotted = True

    ax.set_xlabel("Day")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.set_title(f"finger={finger}, shift={shift}, gamma_mode={mode_str}")
    if any_plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No matching files", ha="center", va="center", transform=ax.transAxes)


