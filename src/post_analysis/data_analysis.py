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
    std_all = float(np.std(x, ddof=1)) if n > 1 else np.nan

    neg = x[x < 0]
    n_neg = int(neg.size)
    var_down = float(np.var(neg, ddof=1)) if n_neg > 1 else (0.0 if n_neg == 1 else np.nan)

    downside_risk = float(np.mean(np.minimum(x, 0.0) ** 2))
    p_drop = float(np.mean(x < -eps))

    return {
        "model": model,
        "n": n,
        "variance": var_all,
        "standard_deviation": std_all,
        "downside_risk": downside_risk,
        "n_neg": n_neg,
        "var_down": var_down,
        "p_drop": p_drop,
        "eps": float(eps),
        "include_boundary": bool(include_boundary),
    }


