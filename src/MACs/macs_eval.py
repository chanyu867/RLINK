import numpy as np

def calc_from_weight_npz(weight_npz_path: str):
    d = np.load(weight_npz_path, allow_pickle=True)
    state_obj = d["state_dict"]
    state = state_obj.item() if (isinstance(state_obj, np.ndarray) and state_obj.dtype == object) else state_obj

    W = state["fc.weight"]  # shape (K, D)
    b = state.get("fc.bias", None)

    K, D = W.shape
    macs = D * K
    params = D * K + (K if b is not None else 0)

    print("=== Perceptron complexity (batch=1) ===")
    print(f"in_dim D = {D}")
    print(f"out_dim K = {K}")
    print(f"MACs/forward = {macs}")
    print(f"Params       = {params}")

# example
calc_from_weight_npz("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/Perceptron/random_no_lag/weights/perceptron_random_b0.33_0.66_perceptron_seed0_Nclasses3_day1_weights.npz")
calc_from_weight_npz("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/Perceptron/random/weights/perceptron_random_b0.33_0.66_perceptron_seed0_Nclasses3_day1_weights.npz")
