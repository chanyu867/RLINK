import argparse
import json
import logging
import os
import numpy as np
import optuna
import tomli
from sklearn.metrics import balanced_accuracy_score
from src.utils.npy_loader import npy_loader
from src.utils.data_loader import assemble_features
from src.RL_decoders.algorithms import banditron, banditronRP, HRL, AGREL, DQN, QLGBM
import gc
import tensorflow as tf
# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_transition_mask(y_true, prev_samples, post_samples):
    """Creates a boolean mask covering windows around class transitions."""
    T = len(y_true)
    changes = np.where(y_true[1:] != y_true[:-1])[0] + 1
    mask = np.zeros(T, dtype=bool)
    
    for c in changes:
        start = max(0, c - prev_samples)
        end = min(T, c + post_samples)
        mask[start:end] = True
    return mask

def make_lagged_features(X, group, n_lags, lag_step=1):
    if n_lags <= 0:
        return X.astype(np.float32), np.arange(len(X))
        
    N, D = X.shape
    X_out, valid_idx = [], []
    
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        start = n_lags * lag_step
        if idx.size <= start:
            continue
            
        for j in range(start, idx.size):
            t = idx[j]
            feats = [X[t]]
            for k in range(1, n_lags + 1):
                feats.append(X[idx[j - k * lag_step]])
            X_out.append(np.concatenate(feats, axis=0))
            valid_idx.append(t)
            
    if len(X_out) == 0:
        raise ValueError("CRITICAL: No samples left after lagging! All continuous blocks are too short.")
        
    X_lag = np.stack(X_out, axis=0).astype(np.float32)
    return X_lag, np.array(valid_idx, dtype=int)

def build_trial_ids(trial_bin_used):
    trial_starts = (trial_bin_used == 0.0)
    if len(trial_bin_used) > 0 and (trial_bin_used[0] != 0.0):
        trial_starts[0] = True
    return np.cumsum(trial_starts).astype(int) - 1

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================
def objective_factory(args, X, y, day_info, trial_ids, config, model_func):
    def objective(trial):
        # 1. Suggest Hyperparameters based on model type
        params = {
            "error": config["feedback"]["error"],
            "sparsity_rate": config["feedback"]["sparse_rate"],
        }

        if args.model_type == "banditron":
            params["gamma"] = trial.suggest_float("gamma", 0.0, 0.5)
            params["k"] = np.unique(y).size
        
        elif args.model_type == "banditronRP":
            params["gamma"] = trial.suggest_float("gamma", 0.0, 0.5)
            params["k"] = trial.suggest_categorical("k_rp", [64, 128, 256, 512])
        
        elif args.model_type == "HRL":
            params["muH"] = trial.suggest_float("muH", 1e-4, 0.1, log=True)
            params["muO"] = trial.suggest_float("muO", 1e-4, 0.1, log=True)
            n_layers = trial.suggest_int("depth", 1, 10)
            hidden_layers = []
            for i in range(n_layers):
                hidden_layers.append(trial.suggest_int(f"width_l{i}", 32, 512))
            
            params["num_nodes"] = [X.shape[1]] + hidden_layers + [np.unique(y).size]
            
        elif args.model_type == "AGREL":
            params["gamma"] = trial.suggest_float("gamma", 0.0, 0.2)
            params["alpha"] = trial.suggest_float("alpha", 1e-3, 0.5, log=True)
            params["beta"] = trial.suggest_float("beta", 1e-3, 0.5, log=True)
            n_layers = trial.suggest_int("depth", 1, 10)
            hidden_layers = []
            for i in range(n_layers):
                hidden_layers.append(trial.suggest_int(f"width_l{i}", 32, 512))
            
            params["num_nodes"] = [X.shape[1]] + hidden_layers + [np.unique(y).size]

        elif args.model_type == "DQN":
            params["epsilon"] = trial.suggest_float("epsilon", 0.0, 0.5)
            params["gamma"] = trial.suggest_float("gamma", 0.0, 0.99)
            
        # elif args.model_type == "QLGBM":
        #     params["epsilon"] = trial.suggest_float("epsilon", 0.0, 0.5)
        #     params["gamma"] = trial.suggest_float("gamma", 0.0, 0.99)

        # 2. Run the RL model
        result = model_func(X, y, day_info, **params)
        
        # Safely extract predictions (QLGBM returns 1 array, others return 3 variables)
        if isinstance(result, tuple):
            y_pred_raw = result[0]
        else:
            y_pred_raw = result
            
        y_pred = np.asarray(y_pred_raw).astype(int)
        y_true = np.asarray(y).astype(int)

        # 3. Calculate Metric based on mode
        if args.metric_type == "whole":
            score = balanced_accuracy_score(y_true, y_pred)
            
        elif args.metric_type == "transition":
            mask = get_transition_mask(y_true, args.prev_samples, args.post_samples)
            if not np.any(mask):
                return 0.0
            score = balanced_accuracy_score(y_true[mask], y_pred[mask])
            
        elif args.metric_type == "all_class_trials":
            unique_classes = np.unique(y_true)
            num_req_classes = len(unique_classes)
            valid_trial_mask = np.zeros(len(y_true), dtype=bool)
            
            for tid in np.unique(trial_ids):
                idx = (trial_ids == tid)
                if len(np.unique(y_true[idx])) == num_req_classes:
                    valid_trial_mask[idx] = True
                    
            if not np.any(valid_trial_mask):
                logger.warning("No trials with all classes found in this data slice!")
                return 0.0
                
            score = balanced_accuracy_score(y_true[valid_trial_mask], y_pred[valid_trial_mask])

        if args.model_type == "DQN":
            tf.keras.backend.clear_session()
            gc.collect()

        return score
    return objective

def main():
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument('--sbp_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--day_info_path', type=str, required=True)
    parser.add_argument('--toml_path', type=str, required=True)
    parser.add_argument('--trial_bin_path', type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy")
    parser.add_argument('--target_style_path', type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy")
    
    # Matching Data Filters
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--shift_mask_path', type=str, default=None)
    parser.add_argument('--slicing_day', type=int, default=None)
    parser.add_argument('--target_type', type=str, default="random", help="Options: center-out, random")
    parser.add_argument('--label_mask', type=str, default="0,2", help="E.g., '0,2' to drop class 1")
    parser.add_argument('--n_lags', type=int, default=1)
    parser.add_argument('--lag_step', type=int, default=1)
    
    # HPO Settings
    parser.add_argument('--model_type', type=str, choices=['banditron', 'banditronRP', 'HRL', 'AGREL', 'DQN', 'QLGBM'], required=True)
    parser.add_argument('--metric_type', type=str, choices=['whole', 'transition', 'all_class_trials'], default='all_class_trials')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=None, help="Max samples to run HPO on (for speed)")
    parser.add_argument('--prev_samples', type=int, default=10)
    parser.add_argument('--post_samples', type=int, default=50)
    parser.add_argument('--update_W', action='store_true', help="Whether to pass day_info to update W matrix")
    
    args = parser.parse_args()

    with open(args.toml_path, "rb") as f:
        config = tomli.load(f)

    logger.info("Loading raw data...")
    sbp = npy_loader(args.sbp_path)
    labels = npy_loader(args.label_path)
    day_info = npy_loader(args.day_info_path)
    time_within_trial = npy_loader(args.trial_bin_path)
    target = npy_loader(args.target_style_path)
    target = ~np.asarray(target).astype(bool) # flipping

    # # ========================================================
    # day_changes = np.where(day_info[1:] != day_info[:-1])[0] + 1
    # day_starts = np.insert(day_changes, 0, 0)
    # unique_days = np.unique(day_info)
    
    # logger.info("--- RAW DATA: DAY START INDICES ---")
    # for day, start_idx in zip(unique_days, day_starts):
    #     logger.info(f"Day {int(day)} starts at sample index: {start_idx}")
    # logger.info("-" * 35)
    # # ========================================================

    if args.max_samples is not None:
        # print(f"full length: sbp={len(sbp)}, labels={len(labels)}, day_info={len(day_info)}, time_within_trial={len(time_within_trial)}, target={len(target)}")
        sbp = sbp[:args.max_samples]
        labels = labels[:args.max_samples]
        day_info = day_info[:args.max_samples]
        time_within_trial = time_within_trial[:args.max_samples]
        target = target[:args.max_samples]
        logger.info(f"Short run mode: clipped data to {args.max_samples} samples, remained labels: {np.unique(labels)}, days: {np.unique(day_info)}")


    trial_ids = build_trial_ids(time_within_trial)

    # ========================================================
    # STEP A: LAG FIRST
    # ========================================================
    logger.info(f"Applying lags: n_lags={args.n_lags}, step={args.lag_step}")
    sbp, valid_idx = make_lagged_features(sbp, trial_ids, args.n_lags, args.lag_step)

    labels = labels[valid_idx]
    day_info = day_info[valid_idx]
    target = target[valid_idx]
    trial_ids = trial_ids[valid_idx]

    # ========================================================
    # STEP B: APPLY MASKS 
    # ========================================================
    if args.shift > 0 and args.shift_mask_path:
        shift_mask = npy_loader(args.shift_mask_path).astype(bool)[valid_idx]
        sbp, labels, day_info, target, trial_ids = sbp[shift_mask], labels[shift_mask], day_info[shift_mask], target[shift_mask], trial_ids[shift_mask]

    #slice the data for HPO
    if args.slicing_day is not None:
        from src.baselines.mlp_train import compute_slicing_day_value
        slicing_day_value = compute_slicing_day_value(day_info, args.slicing_day)
        logger.info(f"Slicing data up to day {args.slicing_day} (day value <= {slicing_day_value:.2f}) for HPO")
        day_mask = day_info <= args.slicing_day
        print(f"{np.sum(day_mask)} of data is available for HPO")
        sbp, labels, day_info, target, trial_ids = sbp[day_mask], labels[day_mask], day_info[day_mask], target[day_mask], trial_ids[day_mask]

    if args.target_type == "center-out":
        sbp, labels, day_info, trial_ids = sbp[target], labels[target], day_info[target], trial_ids[target]
    elif args.target_type == "random":
        sbp, labels, day_info, trial_ids = sbp[~target], labels[~target], day_info[~target], trial_ids[~target]

    if args.label_mask is not None:
        if isinstance(args.label_mask, str):
            allowed_labels = [int(x.strip()) for x in args.label_mask.split(",") if x.strip()]
        else:
            allowed_labels = [int(x) for x in "".join(args.label_mask).split(",") if x.strip()]
            
        label_mask = np.isin(labels.astype(int), allowed_labels)
        
        # Apply the mask
        sbp, labels, day_info, trial_ids = sbp[label_mask], labels[label_mask], day_info[label_mask], trial_ids[label_mask]
        
        # SAFETY CHECK: Ensure we didn't filter out ALL samples
        if len(labels) == 0:
            raise ValueError(f"CRITICAL: No samples remaining after applying label_mask {allowed_labels}. ")

        unique_labels = np.sort(np.unique(labels))
        remapper = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
        labels = np.vectorize(remapper.get)(labels)

    logger.info("=" * 40)
    logger.info("FINAL HPO DATA SUMMARY:")
    logger.info(f"Total samples being used: {len(sbp)}")
    logger.info(f"Unique days in slice:     {np.unique(day_info).size}")
    logger.info(f"Number of unique trials:  {np.unique(trial_ids).size}")
    logger.info(f"Classes being optimized:  {np.unique(labels)}")
    logger.info("=" * 40)

    logger.info("Breakdown by Class:")
    unique_classes = np.unique(labels)
    for c in unique_classes:
        c_mask = (labels == c)
        n_samples = np.sum(c_mask)  # Total number of datapoints for this class
        n_trials = np.unique(trial_ids[c_mask]).size # Total unique trials containing this class
        logger.info(f"  -> Class {int(c)}: {n_samples} samples (across {n_trials} trials)")
    logger.info("=" * 40)
    
    day_changes = np.where(day_info[1:] != day_info[:-1])[0] + 1
    day_starts = np.insert(day_changes, 0, 0)
    unique_days = np.unique(day_info)

    logger.info(f"--- HPO DATA: DAY START INDICES ---") # Print first 100 for sanity check
    for day, start_idx in zip(unique_days, day_starts):
        logger.info(f"Day {int(day):02d} starts at sample index: {start_idx}")
    logger.info("-" * 35)

    labeled_sbp = assemble_features(neural_data=sbp, labels=labels)
    X = labeled_sbp[:, :-1]
    y = labeled_sbp[:, -1].astype(int)
    day_info_pass = day_info if args.update_W else None

    # Setup Optuna
    model_map = {
        'banditron': banditron, 
        'banditronRP': banditronRP, 
        'HRL': HRL, 
        'AGREL': AGREL, 
        'DQN': DQN, 
        # 'QLGBM': QLGBM
    }
    
    study = optuna.create_study(direction="maximize")
    objective = objective_factory(args, X, y, day_info_pass, trial_ids, config, model_map[args.model_type])
    
    logger.info(f"Starting HPO for {args.model_type} optimizing for {args.metric_type} accuracy...")
    study.optimize(objective, n_trials=args.n_trials)

    # Results
    print("\n" + "="*30)
    print("HPO COMPLETE")
    print(f"Best Score ({args.metric_type}): {study.best_value:.4f}")
    print("Best Params:", json.dumps(study.best_params, indent=2))
    print("="*30)

    save_path = f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/HPO/best_params_{args.model_type}_{args.metric_type}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f, indent=2)
    logger.info(f"Saved best parameters to {save_path}")

if __name__ == "__main__":
    main()