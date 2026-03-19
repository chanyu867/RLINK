'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.1
''' 

# Importing the necessary libraries and modules.
from src.utils import *
from src.RL_decoders.algorithms import *
import warnings
import time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.activations import relu
from keras import Model, Input
from keras.losses import binary_crossentropy
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.models import load_model, Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.special as sp
import tomli
import argparse
import logging
from src.utils.npy_loader import npy_loader, add_block_id
from src.utils.day_date import dates_from_days, plot_blocks_and_avg_samples_per_block, heatmap_sbp_mean_by_electrode_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--update_W', action='store_true')
parser.add_argument('--toml_path', type = str, required=True)
parser.add_argument('--finger_ID', type = str, required=True)
parser.add_argument('--sbp_path', type = str)
parser.add_argument('--day_info_path', type = str)
parser.add_argument('--onset_path', type = str)
parser.add_argument('--label_path', type = str)
parser.add_argument('--label_mask', type = str, default=None)
parser.add_argument('--shift', type = int, required=True)
parser.add_argument('--mode', type = str, required=True)
parser.add_argument('--shift_mask_path', type = str, required=True)
parser.add_argument('--slicing_day', type = int, required=True)
parser.add_argument('--upper_slicing_day', type = int)
parser.add_argument('--target_type', type = str, default=None, help="Options: center-out, random")
parser.add_argument('--short_run', action='store_true', help="Whether to run a short version for testing")
parser.add_argument('--best_params_path', type=str, default=None, help="Path to load best params for quick testing")
parser.add_argument('--hpo_mode', type=str, default=None, help="HPO mode to determine which params to load (e.g., 'transition' or 'whole')")
parser.add_argument('--n_lags', type=int, default=0, help="Number of past steps to stack")
parser.add_argument('--lag_step', type=int, default=1, help="Distance between lags")
parser.add_argument('--lag_group', type=str, default="trial", help="Prevent crossing trial boundaries")

args = parser.parse_args()
scaler = StandardScaler()


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
        raise ValueError("CRITICAL: No samples left after lagging! All continuous blocks are too short for the sliding window.")
        
    X_lag = np.stack(X_out, axis=0).astype(np.float32)
    return X_lag, np.array(valid_idx, dtype=int)

def build_trial_ids(trial_bin_used):
    trial_starts = (trial_bin_used == 0.0)
    if len(trial_bin_used) > 0 and (trial_bin_used[0] != 0.0):
        trial_starts[0] = True
    return np.cumsum(trial_starts).astype(int) - 1


def accuracy_over_time(y_true, y_pred, day_info, threshold=0.9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    days = np.asarray(day_info).astype(int)

    xs, accs, day_to_accs = [], [], []

    for d in np.unique(days):  # unique days (sorted)
        idx = (days == d)
        acc = float(np.mean(y_true[idx] == y_pred[idx]))
        acc_01 = (y_true[idx] == y_pred[idx]).astype(int)  # 0/1 series
        
        #register the info
        xs.append(int(d))          # x-axis: day number (e.g., 1, 37, ...)
        accs.append(acc)           # y-axis: accuracy (%) within that day
        day_to_accs.append(acc_01)

    bad_days = days_below_threshold(xs, accs, threshold=threshold)

    return xs, accs, day_to_accs, bad_days

def days_below_threshold(xs, accs, threshold):
    xs = np.asarray(xs)
    accs = np.asarray(accs, dtype=float)
    return xs[accs <= threshold].tolist()

def load_model(case):
    cases = { #call all classes from algorithms.py
        'banditron': banditron,
        'banditronRP': banditronRP,
        'HRL': HRL,
        'AGREL': AGREL,
        # 'DQN': DQN,
        'DQN': DQN_ewc,
        'QLGBM': QLGBM
    }
    # Get the function associated with the case and call it
    if case in cases:
        # return cases[case]()
        return cases[case]
    else:
        return "Case not found"
    

def threshold_binary(x, thresh=0.85):
    x = np.asarray(x)
    return (x < thresh).astype(int)

# original: for file in files: #this time program is done only for single .npy file
def preprocess(model_type, sbp, cfg=None, output_dim=2, **kwargs):

    # 1) load data depending on given file format
    data_format = cfg.get("dataset").get("format")
    if data_format == "npy":
        X = sbp[:,:-1] #spike data
        y = sbp[:,-1] #label data
    else:
        logger.error(f"{model_type} is not supported in this implementation")
        exit()

    # 2) retrieve necessary parameters depending on given model type
    from src.RL_decoders.build_params import build_params
    params = build_params(model_type, cfg, input_dim=X.shape[1], output_dim=output_dim)
    setting = params["setting"]
    logger.info(f"{model_type} will run with {setting}")

    # 3) load model
    model = load_model(model_type)

    return X, y, model, setting

def run_decoder(X, y, model, day_info, **setting):

    #1. run model
    # setting["n_lag"] = 10
    # setting["lag_step"] = 1
    pred, when_explore, gamma = model(X, y, day_info, **setting)

    #2. convert data type
    y_true = (np.array(y)).astype(int)
    y_pred = (np.array(pred)).astype(int)

    logger.info(f"model run done for {model_type} with gamma{gamma}: %s %s %s %s", y_true.shape, y_pred.shape, y.shape, X.shape) #(7594816,) (7594816,) (7594816,) (7594816, 96)
    
    if when_explore is not None:
        when_explore = (np.array(when_explore)).astype(int)

    # return acc        
    return [y_true, y_pred], X, y, when_explore


def build_result_path(save_dir, finger_ID, model_type, shift, seed, gamma):
    return os.path.join(
        save_dir,
        f"results_{finger_ID}_{model_type}_shift{shift}_seed{seed}_gamma{gamma}.npy"
    )

#for batch-based DQN
def run_decoder_batch_dqn(X, y, model, day_info, setting, save_dir, seed, update_W, trial_ids, chunk_size=10000):
    """
    Safely processes massive datasets for DQN by splitting them into chunks, 
    saving/loading Keras weights between chunks, and clearing memory to prevent OOM.
    """
    logger.info("DQN detected: Running in batch mode to prevent memory exhaustion.")
    num_chunks = int(np.ceil(len(X) / chunk_size))
    
    all_y_true, all_y_pred, all_when_explore = [], [], []
    
    # Temporary file to hand weights between batches
    weights_path = os.path.join(save_dir, f"temp_dqn_weights_seed{seed}.h5")
    if os.path.exists(weights_path):
        os.remove(weights_path) # Clear old runs
        
    for c_idx in range(num_chunks):
        start = c_idx * chunk_size
        end = min((c_idx + 1) * chunk_size, len(X))
        
        # Slice data for this chunk
        X_chunk = X[start:end]
        y_chunk = y[start:end]
        day_chunk = day_info[start:end] if day_info is not None else None
        
        logger.info(f"\n{'='*40}\n--- DQN Batch {c_idx+1}/{num_chunks} (Samples {start} to {end}) ---\n{'='*40}")
        
        # Configure the setting to load/save weights correctly
        chunk_setting = setting.copy()
        if c_idx > 0:
            chunk_setting["weights_load_path"] = weights_path
        chunk_setting["weights_save_path"] = weights_path

        chunk_setting["trial_ids"] = trial_ids[start:end] if trial_ids is not None else None
        
        # Run the chunk
        if update_W:
            acc_c, _, _, when_explore_c = run_decoder(X_chunk, y_chunk, model, day_chunk, **chunk_setting)
        else:
            acc_c, _, _, when_explore_c = run_decoder(X_chunk, y_chunk, model, None, **chunk_setting)
        
        # Append results
        all_y_true.extend(acc_c[0])
        all_y_pred.extend(acc_c[1])
        all_when_explore.extend(when_explore_c)
        
        # Clears Keras memory graph to prevent soft-kill
        tf.keras.backend.clear_session()
        
    # Reconstruct the outputs to look exactly like a single huge run
    acc = [np.array(all_y_true), np.array(all_y_pred)]
    when_explore = np.array(all_when_explore)
    
    # Clean up the temporary weights file
    if os.path.exists(weights_path):
        os.remove(weights_path)
        
    return acc, X, y, when_explore

#    ------- main -------    #

#1. load setting
with open(args.toml_path, "rb") as f: # Open in binary mode ('rb')
    config = tomli.load(f)

#2. retrieve temporal info
day_info = npy_loader(path=args.day_info_path)
master_indices = np.arange(len(day_info))
# day_info = add_block_id(day_info) #-> when you apply mask for specific labels

#3. load stimulus onset info for all samples
stimulus_onset = npy_loader(path=args.onset_path)

#4. load SBP, labels, and trial info
sbp = npy_loader(path=args.sbp_path)
labels = npy_loader(path=args.label_path)
time_within_trial = npy_loader("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy")

#5. load target style info
target = npy_loader("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy")
target = np.asarray(target).astype(bool)
target = ~target #flipping

# task: check
trial_ids = build_trial_ids(time_within_trial)

# STEP 0: clip data for short_run testing
# if args.short_run:
#     clip_size = 380000
#     sbp = sbp[:clip_size]
#     labels = labels[:clip_size]
#     day_info = day_info[:clip_size]
#     stimulus_onset = stimulus_onset[:clip_size]
#     target = target[:clip_size]
#     time_within_trial = time_within_trial[:clip_size]
#     trial_ids = trial_ids[:clip_size]
#     logger.info(f"Short run mode: clipped data to {clip_size} samples, remained labels: {np.unique(labels)}, days: {np.unique(day_info)}")

#5. masking for specific days
if args.slicing_day is not None:
    if args.upper_slicing_day is not None:
        day_mask = (day_info > args.slicing_day) & (day_info <= args.upper_slicing_day)
    else:
        day_mask = day_info > args.slicing_day
    sbp = sbp[day_mask]
    labels = labels[day_mask]
    day_info = day_info[day_mask]
    target = target[day_mask]
    master_indices = master_indices[day_mask]
    trial_ids = trial_ids[day_mask]

# STEP A: introduce lags
logger.info(f"Applying lags: n_lags={args.n_lags}, step={args.lag_step}")
sbp, valid_idx = make_lagged_features(sbp, trial_ids, args.n_lags, args.lag_step)

# Sync all other arrays to the samples that survived the lag window
labels = labels[valid_idx]
day_info = day_info[valid_idx]
target = target[valid_idx]
master_indices = master_indices[valid_idx]
trial_ids = trial_ids[valid_idx]

#6. masking for shifting
if args.shift > 0:
    shift_mask = npy_loader(args.shift_mask_path).astype(bool)
    if args.slicing_day is not None:
        shift_mask = shift_mask[day_mask]
    else:
        shift_mask = shift_mask

    shift_mask = shift_mask[valid_idx] # Sync mask!
    sbp = sbp[shift_mask]
    labels = labels[shift_mask]
    day_info = day_info[shift_mask]
    target = target[shift_mask]
    trial_ids = trial_ids[shift_mask]
    master_indices = master_indices[shift_mask]

#8. masking for target type
if args.target_type == "center-out":
    sbp = sbp[target]
    labels = labels[target]
    day_info = day_info[target]
    master_indices = master_indices[target]
    trial_ids = trial_ids[target]
elif args.target_type == "random":
    sbp = sbp[~target]
    labels = labels[~target]
    day_info = day_info[~target]
    master_indices = master_indices[~target]
    trial_ids = trial_ids[~target]

#9. masking for deadzone labels
if args.label_mask is not None:
    allowed_labels = [int(x.strip()) for x in args.label_mask.split(",")]
    label_mask = np.isin(labels.astype(int), allowed_labels)
    
    sbp = sbp[label_mask]
    labels = labels[label_mask]
    print("unique labels after mask:", np.unique(labels))
    day_info = day_info[label_mask]
    master_indices = master_indices[label_mask]
    trial_ids = trial_ids[label_mask]

    # Remap labels so RL matrices don't crash
    unique_labels = np.sort(np.unique(labels))
    remapper = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
    labels = np.vectorize(remapper.get)(labels)

#10. assemble features ONLY at the very end
from src.utils.data_loader import assemble_features
labeled_sbp = assemble_features(neural_data=sbp, labels=labels)
logger.info("Final data length: labels=%d, day_info=%d, labeled_sbp=%d", len(labels), len(day_info), len(labeled_sbp))

#1. run models
models = ["DQN"]
# models = ["banditron", "banditronRP", "HRL", "AGREL", "DQN", "QLGBM"]
output_dim = np.unique(labels).size
logger.info("debug: output dim: %d %d", output_dim, args.shift)

if args.target_type is not None:
    logger.info(f"data is masked according to the given target types, {args.target_type}")

for i in range(0,3): #testing different seeds
    accs_list = []
    np.random.seed(i)

    logger.info(f"seed: {i}, shift: {args.shift}, labels: {args.label_path}")

    for model_type in models:

        #[Optional] If best_params_path is provided, load the best parameters
        if args.best_params_path is not None:
            import json
            params_path = os.path.join(args.best_params_path, f"best_params_{model_type}_{args.hpo_mode}.json")
            print(f"Loading best parameters for {model_type} from: {params_path}")
            with open(params_path, "r") as f:
                best_params_data = json.load(f)
                best_params = best_params_data.get("best_params", {})

                for key, value in best_params.items():
                    config[model_type][key] = value
                    # config["HRL"][key] = value

                logger.info(f"Loaded best parameters from {args.best_params_path}: {best_params}")

        #0. prepare for model
        X, y, model, setting = preprocess(model_type=model_type, sbp=labeled_sbp, cfg=config, output_dim=output_dim)
        
        #1. check whether the current setting is already done
        save_dir = f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/pred_results/{args.mode}/{output_dim}classes"
        os.makedirs(save_dir, exist_ok=True)
        expected_path = build_result_path(save_dir, args.finger_ID, model_type, shift=args.shift, seed=i, gamma=setting.get("gamma", None))
        logger.info("expected path: %s", expected_path)
        if os.path.exists(expected_path):
            logger.info("SKIP (already exists): %s", expected_path)
            continue
        else:
            logger.info("RUNNING for: %s", expected_path)

        #2. run model if the result is not existing
        if model_type == "DQN":
            # Use the new batching function for memory-heavy DQN
            acc, sbp, label, when_explore = run_decoder_batch_dqn(
                X, y, model, day_info, setting, save_dir, seed=i, update_W=args.update_W, trial_ids=trial_ids
            )
        else:
            # Standard run for lightweight models (Banditron, AGREL, etc.)
            if args.update_W:
                acc, sbp, label, when_explore = run_decoder(X, y, model, day_info, **setting)
            else:
                acc, sbp, label, when_explore = run_decoder(X, y, model, None, **setting)

        #3. calculate accuracy for normal setting(y_tilde)
        xs, accs, day_to_accs, bad_days = accuracy_over_time(y_true=acc[0], y_pred=acc[1], day_info=day_info)
        
        # 4. save all results
        results = {
            "meta": {
            "model_type": model_type,
            "output_dim": output_dim,
            "config": config,
            },

            "prediction": {
            "y_true": acc[0],
            "y_pred": acc[1],
            "when_explore": when_explore,
            },

            "performance": {
            "xs": xs,
            "accs": accs, #basically trust this
            "day_to_accs": day_to_accs,
            "bad_days": bad_days,
            },
            "master_indices": master_indices
        }
        
        np.save(expected_path, results, allow_pickle=True)

        accs_list.append(accs)

        logger.info(f"{model_type}: bad_days: {bad_days}")

    # #4. plot comparison among all models
    # fig_save_dir = f"/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/performance/{args.mode}/{output_dim}classes"
    # os.makedirs(fig_save_dir, exist_ok=True)
    # expected_fig_path = os.path.join(fig_save_dir, f"{args.finger_ID}_shift{args.shift}_seed{i}_gamma{setting.get('gamma', None)}.png")
    # title=f"task: {args.mode} - {args.finger_ID}_{output_dim}class_shift{args.shift}_seed{i}_gamma{setting.get('gamma', None)}"
    # if not expected_fig_path: #one model done mean all models also done
    #     from src.utils.performance_plot import plot_performance_compare
    #     #last gamma will be used for plot title
    #     plot_performance_compare(expected_fig_path, title, xs, accs_list, labels, ylim=(0, 1))

    logger.info("NEWLY calculated for seed: %d", i)
    logger.info("=" * 80)
# acc_DQN = analysis('e', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)
# acc_QLGBM = analysis('f', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)