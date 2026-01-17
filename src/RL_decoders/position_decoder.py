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
parser.add_argument('--toml_path', type = str, required=True)
parser.add_argument('--finger_ID', type = str, required=True)
parser.add_argument('--sbp_path', type = str)
parser.add_argument('--day_info_path', type = str)
parser.add_argument('--onset_path', type = str)
parser.add_argument('--label_path', type = str)
parser.add_argument('--label_mask', type = list, default=None)
parser.add_argument('--shift', type = int, required=True)
parser.add_argument('--mode', type = str, required=True)
parser.add_argument('--shift_mask_path', type = str, required=True)
parser.add_argument('--slicing_day', type = int, required=True)

args = parser.parse_args()
scaler = StandardScaler()

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
        'DQN': DQN,
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
    params = build_params(model_type, cfg, output_dim=output_dim)
    setting = params["setting"]
    logger.info(f"{model_type} will run with {setting}")

    # 3) load model
    model = load_model(model_type)

    return X, y, model, setting

def run_decoder(X, y, model, day_info, **setting):

    #1. run model
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

#    ------- main -------    #

#1. load setting
with open(args.toml_path, "rb") as f: # Open in binary mode ('rb')
    config = tomli.load(f)

#2. retrieve temporal info
day_info = npy_loader(path=args.day_info_path)
# day_info = add_block_id(day_info) #-> when you apply mask for specific labels

#3. load stimulus onset info for all samples
stimulus_onset = npy_loader(path=args.onset_path)

#4. load SBP data
sbp = npy_loader(path=args.sbp_path)

#5. load label data (assume that user already defined the label data)
from src.utils.data_loader import assemble_features, make_mask
labels = npy_loader(path=args.label_path)
labeled_sbp = assemble_features(neural_data=sbp, labels=labels) # add class as the last dimension
del sbp # be merciful to RAM, save it from keeping useless 5 GB

#6. masking for shifting
if args.shift > 0:
    time_within_trial = npy_loader("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy")
    shift_mask = npy_loader(args.shift_mask_path).astype(bool)
    count_1 = time_within_trial[time_within_trial==0.0] #count how many trials are there
    labeled_sbp = labeled_sbp[shift_mask]
    labels = labels[shift_mask]
    day_info = day_info[shift_mask]
    logger.info("masking data for shift: %s %s %d", time_within_trial.shape, labeled_sbp.shape, time_within_trial.shape[0]-(len(count_1)*args.shift)) #(7711816,) (7360816,)
else:
    shift_mask = None

#7. masking for specific days
if args.slicing_day is not None:
    day_mask = day_info>args.slicing_day
    labels = labels[day_mask]
    day_info = day_info[day_mask]
    labeled_sbp = labeled_sbp[day_mask]

#8. masking for labels
if args.label_mask is not None:
    label_mask = make_mask(labels, args.label_mask) # keep [0., 0.15] and [0.85, 1.]
    labeled_sbp = labeled_sbp[label_mask] # apply mask
    stimulus_onset = stimulus_onset[label_mask]

#9. validate data consistency
if not (len(labels) == len(day_info) == len(labeled_sbp)):
    logger.error(f"Data length mismatch: labels={len(labels)}, day_info={len(day_info)}, labeled_sbp={len(labeled_sbp)}")
    exit()
else:
    logger.info("final data length: %d %d %d", len(labels), len(day_info), len(labeled_sbp))

#10. run models
models = ["banditron", "banditronRP", "HRL", "AGREL"]
# models = ["banditronRP"]
output_dim = np.unique(labels).size
logger.info("debug: output dim: %d %d", output_dim, args.shift)

for i in range(10): #testing different seeds
    accs_list = []
    np.random.seed(i)

    logger.info(f"seed: {i}, shift: {args.shift}, labels: {args.label_path}")

    for model_type in models:

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
        # acc, sbp, label, when_explore = run_decoder(X, y, model, None, **setting)
        acc, sbp, label, when_explore = run_decoder(X, y, model, day_info, **setting)

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
            }
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