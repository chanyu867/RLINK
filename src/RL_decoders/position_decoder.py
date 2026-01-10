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
parser.add_argument('--sbp_path', type = str)
parser.add_argument('--time_info_path', type = str)

args = parser.parse_args()
scaler = StandardScaler()

def plot_performance_time(xs, accs, title=None, ylim=(0, 100), show_values=False):
    plt.figure(figsize=(10, 5))
    plt.plot(xs, accs, marker="o")
    plt.grid(True)
    plt.ylim(ylim)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Time (block index)")

    if show_values:
        for x, y in zip(xs, accs):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def accuracy_over_time(y_true, y_pred, temp_info, threshold=90):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    temp_info = np.asarray(temp_info)

    days = temp_info[:, 1].astype(int)

    xs, accs = [], []

    for d in np.unique(days):  # unique days (sorted)
        idx = (days == d)
        acc = float(np.mean(y_true[idx] == y_pred[idx])) * 100.0
        xs.append(int(d))          # x-axis: day number (e.g., 1, 37, ...)
        accs.append(acc)           # y-axis: accuracy (%) within that day

    bad_days = days_below_threshold(xs, accs, threshold=threshold)

    return xs, accs, bad_days


def plot_samples_per_day(temp_info, sort=True, title="Samples per day"):
    temp_info = np.asarray(temp_info)
    days = temp_info[:, 1].astype(int)

    uniq, counts = np.unique(days, return_counts=True)

    if sort:
        order = np.argsort(uniq)
        uniq, counts = uniq[order], counts[order]

    plt.figure()
    plt.bar(uniq, counts)
    plt.xlabel("Day")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_performance_compare(xs, accs_list, labels, title=None, ylim=(0, 100)):
    plt.figure(figsize=(10, 5))
    for accs, lab in zip(accs_list, labels):
        plt.plot(xs, accs, marker="o", label=lab)

    plt.grid(True)
    plt.ylim(ylim)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Time (block index)")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
def analysis(model_type, file_path, cfg=None, **kwargs):

    # 1) load data depending on given file format
    data_format = cfg.get("dataset").get("format")
    if data_format == "npy":
        data = npy_loader(file_path)
        X = data[:,:-1] #spike data
        y = data[:,-1] #label data
        #convert position into binary labels
        y = threshold_binary(y, thresh=0.85)
    elif data_format == "mat":
        from src.utils.mat_loader import load_feature_mat
        data = load_feature_mat(file_path) #handle v7.3 files
        logger.info(f"data keys are: {data.keys()}")
        f"{model_type}: bad_dates: {bad_dates}, bad_days: {bad_days}"
        feature_mat = data["feature_mat"] #error -> need to check the data
        X = feature_mat[:,:-1] #22 channels data, just spike counts
        y = feature_mat[:,-1]//90 #last part is the labels, 0, 90, 180 degrees -> ignore the stopping condition, for making it simple?
    else:
        logger.error(f"{model_type} is not supported in this implementation")
        exit()

    # 2) retrieve necessary parameters depending on given model type
    from src.RL_decoders.build_params import build_params
    params = build_params(model_type, cfg)
    setting = params["setting"]
    logger.info(f"{model_type} will run with {setting}")

    # 3) load model
    model = load_model(model_type)
    # pred = model(X[:1000], y[:1000], **setting) #small run
    pred = model(X, y, **setting)

    # 4) accuracy
    y_true = (np.array(y)).astype(int)
    y_pred = (np.array(pred)).astype(int)

    # return acc        
    return [y_true, y_pred]

#1. load setting
with open("src/RL_decoders/config/config.toml", "rb") as f: # Open in binary mode ('rb')
    config = tomli.load(f)

#2. retrieve temporal info
temp_info = npy_loader(path = args.time_info_path)
temp_info = add_block_id(temp_info)

#3. run models
accs_list = []
models = ["banditron", "banditronRP", "HRL", "AGREL"]
for model_type in models:
    acc = analysis(model_type=model_type, file_path=args.sbp_path, cfg=config)
    xs, accs, bad_days = accuracy_over_time(y_true=acc[0], y_pred=acc[1], temp_info=temp_info)
    bad_dates = dates_from_days(bad_days, temp_info)
    data = npy_loader(args.sbp_path)
    accs_list.append(accs)

    logger.info(f"{model_type}: bad_dates: {bad_dates}, bad_days: {bad_days}")

#4. plot comparison among all models
plot_performance_compare(xs, accs_list, models, title="Model comparison", ylim=(20, 100))

# plot_performance_time(xs, accs, title=f"{model_type}", ylim=(20, 100))
# plot_samples_per_day(temp_info)
# plot_blocks_and_avg_samples_per_block(temp_info)
# heatmap_sbp_mean_by_electrode_day(data[:,:-1], temp_info, day_col=1)

# acc_DQN = analysis('e', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)
# acc_QLGBM = analysis('f', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)

#idx:
# bad_dates:  [20210309, 20210312, 20210313, 20230113, 20230127]
# bad_days:  [408, 411, 412, 1083, 1097]
#mrs:
# bad_dates:  [20210309, 20210312, 20210313, 20230113, 20230127]
# bad_days:  [408, 411, 412, 1083, 1097]