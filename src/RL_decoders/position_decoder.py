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
# import FileBrowser
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
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

def accuracy_over_time(y_true, y_pred, block_size=10, step=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if step is None:
        step = block_size

    n = len(y_true)
    xs, accs = [], []
    block_id = 1

    for start in range(0, n - block_size + 1, step):
        end = start + block_size
        # acc = (y_true[start:end] == y_pred[start:end]).mean() * 100.0
        acc = float(np.mean(y_true[start:end] == y_pred[start:end])) * 100.0
        xs.append(block_id)
        accs.append(float(acc))
        block_id += 1

    return xs, accs


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

# def analysis(choice, dir, file, flag, **kwargs):
# original: for file in files: #this time program is done only for single .npy file
def analysis(model_type, file_path, cfg=None, **kwargs):

    # 1) load data depending on given file format
    data_format = cfg.get("dataset").get("format")
    if data_format == "npy":
        from src.utils.npy_loader import npy_loader
        data = npy_loader(file_path)
        X = data[:,:-1] #spike data
        y = data[:,-1] #label data
        #convert position into binary labels
        y = threshold_binary(y, thresh=0.85)
    elif data_format == "mat":
        from src.utils.mat_loader import load_feature_mat
        data = load_feature_mat(file_path) #handle v7.3 files
        print("data keys are: ", data.keys())
        feature_mat = data["feature_mat"] #error -> need to check the data
        X = feature_mat[:,:-1] #22 channels data, just spike counts
        y = feature_mat[:,-1]//90 #last part is the labels, 0, 90, 180 degrees -> ignore the stopping condition, for making it simple?
    else:
        logger.error(f"{model_type} is not supported in this implementation")
        exit()

    print("correct label y: ", y)

    # 2) retrieve necessary parameters depending on given model type
    from src.RL_decoders.build_params import build_params
    params = build_params(model_type, cfg)
    setting = params["setting"]
    print(f"{model_type} will run with {setting}")

    # 3) load model
    model = load_model(model_type)
    # pred = model(X[:1000], y[:1000], **setting) #small run
    pred = model(X, y, **setting)

    # 4) accuracy
    y_true = (np.array(y)).astype(int)
    y_pred = (np.array(pred)).astype(int)

    print("np.array(pred): ", np.array(pred).shape) #(1000,)

    # return acc        
    return [y_true, y_pred]

#How to run: python -m src.RL_decoders.position_decoder
model_type="banditron"
file_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/final_data/sbp_with_mrs_position_min0.15_max0.85.npy"
import tomli
with open("/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/config/config.toml", "rb") as f: # Open in binary mode ('rb')
    config = tomli.load(f)

# Getting the decoding accuracy for each algorithm
acc_Banditron = analysis(model_type=model_type, file_path=file_path, cfg=config)
# acc_BanditronRP = analysis('b', directory, files, 1, error=error, sparsity_rate=sparsity_rate, gamma=gamma)
# acc_HRL = analysis('c', directory, files, 1, muH=muH, muO=muO, num_nodes=num_nodes, error=error, sparsity_rate=sparsity_rate)
# acc_AGREL = analysis('d', directory, files, 1, gamma_AGREL=gamma_AGREL, alpha=alpha, beta=beta, num_nodes_AGREL=num_nodes_AGREL, error=error, sparsity_rate=sparsity_rate)
# acc_DQN = analysis('e', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)
# acc_QLGBM = analysis('f', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)

# Plotting the Decoding accuracy of all the Algorithms
print("shape of outputs: ", acc_Banditron[0].shape)
xs, accs = accuracy_over_time(y_true=acc_Banditron[0], y_pred=acc_Banditron[1], block_size=1000)
plot_performance_time(xs, accs, title=f"{model_type} | block_size=1000", ylim=(20, 100))