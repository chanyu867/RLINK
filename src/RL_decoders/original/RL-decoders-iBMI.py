'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.1
''' 

# Importing the necessary libraries and modules.
from utils import *
from algorithms import *
import warnings
import time
warnings.filterwarnings('ignore')
import numpy as np
from scipy.io import loadmat
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
from keras.models import load_model, Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.special as sp
# import FileBrowser
import argparse

parser = argparse.ArgumentParser()
scaler = StandardScaler()

# Adding arguments
parser.add_argument('--dir', type = str, help = "directory of dataset")
parser.add_argument('--expt', type = str, help = "name of the particular experiment")
parser.add_argument('--spars', type = int, default = 0, help = "Sparsity Rate")
parser.add_argument('--error', type = int, default = 0, help = "Error Rate")
parser.add_argument('--gamma', type = int, default = 0.0001, help = "Gamma value")
parser.add_argument('--muH', type = int, default = 0.01, help = "Hidden Layer Learning rate")
parser.add_argument('--muO', type = int, default = 0.01, help = "Hidden Layer Learning rate")
parser.add_argument('--num_nodes', type = list, default=[22,3], help = "Number of modes")

args = parser.parse_args()

# Defining the Global Variables --> Directory, error and sparsity error.
absolute_path = os.path.dirname(os.path.abspath('__file__'))
relative_path = args.dir 
expt = args.expt # subject to change depending on the experiment you are trying to execute
directory = os.path.join(absolute_path,os.path.join(relative_path, expt))

# files = FileBrowser.uigetfile() 
import glob, os
files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(directory, "*.mat")))]
print("used file path: ", directory, files)
#used file path:  /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/datasets/original/monkey_1_set_1 []
error = args.error # Defining the error in Feedback
sparsity_rate = args.spars # Sparsity in the Feedback signals
gamma = args.gamma
muH = args.muH
muO = args.muO
num_nodes = args.num_nodes
alpha = 0.1
beta = 0.1
gamma_AGREL = 0.02
num_nodes_AGREL = [1000,4]
epsilon = 0.01 # Exploration rate
gamma_DQN = 0.1 # Discount Factor

def match_case(case):
    cases = { #call all classes from algorithms.py
        'a': banditron,
        'b': banditronRP,
        'c': HRL,
        'd': AGREL,
        'e': DQN,
        'f': QLGBM
    }
    # Get the function associated with the case and call it
    if case in cases:
        # return cases[case]()
        return cases[case]
    else:
        return "Case not found"
    

def load_feature_mat(mat_path):
    try:
        data = loadmat(mat_path)
        return data
        raise KeyError(f"'feature_mat' not found in {mat_path}. keys={list(data.keys())}")

    except NotImplementedError:
        import h5py
        with h5py.File(mat_path, "r") as f:
            keys = list(f.keys())
            if "feature_mat" in f:
                ds = f["feature_mat"]
            else:
                raise KeyError(f"'feature_mat' not found in {mat_path}. keys={keys}")

            arr = ds[()]
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            return arr


def analysis(choice, dir, files, flag, **kwargs):
    acc = []
    error_acc = []
    interval = []
    
    for file in files:
        print("os.path.join(dir,file): ", dir, file)
        # data = loadmat(os.path.join(dir,file))
        data = load_feature_mat(os.path.join(dir,file)) #handle v7.3 files
        print("data keys are: ", data.keys())
        feature_mat = data["feature_mat"] #error -> need to check the data
        X = feature_mat[:,:-1] #22 channels data, just spike counts
        y = feature_mat[:,-1]//90 #last part is the labels, 0, 90, 180 degrees -> ignore the stopping condition, for making it simple?
        print("y: ", y)
        if choice == 'a':
        #   for error, sparsity_rate, gamma in kwargs.items(): #dict_items([('error', 0), ('sparsity_rate', 0), ('gamma', 0.0001)])
            error, sparsity_rate, gamma = kwargs['error'], kwargs['sparsity_rate'], kwargs['gamma']
            model = match_case(choice)
            pred = model(X, y, error, sparsity_rate, gamma=gamma)
        elif choice == 'b':
            # for error, sparsity_rate, gamma in kwargs.items():
            error, sparsity_rate, gamma = kwargs['error'], kwargs['sparsity_rate'], kwargs['gamma']
            model = match_case(choice)
            pred = model(X, y, 128, error, sparsity_rate, gamma=gamma)
        elif choice == 'c':
            # for error, sparsity_rate, muH, muO, num_modes in kwargs.items():
            error, muH, muO, num_nodes, sparsity_rate = kwargs['error'], kwargs['muH'], kwargs['muO'], kwargs['num_nodes'], kwargs['sparsity_rate']
            model = match_case(choice)
            # num_nodes = [int(x) for x in list(num_nodes)] #convert given number as int values
            num_nodes = [X.shape[1], 3] #-> dummy
            print("num_nodes: ", num_nodes)
            pred = model(X, y, muH, muO, num_nodes=num_nodes, error=error, sparse_rate=sparsity_rate)
        elif choice == 'd':
            # for gamma, alpha, beta, num_nodes, error, sparsity_rate in kwargs.items():
            gamma, alpha, beta, num_nodes, error, sparsity_rate = kwargs['gamma_AGREL'], kwargs['alpha'], kwargs['beta'], kwargs['num_nodes_AGREL'], kwargs['error'], kwargs['sparsity_rate']
            model = match_case(choice)
            num_nodes = [X.shape[1], 3, 5] #-> dummy
            pred = model(X, y, gamma, alpha, beta, num_nodes=num_nodes, error=error, sparse_rate=sparsity_rate)
        elif choice == 'e':
            target = to_categorical(y,4)
            # for epsilon,gamma,error,sparsity_rate in kwargs.items():
            epsilon, gamma, error, sparsity_rate = kwargs['epsilon'], kwargs['gamma_DQN'], kwargs['error'], kwargs['sparsity_rate']
            model = match_case(choice)
            num_nodes = [X.shape[1], 3] #-> dummy
            pred = model(X,target,epsilon,gamma,error,sparsity_rate)
        elif choice == 'f':
            target = to_categorical(y,4)
            # for epsilon,gamma,error,sparsity_rate in kwargs.items():
            model = match_case(choice)
            num_nodes = [X.shape[1], 3]
            epsilon, gamma, error,sparsity_rate = kwargs['epsilon'], kwargs['gamma_DQN'], kwargs['error'], kwargs['sparsity_rate']
            pred = model(X,target,epsilon,gamma,error,sparsity_rate)

        results = np.vstack([y*90,np.array(pred)*90]).T
        results_df = pd.DataFrame(results,columns=["True","Pred"])
        error_intervals = 1/(np.sum(results_df.loc[:,"True"] == results_df.loc[:,"Pred"])/results_df.shape[0]*100)
        error_acc.append(error_intervals)
        interval.append(2.58*np.sqrt(error_intervals*(1-error_intervals))/results_df.shape[0])
        acc.append((np.sum(results_df.loc[:,"True"] == results_df.loc[:,"Pred"])/results_df.shape[0])*100)

    if flag == 1:
        print("length of files: ", len(files), len(acc))
        plt.figure(figsize=(10,5))
        plt.plot(range(1,len(files)+1),acc,'b-o')
        plt.grid()
        plt.ylim((20,120))
        plt.ylabel('Accuracy (in %)')
        plt.xlabel('Sessions')
        plt.xticks(range(1,len(files)+1), labels=['session_'+str(i) for i in range(1,len(files)+1)]) #each sessions name for x values
        for x,y in zip(range(1,len(files)+1),acc):
            label = "{:.2f}".format(y)
            plt.annotate(label, # this is the text
                (x,y), # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(-10,-20)) # distance from text to points (x,y)
        
        plt.title(directory.split('/')[-1]+str(model))
        plt.show()
        
    return acc        

# Getting the decoding accuracy for each algorithm
# acc_Banditron = analysis('a', dir, files, 1, error, sparsity_rate, gamma=gamma)
acc_Banditron = analysis('a', directory, files, 1, error=error, sparsity_rate=sparsity_rate, gamma=gamma)
acc_BanditronRP = analysis('b', directory, files, 1, error=error, sparsity_rate=sparsity_rate, gamma=gamma)
acc_HRL = analysis('c', directory, files, 1, muH=muH, muO=muO, num_nodes=num_nodes, error=error, sparsity_rate=sparsity_rate)
acc_AGREL = analysis('d', directory, files, 1, gamma_AGREL=gamma_AGREL, alpha=alpha, beta=beta, num_nodes_AGREL=num_nodes_AGREL, error=error, sparsity_rate=sparsity_rate)
# acc_DQN = analysis('e', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)
# acc_QLGBM = analysis('f', directory, files, 1, epsilon=epsilon, gamma_DQN=gamma_DQN, error=error, sparsity_rate=sparsity_rate)

# Plotting the Decoding accuracy of all the Algorithms
plt.figure(figsize=(10,5))

plt.plot(range(1,len(files)+1),acc_Banditron,'b--o')
plt.plot(range(1,len(files)+1),acc_BanditronRP,'r--o')
plt.plot(range(1,len(files)+1),acc_HRL,'k--o')
plt.plot(range(1,len(files)+1),acc_AGREL,'g--o')
# plt.plot(range(1,len(files)+1),acc_DQN,'y--o')
# plt.plot(range(1,len(files)+1),acc_QLGBM,'--o')

# plt.legend(['Banditron','BanditronRP','HRL','AGREL','Deep Q-Learning','LightGBM based Q-Learning'])
plt.legend(['Banditron','BanditronRP','HRL','AGREL'])

plt.grid()
plt.ylim((20,120))
plt.ylabel('Accuracy (in %)')
plt.xlabel('Sessions')
plt.xticks(range(1,len(files)+1), labels=['session_'+str(i) for i in range(1,len(files)+1)])
                 
plt.title(directory.split('/')[-1]+' Performance plots ')
#plt.savefig(directory.split('/')[-1]+' Performance plots.jpg')
plt.show()

# For taking out the accuracy data
# acc_df = pd.DataFrame([acc_Banditron, acc_BanditronRP, acc_HRL, acc_AGREL, acc_DQN, acc_QLGBM]).T
# acc_df.rename(dict(enumerate(['Banditron','BanditronRP','HRL','AGREL','Deep Q-Learning','QLGBM'])),axis=1,inplace=True)
# acc_df.to_csv(directory.split('/')[-1]+' any-name-you-prefer.csv') # subject to change
