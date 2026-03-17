'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 
import os
from src.utils import *
import warnings
import time
warnings.filterwarnings('ignore')
import numpy as np
# from scipy.io import loadmat
# import pandas as pd
# import os
# from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, BatchNormalization
# from keras.utils import to_categorical, plot_model
# from keras.optimizers import Adam, RMSprop, SGD
# from keras.activations import relu
# from keras import Model, Input
# from keras.losses import binary_crossentropy
# from keras.metrics import Accuracy
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
# import scipy.special as sp

from scipy.special import expit as sigmoid
from scipy.special import softmax
import logging

# Defining the Banditron Function (Single Layered Network)
'''
Here, **kwargs is used to denote the arbitary input functions. The error and sparsity_rate
corresponds to the error and the sparsity introduced to the feedback signal (refer to the 
paper to understand the physical significance). k denotes the number of classes and is 
fixed at 4 for our experiments. X denotes the spike count data (observation -- input dataset) 
and y denotes the true labels associated with each observation (X) 
'''

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# def banditron(X, y, day_info, error, sparsity_rate, k, gamma): #**kwargs: gamma, eta: The exploration exploitation constant and eta are given as optional arguments.
def banditron(X, y, day_info, error, sparsity_rate, k, gamma):
    T = X.shape[0]
    d = X.shape[1]

    W = np.zeros((k, d))
    error_count = np.zeros(T)
    pred = []
    when_explore = []

    for t in range(T):

        # Day boundary: reset weights (online assumption)
        if day_info is not None and t > 0 and day_info[t] != day_info[t-1]:
            W = np.zeros((k, d))
            logger.info(f"[Banditron/BanditronRP] - Day change detected at t={t}: {day_info[t-1]} -> {day_info[t]}, resetting weights")

        x_t = X[t]

        p = [gamma / k for _ in range(k)]
        y_hat = int(np.argmax(np.dot(W, x_t)))
        p[y_hat] = p[y_hat] + 1 - gamma
        y_tilde = int(np.random.choice(range(k), p=p))

        pred.append(y_tilde)
        when_explore.append(int(y_tilde != y_hat))

        sparsify = np.random.choice([True, False], p=[sparsity_rate, 1 - sparsity_rate])
        if not sparsify:
            if y_tilde != y[t]:
                choice = np.random.choice(range(2), p=[error, 1 - error])
                if choice == 1:
                    W[y_hat] = W[y_hat] - x_t
                else:
                    W[y_hat] = W[y_hat] - x_t
                    W[y_tilde] = W[y_tilde] + x_t / p[y_tilde]
            else:
                choice = np.random.choice(range(2), p=[error, 1 - error])
                if choice == 1:
                    W[y_hat] = W[y_hat] - x_t
                    W[y_tilde] = W[y_tilde] + x_t / p[y_tilde]
                else:
                    W[y_hat] = W[y_hat] - x_t

    return pred, when_explore, gamma

# Defining the Banditron-RP Function (Three Layered Network)
'''
Here, **kwargs is used to denote the arbitary input functions. The error and sparsity_rate
corresponds to the error and the sparsity introduced to the feedback signal (refer to the 
paper to understand the physical significance). k denotes the number of classes and is 
fixed at 4 for our experiments. X denotes the spike count data (observation -- input dataset) 
and y denotes the true labels associated with each observation (X) 
'''


def banditronRP(X, y, day_info, error, sparsity_rate, k, gamma):
    d = X.shape[1]
    Wrand = np.random.uniform(size=(k,d)) # The random Weight matrix generated from a normal distribution.
    f = sigmoid(np.dot(Wrand,X.T)) # The non-linear projection vector input to the hidden layer.
    pred, when_explore, gamma = banditron(f.T, y, day_info, error, sparsity_rate, k=2, gamma=gamma) # f(t) = Sigmoid(Wrand.x(t)) is given as an input to the Banditron.
    return pred, when_explore, gamma



# Defining the HRL function (Three Layered Network)
# Initializing the weight matrices
'''
In HRL the weight matrices are not initialized as zero matrix instead random floats
are extracted from Gaussian Distribution (mean = 0, var = 1). Here inp_shp denotes the
number of rows, and out_shp the number of columns for the weight matrix.
'''
def initialize(inp_shp,out_shp):
  W = np.random.randn(out_shp,inp_shp)
  return W

'''
The error and sparsity_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). muH and muO
denotes the learning rates corresponding to the weight updation policy. num_nodes is a matrix referring to 
the number of hidden nodes and output nodes. X denotes the spike count data (observation -- input dataset) 
and y denotes the true labels associated with each observation (X) 
'''  
def HRL(X, y, day_info, muH, muO, num_nodes, error, sparsity_rate):
    T = X.shape[0]
    d = X.shape[1]

    # copy num_nodes to avoid in-place modification
    nodes = list(num_nodes)
    nodes[0] = d 

    # bias is inserted => +1
    nodes0_with_bias = nodes[0] + 1

    W = [0] * (len(nodes) - 1)
    pred = []

    # init weights (same style as original HRL initialize())
    # first layer expects nodes0_with_bias
    W[0] = initialize(nodes0_with_bias, nodes[1])
    for i in range(2, len(nodes)):
        W[i - 1] = initialize(nodes[i - 1], nodes[i])

    for t in range(T):

        if day_info is not None and t > 0 and day_info[t] != day_info[t-1]:
            W[0] = initialize(nodes0_with_bias, nodes[1])
            for i in range(2, len(nodes)):
                W[i - 1] = initialize(nodes[i - 1], nodes[i])

        x_feat = X[t]
        x = np.insert(x_feat, 0, 1)  # add bias
        out = [x.reshape(-1, 1)] * (len(nodes))

        for i in range(1, len(nodes)):
            out[i] = np.tanh(np.dot(W[i - 1], out[i - 1]))

        out[-1] = np.tanh(np.dot(W[-1], np.sign(out[-2])))
        yhat = int(np.argmax(out[-1]))

        sparsify = np.random.choice([True, False], p=[sparsity_rate, 1 - sparsity_rate])
        if not sparsify:
            if yhat == y[t]:
                f = np.random.choice([-1, 1], p=[error, 1 - error])
            else:
                f = np.random.choice([1, -1], p=[error, 1 - error])

            dW = [0] * (len(nodes) - 1)
            for i in range(1, len(nodes)):
                dW[i - 1] = muH * f * (np.dot((np.sign(out[i]) - out[i]), out[i - 1].T)) + \
                            muH * (1 - f) * (np.dot((1 - np.sign(out[i]) - out[i]), out[i - 1].T))
                W[i - 1] = W[i - 1] + dW[i - 1]

            dW[-1] = muO * f * (np.dot((np.sign(out[-1]) - out[-1]), out[-2].T)) + \
                     muO * (1 - f) * (np.dot((1 - np.sign(out[-1]) - out[-1]), out[-2].T))
            W[-1] = W[-1] + dW[-1]

        pred.append(yhat)

    return pred, None, None


# Defining the AGREL function (Three Layered Network)
# Initializing the weight matrices
'''
In AGREL the weight matrices are not initialized as zero matrix instead random floats
are extracted from Gaussian Distribution (mean = 0, var = 1). Here inp_shp denotes the
number of rows, and out_shp the number of columns for the weight matrix.
'''
def initialize(inp_shp,out_shp):
  W = np.random.uniform(low=-1,high=1,size=(out_shp,inp_shp))
  return W

'''
The error and sparsity_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). alpha, and beta
denotes the learning rates corresponding to the weight updation policy. num_nodes is a matrix referring to 
the number of hidden nodes and output nodes. gamma denotes the exploration-exploitation trade-off. 
X denotes the spike count data (observation -- input dataset) and y denotes the true labels associated with each observation (X) 
''' 
def AGREL(X, y, day_info, error, sparsity_rate, gamma, alpha, beta, num_nodes):
    T = X.shape[0]
    d = X.shape[1]

    nodes = list(num_nodes)
    nodes[0] = d 

    pred = []
    when_explore = []

    # W_H expects (input+1) because bias is inserted
    W_H = initialize(nodes[0] + 1, nodes[1])
    W_O = initialize(nodes[1], nodes[2])

    for t in range(T):

        if day_info is not None and t > 0 and day_info[t] != day_info[t-1]:
            W_H = initialize(nodes[0] + 1, nodes[1])
            W_O = initialize(nodes[1], nodes[2])

        x_feat = X[t]
        x = np.insert(x_feat, 0, 1).reshape(-1, 1)

        y_H = sigmoid(np.dot(W_H, x))
        Z = np.dot(W_O, y_H)
        y_O = softmax(Z)
        yhat = int(np.argmax(y_O))

        outs = np.zeros(Z.shape)
        outs[yhat] = 1

        explore = (np.random.uniform() < gamma)
        when_explore.append(explore)

        y_tilde = int(np.random.randint(low=0, high=nodes[-1])) if explore else yhat

        sparsify = np.random.choice([True, False], p=[sparsity_rate, 1 - sparsity_rate])
        if not sparsify:
            if y_tilde == y[t]:
                delta = np.random.choice([-1, 1 - float(outs[y_tilde])], p=[error, 1 - error])
            else:
                delta = np.random.choice([-1, 1 - float(outs[y_tilde])], p=[1 - error, error])

            if delta >= 0:
                f = delta / (1 - delta + 1e-4)
            else:
                f = delta

            dW_O = beta * f * y_H.T
            dW_H = alpha * f * np.dot((y_H * (1 - y_H) * W_O[y_tilde, :].reshape(-1, 1)), x.T)

            W_O[y_tilde, :] = W_O[y_tilde, :] + dW_O
            W_H = W_H + dW_H

        pred.append(yhat)

    return pred, when_explore, gamma


# Defining the DQN function (Four Layered Network)
# Defining the DQN model
'''
Here, the Deep Q Learning model is computed as a four layer network, where the number of
nodes in the input layer changes with the experiment, and is given by inp_shp. This 
following function sequentially builds the model, where the output layer has 4 classes,
and the two hidden layers has 128 neurons.
'''
def get_DQN_model(inp_shp, lr=0.01):
  # np.random.seed(101)
  model = Sequential()
  model.add(Dense(128, activation='relu', input_shape=(inp_shp,))) # hidden layer1 neurons = 128
  model.add(Dense(128, activation='relu')) # hidden layer2 neurons = 128
  model.add(Dense(4, activation='linear')) # Output layer neurons = 4
  model.compile(loss='mse',optimizer='adam') # A mse loss function is used with an adam optimizer.
  return model

'''
The error and sparsity_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). epsilon and gamma 
denotes the exploration constant and discount factor. X denotes the spike count data 
(observation -- input dataset) and y denotes the true labels associated with each observation (X). 
'''
# def DQN(X,Y,epsilon,gamma,error,sparsity_rate):
def DQN(X,Y,day_info, error,sparsity_rate,epsilon,gamma):

  inp_shp = X.shape[1] # Number of electrodes --> corresponding to the no. of neurons in the first layer.
  model = get_DQN_model(inp_shp)
  T = X.shape[0]
  scaler = StandardScaler()
  X_norm = scaler.fit_transform(X)
  pred = []
  when_explore = []
  
  # Evaluative framework (refer to the paper to understand the mathematics)
  logger.info(f"each shape: {X_norm.shape}, {Y.shape}, {Y[:10]}")
  #each shape: (10000, 96), (10000,)

  for t in range(T):
    if t % 10 == 0 or t == T - 1:
      logger.info(f"\r[DQN] Processing step {t+1}/{T}")

    x = X_norm[t,:].reshape(1,X.shape[1]).astype(np.float32)
    # y = Y[t,:].reshape(1,Y.shape[1])
    y_true = Y[t]
    Q = model.predict(x, verbose=0)

    yhat = np.argmax(Q)
    explore = np.random.uniform() < epsilon
    if explore:
      ytilde = np.random.randint(low=0,high=4)
    else:
      ytilde = yhat
    sparsify = np.random.choice([True,False],p=[sparsity_rate,1-sparsity_rate])
    if not sparsify:
      # if ytilde == np.argmax(y):
      if ytilde == y_true:
        r = np.random.choice([-1 ,1],p=[error,1-error])
      else:
        r = np.random.choice([1 ,-1],p=[error,1-error])

      Target = r + gamma*np.amax(Q)
      Target_vec = Q
      Target_vec[0,ytilde] = Target
      Target_vec = np.array(Target_vec).reshape(1,4)

      model.fit(x,Target_vec,batch_size=1, epochs=1,verbose=0)

    pred.append(ytilde)
    when_explore.append(explore)

  return np.array(pred), np.array(when_explore), gamma


# Defining the LGBM based Q-Learning Function
# Building the LGBM Model
'''
Here, instead of using a deep neural network to build the DQN framework, we have 
used the LightGBM framework with a Q-Learning policy. We call this model the QLGBM Network.
'''
def get_QLGBM_model():
  model = MultiOutputRegressor(LGBMRegressor(n_jobs=-1)) 
  return model

'''
The error and sparsity_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). epsilon and gamma 
denotes the exploration constant and discount factor. X denotes the spike count data 
(observation -- input dataset) and y denotes the true labels associated with each observation (X). 
'''
def QLGBM(X, Y, day_info, error, sparsity_rate, epsilon, gamma):
  model = get_QLGBM_model()
  T = X.shape[0]
  scaler = StandardScaler()
  X_norm = scaler.fit_transform(X)
  
  pred = []
  when_explore = []
  isFit = False
  
  # Evaluative framework 
  for t in range(T):
    # Clean progress tracker
    if t % 1000 == 0 or t == T - 1:
        print(f"\r[QLGBM] Processing step {t+1}/{T}", end="")

    # Trick to keep batch size = 2 for LightGBM's tree splits, even on the last sample
    if t == T - 1:
        x = np.vstack([X_norm[t, :], X_norm[t, :]]).astype(np.float32)
    else:
        x = X_norm[t:t+2, :].astype(np.float32)
        
    # FIX: Correctly index 1D label array
    y_true = Y[t] 
    
    if isFit:
      Q = model.predict(x)
    else:
      Q = np.random.uniform(low=-1, high=1, size=(2, 4))
    
    yhat = np.argmax(Q[0, :])
    explore = np.random.uniform() < epsilon
    
    if explore:
      ytilde = np.random.randint(low=0, high=4)
    else:
      ytilde = yhat
      
    sparsify = np.random.choice([True, False], p=[sparsity_rate, 1-sparsity_rate])
    
    if not sparsify:
      # FIX: Compare directly against the integer label instead of using argmax
      if ytilde == y_true: 
        r = np.random.choice([-1, 1], p=[error, 1-error])
      else:
        r = np.random.choice([1, -1], p=[error, 1-error])
        
      Target = r + gamma * np.amax(Q)
      Target_vec = Q
      Target_vec[0, ytilde] = Target
      
      model.fit(x, Target_vec)
      isFit = True
      
    pred.append(ytilde)
    when_explore.append(explore)

  print() # Clean newline
  
  # FIX: Return all three expected variables and ensure predictions match length T
  return np.array(pred), np.array(when_explore), gamma