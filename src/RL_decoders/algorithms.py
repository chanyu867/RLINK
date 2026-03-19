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
# def DQN(X,Y,day_info, error,sparsity_rate,epsilon,gamma):
def DQN(X, Y, day_info, error, sparsity_rate, epsilon, gamma, weights_load_path=None, weights_save_path=None):

  inp_shp = X.shape[1] # Number of electrodes --> corresponding to the no. of neurons in the first layer.
  model = get_DQN_model(inp_shp)

  if weights_load_path is not None and os.path.exists(weights_load_path):
      model.load_weights(weights_load_path)
      logger.info(f"[DQN] Successfully loaded saved weights from {weights_load_path}")
    
  T = X.shape[0]
  scaler = StandardScaler()
  X_norm = scaler.fit_transform(X)
  pred = []
  when_explore = []
  
  # Evaluative framework (refer to the paper to understand the mathematics)
  logger.info(f"each shape: {X_norm.shape}, {Y.shape}, {Y[:10]}")
  #each shape: (10000, 96), (10000,)

  for t in range(T):
    if t % 1000 == 0 or t == T - 1:
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

  if weights_save_path is not None:
    model.save_weights(weights_save_path)
    logger.info(f"[DQN] Saved weights to {weights_save_path}")

  return np.array(pred), np.array(when_explore), gamma

# def DQN_ewc(X, Y, day_info, error, sparsity_rate, epsilon, gamma, ewc_lambda=10.0, weights_load_path=None, weights_save_path=None):
#     """
#     DQN implemented with Elastic Weight Consolidation (EWC) using tf.GradientTape.
#     ewc_lambda controls how strongly the model remembers previous days (tasks).
#     """
#     inp_shp = X.shape[1] 
#     model = get_DQN_model(inp_shp)
    
#     # We define the optimizer and loss function manually for GradientTape
#     optimizer = tf.keras.optimizers.Adam()
#     loss_fn = tf.keras.losses.MeanSquaredError()

#     if weights_load_path is not None and os.path.exists(weights_load_path):
#         model.load_weights(weights_load_path)
#         logger.info(f"[DQN] Successfully loaded saved weights from {weights_load_path}")
        
#     T = X.shape[0]
#     scaler = StandardScaler()
#     X_norm = scaler.fit_transform(X)
    
#     pred = []
#     when_explore = []
    
#     # --- EWC Variables ---
#     star_weights = []     # Saves the critical weights from the previous day (theta_A)
#     fisher_matrix = []    # Saves the importance of each weight (F)
#     recent_x_buffer = []  # Holds recent data to calculate the Fisher matrix at boundaries
#     buffer_size = 200     # How many samples to use for Fisher estimation

#     logger.info(f"DQN Data shape: {X_norm.shape}, Labels shape: {Y.shape}")

#     for t in range(T):
#         if t % 1000 == 0 or t == T - 1:
#             logger.info(f"\r[DQN] Processing step {t+1}/{T}")

#         x = X_norm[t,:].reshape(1, X.shape[1]).astype(np.float32)
#         y_true = Y[t]
        
#         # Keep a rolling buffer of recent inputs to calculate the Fisher matrix later
#         recent_x_buffer.append(x)
#         if len(recent_x_buffer) > buffer_size:
#             recent_x_buffer.pop(0)

#         # =================================================================
#         # 1. TASK BOUNDARY DETECTION & FISHER MATRIX CALCULATION
#         # =================================================================
#         if day_info is not None and t > 0 and day_info[t] != day_info[t-1]:
#             logger.info(f"\n[DQN] Day change detected at t={t}: Day {day_info[t-1]} -> Day {day_info[t]}. Calculating EWC Fisher Matrix...")
            
#             # Save the current "perfect" weights for the day that just finished
#             star_weights = [tf.identity(w) for w in model.trainable_variables]
            
#             # Approximate the Fisher Information Matrix using the recent buffer
#             fisher_matrix = [tf.zeros_like(w) for w in model.trainable_variables]
            
#             for bx in recent_x_buffer:
#                 with tf.GradientTape() as tape:
#                     # Log-likelihood approximation for RL (gradient of max Q-value)
#                     q_vals = model(bx, training=False)
#                     max_q = tf.reduce_max(q_vals)
                
#                 # Get gradients of the max Q-value with respect to weights
#                 grads = tape.gradient(max_q, model.trainable_variables)
                
#                 # Square the gradients and accumulate them
#                 for i, g in enumerate(grads):
#                     if g is not None:
#                         fisher_matrix[i] += tf.square(g) / float(len(recent_x_buffer))
            
#             logger.info("[DQN] Fisher Matrix calculated. Spring penalties active for the new day.")
#             recent_x_buffer.clear() # Reset buffer for the new day

#         # =================================================================
#         # 2. STANDARD DQN LOGIC (Exploration, Q-values, Target Vector)
#         # =================================================================
#         Q = model(x, training=False).numpy()
#         yhat = np.argmax(Q)
        
#         explore = np.random.uniform() < epsilon
#         if explore:
#             ytilde = np.random.randint(low=0, high=4)
#         else:
#             ytilde = yhat
            
#         sparsify = np.random.choice([True, False], p=[sparsity_rate, 1 - sparsity_rate])
        
#         if not sparsify:
#             if ytilde == y_true:
#                 r = np.random.choice([-1, 1], p=[error, 1 - error])
#             else:
#                 r = np.random.choice([1, -1], p=[error, 1 - error])

#             Target = r + gamma * np.amax(Q)
#             Target_vec = np.copy(Q)
#             Target_vec[0, ytilde] = Target

#             # =================================================================
#             # 3. EWC CUSTOM TRAINING STEP (Replaces model.fit)
#             # =================================================================
#             with tf.GradientTape() as tape:
#                 # Get predictions
#                 Q_pred = model(x, training=True)
                
#                 # Base Loss: Mean Squared Error
#                 loss = loss_fn(Target_vec, Q_pred)
                
#                 # EWC Penalty: Add the "spring" mechanism if we have past tasks
#                 if len(star_weights) > 0:
#                     ewc_penalty = 0.0
#                     for v, star_v, fisher in zip(model.trainable_variables, star_weights, fisher_matrix):
#                         # Formula: lambda/2 * Fisher * (current_weight - star_weight)^2
#                         ewc_penalty += tf.reduce_sum(fisher * tf.square(v - star_v))
                    
#                     # Combine Base Loss and Penalty
#                     loss += (ewc_lambda / 2.0) * ewc_penalty
            
#             # Calculate gradients based on the COMBINED loss and apply them
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         pred.append(ytilde)
#         when_explore.append(explore)

#     if weights_save_path is not None and t == T - 1:
#         model.save_weights(weights_save_path)
#         logger.info(f"[DQN] Saved final weights to {weights_save_path}")

#     return np.array(pred), np.array(when_explore), gamma


def DQN_ewc(X, Y, day_info, error, sparsity_rate, epsilon, gamma, ewc_lambda=1.0, fisher_decay=0.95, trial_ids=None, weights_load_path=None, weights_save_path=None):
    """
    Online EWC for DQN. Calculates the Fisher Information Matrix at the end of 
    every trial and applies an exponential decay to forget distant past data.
    """
    inp_shp = X.shape[1] 
    model = get_DQN_model(inp_shp)
    
    # Using legacy Adam for M1/M2 Mac speedups
    optimizer = tf.keras.optimizers.legacy.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    if weights_load_path is not None and os.path.exists(weights_load_path):
        model.load_weights(weights_load_path)
        logger.info(f"[DQN] Successfully loaded saved weights from {weights_load_path}")
        
    T = X.shape[0]
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    pred = []
    when_explore = []
    
    # --- Online EWC Variables ---
    star_weights = []     
    # Initialize a running global Fisher matrix with zeros
    running_fisher = [tf.zeros_like(w) for w in model.trainable_variables]
    current_trial_buffer = []  

    logger.info(f"DQN Data shape: {X_norm.shape}, Labels shape: {Y.shape}")
    if trial_ids is None:
        logger.warning("[DQN_ewc] trial_ids not provided! Will default to day-boundary updates.")

    for t in range(T):
        if t % 1000 == 0 or t == T - 1:
            logger.info(f"\r[DQN] Processing step {t+1}/{T}")

        x = X_norm[t,:].reshape(1, X.shape[1]).astype(np.float32)
        y_true = Y[t]
        
        # Collect data for the current trial
        current_trial_buffer.append(x)

        # =================================================================
        # 1. STANDARD DQN LOGIC (Exploration, Q-values, Target Vector)
        # =================================================================
        Q = model(x, training=False).numpy()
        yhat = np.argmax(Q)
        
        explore = np.random.uniform() < epsilon
        if explore:
            ytilde = np.random.randint(low=0, high=4)
        else:
            ytilde = yhat
            
        sparsify = np.random.choice([True, False], p=[sparsity_rate, 1 - sparsity_rate])
        
        if not sparsify:
            if ytilde == y_true:
                r = np.random.choice([-1, 1], p=[error, 1 - error])
            else:
                r = np.random.choice([1, -1], p=[error, 1 - error])

            Target = r + gamma * np.amax(Q)
            Target_vec = np.copy(Q)
            Target_vec[0, ytilde] = Target

            # =================================================================
            # 2. EWC CUSTOM TRAINING STEP
            # =================================================================
            with tf.GradientTape() as tape:
                Q_pred = model(x, training=True)
                loss = loss_fn(Target_vec, Q_pred)
                
                # Apply the decaying spring penalty if we have past tasks
                if len(star_weights) > 0:
                    ewc_penalty = 0.0
                    for v, star_v, fisher in zip(model.trainable_variables, star_weights, running_fisher):
                        ewc_penalty += tf.reduce_sum(fisher * tf.square(v - star_v))
                    loss += (ewc_lambda / 2.0) * ewc_penalty
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pred.append(ytilde)
        when_explore.append(explore)

        # =================================================================
        # 3. TRIAL BOUNDARY DETECTION & ONLINE EWC UPDATE
        # =================================================================
        is_trial_end = False
        if trial_ids is not None:
            # Check if the next sample belongs to a different trial
            if t < T - 1 and trial_ids[t] != trial_ids[t+1]:
                is_trial_end = True
            elif t == T - 1:
                is_trial_end = True
        else:
            # Fallback to day changes if trial_ids are missing
            if day_info is not None and t < T - 1 and day_info[t] != day_info[t+1]:
                is_trial_end = True

        if is_trial_end and len(current_trial_buffer) > 0:
            # Calculate Trial-specific Fisher
            trial_fisher = [tf.zeros_like(w) for w in model.trainable_variables]
            for bx in current_trial_buffer:
                with tf.GradientTape() as tape2:
                    q_vals = model(bx, training=False)
                    max_q = tf.reduce_max(q_vals)
                grads2 = tape2.gradient(max_q, model.trainable_variables)
                
                for i, g in enumerate(grads2):
                    if g is not None:
                        trial_fisher[i] += tf.square(g) / float(len(current_trial_buffer))
            
            # Apply exponential decay to the historical Fisher matrix, then add the new one
            for i in range(len(running_fisher)):
                running_fisher[i] = (fisher_decay * running_fisher[i]) + trial_fisher[i]
            
            # Update anchor weights to the CURRENT state
            star_weights = [tf.identity(w) for w in model.trainable_variables]
            
            current_trial_buffer.clear()

    if weights_save_path is not None and t == T - 1:
        model.save_weights(weights_save_path)
        logger.info(f"[DQN] Saved final weights to {weights_save_path}")

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