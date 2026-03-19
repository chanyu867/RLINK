

import joblib

import numpy as np

path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/pre_trained/idx_position_perceptron_weights_3classes.joblib"

obj = joblib.load(path)

print(obj.keys())
print(obj['coef'].T.shape)
