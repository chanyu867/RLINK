'''
Refactored for Real-time Step-by-Step Execution
'''
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper: Online Lag Buffer ---
class OnlineLagBuffer:
    def __init__(self, d, n_lag, lag_step):
        self.d = int(d)
        self.n_lag = n_lag
        self.lag_step = lag_step
        self.enabled = (n_lag is not None) and (n_lag > 0)
        # buffer length needed: current sample + history
        self.maxlen = (n_lag * lag_step + 1) if self.enabled else 1
        self.buf = deque(maxlen=self.maxlen)

    def push(self, x_t):
        self.buf.append(np.asarray(x_t, dtype=float).reshape(-1))

    def feature(self):
        # If buffer isn't full yet, we pad with the oldest available sample or zeros
        # Simple approach: use whatever is in buffer
        if not self.enabled: 
            return self.buf[-1] if len(self.buf) > 0 else np.zeros(self.d)
        
        feat = np.zeros(self.d * (self.n_lag + 1))
        current_len = len(self.buf)
        
        for i in range(self.n_lag + 1):
            # i=0 -> latest, i=1 -> lag_step back...
            idx_back = i * self.lag_step
            buf_idx = current_len - 1 - idx_back
            
            if buf_idx >= 0:
                feat[i*self.d : (i+1)*self.d] = self.buf[buf_idx]
            else:
                # Padding (zeros) if history not available yet
                feat[i*self.d : (i+1)*self.d] = np.zeros(self.d) 
        return feat
    
    def reset(self):
        self.buf.clear()

# --- 1. Banditron Step ---
def Banditron_step(x_raw, y_true, state, params):
    """
    state: {'W': np.array, 'lagger': OnlineLagBuffer}
    """
    lagger = state['lagger']
    W = state['W']
    k = W.shape[0]
    gamma = params['gamma']
    
    # Feature extraction
    lagger.push(x_raw)
    x_t = lagger.feature() # (d_eff,)

    # Prediction
    scores = np.dot(W, x_t)
    y_hat = int(np.argmax(scores))
    
    # Exploration
    p = np.ones(k) * (gamma / k)
    p[y_hat] += (1 - gamma)
    y_tilde = int(np.random.choice(range(k), p=p))
    
    explore = (y_tilde != y_hat)

    # Update (Feedback)
    sparsify = np.random.rand() < params['sparsity_rate']
    if not sparsify:
        # Error = 1 if wrong, 0 if right
        err = 1.0 if (y_tilde != y_true) else 0.0
        
        # Binary feedback simulation (correct or not?)
        # With probability 'error' (noise), the feedback is flipped
        feedback_is_wrong = np.random.rand() < params['error']
        final_err = 1.0 - err if feedback_is_wrong else err
        
        # Update rule
        # If prediction was 'correct' (final_err=0), we reinforce y_tilde
        # If 'wrong' (final_err=1), we penalize
        if final_err == 1:
            # Penalize
            W[y_tilde] -= x_t / p[y_tilde]
        else:
            # Reinforce
            W[y_tilde] += x_t / p[y_tilde]
            
    return y_tilde, state, explore

# --- 2. Banditron RP Step ---
def BanditronRP_step(x_raw, y_true, state, params):
    """
    state: {'W': (k, 2), 'Wrand': (2, d_eff), 'lagger': ...}
    Note: Original paper/code projects d -> k dim via random matrix
    """
    lagger = state['lagger']
    W = state['W']         # Weights for the banditron part
    Wrand = state['Wrand'] # Random projection weights
    gamma = params['gamma']
    k_classes = W.shape[0] # output classes

    # Feature
    lagger.push(x_raw)
    x_t = lagger.feature()
    
    # Random Projection
    # Project d_eff -> m dimensions (often m=k or similar small number)
    f_t = sigmoid(np.dot(Wrand, x_t))
    
    # Banditron Logic on f_t
    scores = np.dot(W, f_t)
    y_hat = int(np.argmax(scores))
    
    p = np.ones(k_classes) * (gamma / k_classes)
    p[y_hat] += (1 - gamma)
    y_tilde = int(np.random.choice(range(k_classes), p=p))
    
    explore = (y_tilde != y_hat)

    # Update
    sparsify = np.random.rand() < params['sparsity_rate']
    if not sparsify:
        err = 1.0 if (y_tilde != y_true) else 0.0
        feedback_is_wrong = np.random.rand() < params['error']
        final_err = 1.0 - err if feedback_is_wrong else err
        
        if final_err == 1:
            W[y_tilde] -= f_t / p[y_tilde]
        else:
            W[y_tilde] += f_t / p[y_tilde]
            
    return y_tilde, state, explore

# --- 3. AGREL Step ---
def AGREL_step(x_raw, y_true, state, params):
    """
    state: {'W_H': ..., 'W_O': ..., 'lagger': ...}
    """
    lagger = state['lagger']
    W_H = state['W_H']
    W_O = state['W_O']
    
    # Feature
    lagger.push(x_raw)
    x_t = lagger.feature()
    x = np.insert(x_t, 0, 1).reshape(-1, 1) # Add Bias
    
    # Forward
    y_H = sigmoid(np.dot(W_H, x))
    Z = np.dot(W_O, y_H)
    y_O = softmax(Z)
    y_hat = int(np.argmax(y_O))
    
    # Explore
    explore = np.random.rand() < params['gamma']
    if explore:
        y_tilde = int(np.random.randint(0, W_O.shape[0]))
    else:
        y_tilde = y_hat
        
    # Update
    sparsify = np.random.rand() < params['sparsity_rate']
    if not sparsify:
        # Reward calculation
        is_correct = (y_tilde == y_true)
        # Flip based on error rate
        flip = np.random.rand() < params['error']
        perceived_correct = not is_correct if flip else is_correct
        
        # AGREL Delta
        # If correct: delta -> 1 - prob
        # If wrong:   delta -> -prob
        if perceived_correct:
            delta = 1.0 - y_O[y_tilde]
        else:
            delta = -y_O[y_tilde] # Note: y_O is column vector usually
            
        # Global reinforcer f
        if delta >= 0:
            f = delta / (1.0 - delta + 1e-6)
        else:
            f = delta
            
        # Backprop-like updates
        # dW_O = beta * f * y_H.T
        # dW_H = alpha * f * (y_H * (1-y_H) * W_O[y_tilde]) * x.T
        
        # Reshaping for safety
        y_O_val = float(y_O[y_tilde])
        
        update_O = params['beta'] * f * y_H.T
        W_O[y_tilde, :] += update_O.flatten()
        
        term1 = (y_H * (1 - y_H)) # elementwise
        term2 = W_O[y_tilde, :].reshape(-1, 1)
        # (hidden, 1) * (hidden, 1) = (hidden, 1)
        hidden_err = term1 * term2 
        
        update_H = params['alpha'] * f * np.dot(hidden_err, x.T)
        W_H += update_H
        
    return y_tilde, state, explore

# --- 4. HRL Step ---
def HRL_step(x_raw, y_true, state, params):
    """
    state: {'W': [W0, W1, ...], 'lagger': ...}
    """
    lagger = state['lagger']
    W = state['W'] # List of weights
    muH = params['muH']
    muO = params['muO']
    
    lagger.push(x_raw)
    x_t = lagger.feature()
    x = np.insert(x_t, 0, 1).reshape(-1, 1) # Bias
    
    # Forward Pass
    outs = [x]
    # Hidden layers (tanh)
    for i in range(len(W) - 1):
        net = np.dot(W[i], outs[-1])
        outs.append(np.tanh(net))
        
    # Output layer (tanh on sign of previous)
    net_last = np.dot(W[-1], np.sign(outs[-1]))
    outs.append(np.tanh(net_last))
    
    y_hat_raw = outs[-1] # This is usually vector of size classes
    y_hat = int(np.argmax(y_hat_raw))
    
    # HRL usually outputs continuous/tanh, we take argmax
    y_tilde = y_hat # HRL standard exploration is often inside update or via noise, simplified here
    
    explore = False # HRL exploration is implicit in Hebbian noise usually
    
    # Update
    sparsify = np.random.rand() < params['sparsity_rate']
    if not sparsify:
        is_correct = (y_tilde == y_true)
        flip = np.random.rand() < params['error']
        perceived_correct = not is_correct if flip else is_correct
        
        # Reinforcement signal f
        # f = 1 if correct, -1 if wrong (probabilistic)
        f = 1.0 if perceived_correct else -1.0
        
        # Update Output Layer
        # dW = mu * f * (sign(out) - out) * input.T
        # Hebbian-like rule from paper
        
        # Last layer
        o = outs[-1] # output
        i_last = outs[-2] # input to last layer (hidden)
        
        # Note: In HRL code provided previously, it used sign(i_last) for forward but i_last for update? 
        # We stick to the provided logic:
        term_O = (np.sign(o) - o)
        dW_O = muO * f * np.dot(term_O, i_last.T) # i_last is actually 'outs[-2]' from forward loop
        W[-1] += dW_O
        
        # Hidden Layers
        for i in range(len(W) - 2, -1, -1):
             curr_o = outs[i+1]
             curr_i = outs[i]
             
             # The update rule for hidden in HRL is local:
             term_H = (np.sign(curr_o) - curr_o)
             dW_H = muH * f * np.dot(term_H, curr_i.T)
             W[i] += dW_H
             
    return y_tilde, state, explore