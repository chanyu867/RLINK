

<img width="2693" height="880" alt="Gemini_Generated_Image_16cblk16cblk16cb-2" src="https://github.com/user-attachments/assets/8f51595e-a718-43fe-ae61-53a74d312f14" />


# RLINK - Reinforcement Learning for LINK dataset
This is a repository for ML final project in Bar-Ilan University in 2025. The code contains implementations and analyses of **reinforcement learning–based neural decoders** applied to intracortical brain–machine interface (iBMI) data from the **LINK dataset**. The project focuses on lightweight, online-learning decoders that are robust to **neural non-stationarity** across recording sessions and days.

---

## Implemented Models

The following decoders are implemented and evaluated:

### 1. Banditron
- Single-layer linear model
- Online multiclass bandit learning
- Extremely low computational cost
- Suitable for approximately linearly separable neural representations

### 2. Banditron-RP
- Extension of Banditron with a fixed **random projection + nonlinearity**
- Introduces limited nonlinearity while preserving efficiency
- Empirically improves performance on some datasets

### 3. AGREL (Attention-Gated Reinforcement Learning)
- Biologically inspired RL algorithm
- Uses stochastic softmax action selection and reward-modulated plasticity
- More expressive but computationally heavier

### 4. HRL (Hebbian Reinforcement Learning)
- Hebbian learning rule modulated by scalar reward
- Local, biologically plausible weight updates
- Serves as a reinforcement-based baseline decoder

### 5. DQN (Deep Q Netwoek)

- A value-based RL algorithm
- utilize a deep neural network to approximate the optimal action-value function

---

## Dataset: LINK

We use neural recordings from the **LINK dataset**, where neural features are preprocessed into **20 ms bins**:

- **SBP (Spiking-Band Power)**  
  Power of high-frequency spiking-band signals (e.g., 300–1000 Hz)

---

## How to run

Example: running a decoder and plotting performance

```
## 0. craete environment
python -m venv RL_env
source activate RL_env/bin/activate
git clone https://github.com/chanyu867/RLINK.git
cd RLINK
pip install -r requirements.txt

## 1. run hyper-parameter optimization
chmod +x src/baselines/mlp_hpo.sh src/RL_decoders/hpo.sh
./src/baselines/mlp_hpo.sh #for baseline model (MLP)
./src/RL_decoders/hpo.sh #for Reinforcement Learning models

## 2. run model training
chmod +x src/baselines/mlp_train.sh src/baselines/mlp_train_day.sh src/RL_decoders/inference.sh
./src/baselines/mlp_train.sh #train fixed-weights model (MLP)
./src/baselines/mlp_train_day.sh #train per-day models (MLP)
./src/RL_decoders/inference.sh #train RL-based models

## 3. evaluation
chmod +x src/baselines/mlp_eval.sh src/baselines/mlp_eval_day.sh
./src/baselines/mlp_eval.sh #for fixded-weights model (MLP)
./src/baselines/mlp_eval_day.sh #for per-day models (MLP)
python -m src.post_analysis.post_performance_analysis --post_analysis #for RL-based models

```

---
