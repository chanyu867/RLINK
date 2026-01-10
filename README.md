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

---

## Dataset: LINK

We use neural recordings from the **LINK dataset**, where neural features are preprocessed into **20 ms bins**:

- **SBP (Spiking-Band Power)**  
  Power of high-frequency spiking-band signals (e.g., 300–1000 Hz)

- **TCR (Threshold Crossing Rate)**  
  Number of voltage threshold crossings within each bin

These features provide complementary representations of multi-unit activity.

## Key Analyses

- Accuracy over time (block-wise / day-wise)
- Identification of low-performance days
- Per-day sample and block statistics
- Electrode × day SBP heatmaps
- Model comparison plots across time

---

## How to run

Example: running a decoder and plotting performance

```
python -m venv RL_env
source activate RL_env/bin/activate
git clone https://github.com/chanyu867/RLINK.git
cd RLINK
git switch RL-side #make sure you are in right branch
pip install -r requirements.txt
chmod u+x src/RL_decoders/inference.sh
src/RL_decoders/inference.sh
```

---

### Outputs
<img width="1000" height="500" alt="idx_performance_comparison" src="https://github.com/user-attachments/assets/2cb4eca9-2714-4c63-b99b-f0ee35b6c50a" />

