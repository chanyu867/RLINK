#!/bin/bash

# --- Data Paths ---
SBP_PATH="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
LABEL_PATH="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/idx_position_labels_0.33_0.66_shift0.npy"
DAY_INFO_PATH="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
TRIAL_BIN_PATH="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy"
TARGET_STYLE_PATH="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"
TOML_PATH="src/RL_decoders/config/config.toml"
LOG_DIR="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/logs"

# --- HPO Parameters ---
N_TRIALS=100
MAX_SAMPLES=1000000
METRIC="all_class_trials" # Options: 'whole', 'transition', 'all_class_trials'

# --- Data Pipeline / Masking Parameters ---
TARGET_TYPE="random"
LABEL_MASK="0,2"
N_LAGS=16
LAG_STEP=4

# --- Models to Optimize ---
# You can remove models from this array if you don't want to run them all at once
MODELS=("banditron" "banditronRP" "HRL" "AGREL" "DQN") # "QLGBM"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Starting HPO for RL Decoders"
echo "Metric: $METRIC"
echo "Masking: target_type=$TARGET_TYPE, labels=$LABEL_MASK"
echo "Lags: n_lags=$N_LAGS, step=$LAG_STEP"
echo "Targeting max $MAX_SAMPLES samples per model"
echo "================================================================"

for MODEL in "${MODELS[@]}"; do
    echo ">>> Optimizing model: $MODEL"
    
    # Using 'python -m' to ensure local imports in src.RL_decoders work correctly
    python -m src.RL_decoders.position_decoder_hpo \
        --sbp_path "$SBP_PATH" \
        --label_path "$LABEL_PATH" \
        --day_info_path "$DAY_INFO_PATH" \
        --trial_bin_path "$TRIAL_BIN_PATH" \
        --target_style_path "$TARGET_STYLE_PATH" \
        --toml_path "$TOML_PATH" \
        --model_type "$MODEL" \
        --metric_type "$METRIC" \
        --target_type "$TARGET_TYPE" \
        --label_mask "$LABEL_MASK" \
        --n_lags $N_LAGS \
        --lag_step $LAG_STEP \
        --n_trials $N_TRIALS \
        --max_samples $MAX_SAMPLES \
        2>&1 | tee -a "$LOG_DIR/hpo_${MODEL}.log"

    echo ">>> Finished $MODEL. Results saved to best_params_${MODEL}_${METRIC}.json"
    echo "----------------------------------------------------------------"
done

echo "All HPO trials complete."