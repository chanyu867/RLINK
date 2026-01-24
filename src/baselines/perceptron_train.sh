finger_ID="idx"        # or mrs
mode="position"        # position decoding

# data paths (same as your mlp_train.sh)
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"
trial_bin_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy"

# output root (you had multiple out_dir definitions; this mirrors your baseline_perf style)
out_root="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/perceptron"

for boundary in "0.33_0.66"; do
    label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"

    for slicing_day in 1; do
        for task in "random"; do

            # optional: if you have HPO json for perceptron, point to it; otherwise remove --best_param line
            best_param="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/hpo/perceptron/hpo_${boundary}_slicing${slicing_day}.json"

            python perceptron_train.py \
                --sbp_path "${sbp_path}" \
                --label_path "${label_path}" \
                --day_info_path "${day_info_path}" \
                --slicing_day "${slicing_day}" \
                --target_type "${task}" \
                --target_style_path "${target_style_path}" \
                --trial_bin_path "${trial_bin_path}" \
                --out_dir "${out_root}/${task}" \
                --scale \
                --lag_group "trial" \
                --batch_size 256 \
                --epochs 250 \
                --lr 0.01 \
                --weight_decay 0.0 \
                --prefix "perceptron_${task}_b${boundary}" \
                --n_lags 10 --lag_step 1 \
                # --best_param "${best_param}"
                # If you want temporal stacking, uncomment:
        done
    done
done
