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
            
            python perceptron_eval.py \
                --sbp_path "${sbp_path}" \
                --label_path "${label_path}" \
                --day_info_path "${day_info_path}" \
                --slicing_day "${slicing_day}" \
                --target_type "${task}" \
                --target_style_path "${target_style_path}" \
                --trial_bin_path "${trial_bin_path}" \
                --out_dir "${out_root}/${task}" \
                --batch_size 64 \
                --max_test_samples 500000 \
                --prefix "perceptron_${task}_b${boundary}" \
                # --best_param "${best_param}"
                # If you want temporal stacking, uncomment:
                # --n_lags 10 --lag_step 1
        done
    done
done
