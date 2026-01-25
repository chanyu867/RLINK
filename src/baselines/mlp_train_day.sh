finger_ID="idx"        # or mrs
mode="position"        # position decoding

#"0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"

target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

for boundary in "0.33_0.66"; do
    label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"
    for task in "random"; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            python mlp_train_day.py \
                --sbp_path ${sbp_path} \
                --label_path ${label_path} \
                --day_info_path ${day_info_path} \
                --target_type ${task} \
                --target_style_path ${target_style_path} \
                --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/mlp_day/${task}" \
                --trial_bin_path /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy \
                --scale \
                --lag_group "trial" \
                --batch_size 256 \
                --epochs 250 \
                --prefix "mlp_${task}_b${boundary}" \
                --best_param "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/hpo/mlp/hpo_${boundary}_slicing1.json" \
                --seed ${seed}
        done
    done
done