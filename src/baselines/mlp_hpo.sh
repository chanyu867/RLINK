finger_ID="idx"        # or mrs
mode="position"        # position decoding
#"0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
trial_bin="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy"
target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

out_dir="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/hpo/mlp"

for boundary in "0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"; do
    for slicing in 1 5; do
        label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"
        python mlp_hpo.py \
            --sbp_path ${sbp_path} \
            --trial_bin_path ${trial_bin} \
            --label_path ${label_path} \
            --day_info_path ${day_info_path} \
            --slicing_day ${slicing} \
            --target_style_path ${target_style_path} \
            --epochs 50 \
            --n_trials 200 \
            --save_best_json "${out_dir}/hpo_${boundary}_slicing${slicing}.json"

    done
done