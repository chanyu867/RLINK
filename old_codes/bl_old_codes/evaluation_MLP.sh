finger_ID="idx"        # or mrs
mode="position"        # position decoding

#"0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"

target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

out_dir="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/${finger_ID}_${mode}"

# for boundary in "0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"; do
for boundary in "0.5"; do
    label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"
    for slicing_day in 1; do
        for task in "random"; do
            python /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/old_codes/bl_old_codes/MLP_finger_decoder_eval.py \
                --sbp_path ${sbp_path} \
                --label_path ${label_path} \
                --day_info_path ${day_info_path} \
                --slicing_day ${slicing_day} \
                --target_type ${task} \
                --target_style_path ${target_style_path} \
                --boundary ${boundary} \
                --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP/idx_position_test_1dim_trials" \
                --prefix "mlp_random" \
                --batch_size 64 \
                --max_test_samples 10000 \
                --trial_bin_path /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy \
                --save_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP"
                
        done
    done
done