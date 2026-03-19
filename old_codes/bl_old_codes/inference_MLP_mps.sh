finger_ID="idx"        # or mrs
mode="position"        # position decoding

#"0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"

target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

out_dir="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/${finger_ID}_${mode}"

#for 4 and 5 class task
# for boundary in "0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"; do
# for boundary in "0.33_0.66"; do
#     label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"
#     for slicing_day in 1; do
#         for task in "center-out" "random"; do
#             python MLP_finger_decoder_mps.py \
#                 --sbp_path ${sbp_path} \
#                 --label_path ${label_path} \
#                 --day_info_path ${day_info_path} \
#                 --slicing_day ${slicing_day} \
#                 --target_type ${task} \
#                 --target_style_path ${target_style_path} \
#                 --hidden_sizes "75, 75" \
#                 --scale \
#                 --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP/idx_position_75_75/${task}" \
#                 --prefix "mlp_idx_position_center-out_day50_b${boundary}" \
#                 --batch_size 256
#         done
#     done
# done

for boundary in "0.5"; do #"0.33_0.66"
    label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"
    for slicing_day in 1; do
        for task in "random"; do
            python /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/old_codes/bl_old_codes/MLP_finger_decoder_with_lag.py \
                --sbp_path ${sbp_path} \
                --label_path ${label_path} \
                --day_info_path ${day_info_path} \
                --slicing_day ${slicing_day} \
                --target_type ${task} \
                --target_style_path ${target_style_path} \
                --hidden_sizes "256" \
                --scale \
                --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP/boudary_${boudary}/${task}" \
                --trial_bin_path /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy \
                --prefix "mlp_center-out_b${boundary}" \
                --batch_size 256 \
                --epochs 250 \
                --lag_group "trial" \
                # --n_lags 10 --lag_step 1
        done
    done
done