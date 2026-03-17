finger_ID="idx"        # or mrs
mode="position"        # position decoding
classification="0.33_0.66"  # classification boundary

#"0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"

target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

out_dir="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/${finger_ID}_${mode}"

label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${classification}_shift0.npy"

python MLP_finger_decoder.py \
  --sbp_path ${sbp_path} \
  --label_path ${label_path} \
  --day_info_path ${day_info_path} \
  --slicing_day 90 \
  --target_type "center-out" \
  --target_style_path ${target_style_path} \
  --max_iter 200 \
  --hidden_sizes "75" \
  --scale \
  --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP/idx_position/center-out" \
  --prefix "mlp_center-out_b0.25_0.5_0.75" \
  --batch_size 256