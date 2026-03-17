# Single run example

finger_ID="idx" #or mrs
mode="position" #or position


sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
onset_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/start_time_stop_time_all.npy"
label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_0.25_0.5_0.75_shift5.npy"
shift_mask_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/masks/${finger_ID}_${mode}_mask_0.25_0.5_0.75_shift5.npy"


python -m src.RL_decoders.position_decoder \
    --update_W \
    --sbp_path ${sbp_path} \
    --toml_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/config/config_exp.toml" \
    --day_info_path ${day_info_path} \
    --label_path ${label_path} \
    --onset_path ${onset_path} \
    --shift 1 \
    --finger_ID ${finger_ID} \
    --mode "position_W_upd" \
    --shift_mask_path ${shift_mask_path} \
    --slicing_day 90 \
    --target_type "center-out" \