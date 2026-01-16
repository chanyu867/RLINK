#data for idx
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/time_data/day_number.npy"
# label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/idx_position_labels_0.25_0.5_0.75.npy"
# label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/idx_direction_3_labels.npy"
onset_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/start_time_stop_time_all.npy"
#data for mrs
# sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/sbp_with_mrs_position_min0.15_max0.85.npy"
# time_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/timestamps_position_min0.15_max0.85.npy"

#setting
finger_ID="idx" #or mrs
mode="position" #or position

for boundary in "0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"; do
    for shift in 0 1 3 5 7 9 11 20 50; do
        label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift${shift}.npy"
        shift_mask_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/masks/${finger_ID}_${mode}_mask_${boundary}_shift${shift}.npy"

        python -m src.RL_decoders.position_decoder \
                --sbp_path ${sbp_path} \
                --day_info_path ${day_info_path} \
                --label_path ${label_path} \
                --onset_path ${onset_path} \
                --shift ${shift} \
                --finger_ID ${finger_ID} \
                --mode ${mode} \
                --shift_mask_path ${shift_mask_path} \
                slicing_day 90 \

    done
done

#original: 
# python -m RL_decoders.RL-decoders-iBMI \
#     --dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/datasets/derived-RL-expt \
#     --expt monkey_1_set_1