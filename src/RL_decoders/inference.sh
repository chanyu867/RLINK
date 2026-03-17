#data for idx
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
onset_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/start_time_stop_time_all.npy"
toml_directory="src/RL_decoders/config"
#data for mrs
# sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/sbp_with_mrs_position_min0.15_max0.85.npy"
# time_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/timestamps_position_min0.15_max0.85.npy"

#setting
finger_ID="idx" #or mrs
mode="position" #or position

for config_file in "config_exp.toml"; do
    toml_path="${toml_directory}/${config_file}"
    # for boundary in "0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"; do
    for boundary in "0.33_0.66"; do
        # for shift in 0 1 3 5 7 9 11 20 50; do
        for shift in 0; do
            label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift${shift}.npy"
            shift_mask_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/masks/${finger_ID}_${mode}_mask_${boundary}_shift${shift}.npy"

            python -m src.RL_decoders.position_decoder \
                    --sbp_path ${sbp_path} \
                    --day_info_path ${day_info_path} \
                    --label_path ${label_path} \
                    --onset_path ${onset_path} \
                    --toml_path ${toml_path} \
                    --shift ${shift} \
                    --finger_ID ${finger_ID} \
                    --mode "run_eliminated1_with_hpo_results" \
                    --shift_mask_path ${shift_mask_path} \
                    --slicing_day 1 \
                    --best_params_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/HPO" \
                    --hpo_mode "all_class_trials" \
                    --target_type "random" \
                    --label_mask "0,2" \
                    --n_lags 16 \
                    --lag_step 4 \
                    --short_run \
                    --best_params_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/HPO" \
                    2>&1 | tee -a "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/logs/inference.log"

            # --short_run \
            # --best_params_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/HPO" \
            # --hpo_mode "whole" \
            

        done
    done
done

#original: 
# python -m RL_decoders.RL-decoders-iBMI \
#     --dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/datasets/derived-RL-expt \
#     --expt monkey_1_set_1