python mlp_eval_day.py \
    --results_dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/mlp_day/random/results \
    --glob_pat "mlp_random_b0.2_0.4_0.6_0.8_eliminate-center_hs512-512_seed0_Nclasses4_day*_outputs_day*.npz" \
    --out_dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/mlp_day/random/plots \
    --max_day 100 \
    --x_tick_step 3

#   --results_dir "/path/to/daywise/results" \
#   --glob_pat "*_test_perday_valacc.npz" \
#   --out_dir "/path/to/daywise/plots" \
#   --tag "mlp_daywise_with_firstday_holdout" \
#   --min_seeds 1 \
#   --x_tick_step 3 \
#   --split_npz "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/baseline_perf/hpo/mlp_cv/split_indices.npz" \
#   --weights_dir "/path/to/training_out_dir/weights" \
#   --sbp_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy" \
#   --label_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/idx_position_labels_0.33_0.66_shift0.npy" \
#   --day_info_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy" \
#   --trial_bin_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy" \
#   --slicing_day 1 \
#   --target_type random \
#   --target_style_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy" \
#   --label_mask "0,1,2"