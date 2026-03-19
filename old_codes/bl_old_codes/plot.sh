boundary="0.5" #"0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8"

# python /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/baseline_models/performance_by_labels.py \
#   --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP/idx_position" \
#   --label_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/idx_position_labels_${boundary}_shift0.npy" \
#   --day_info_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy" \
#   --target_style_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy" \
#   --target_type "center-out" \
#   --boundary ${boundary} \
#   --slicing_day 1 \
# --epochs 2000 \

python /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/baseline_models/performance_by_labels.py \
  --out_dir "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/MLP/idx_position_test_1dim_trials" \
  --label_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/idx_position_labels_${boundary}_shift0.npy" \
  --day_info_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy" \
  --target_style_path "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy" \
  --target_type "random" \
  --boundary ${boundary} \
  --slicing_day 1 \