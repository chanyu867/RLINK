finger_ID="idx"        # or mrs
mode="position"

sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/sbp_all.npy"
day_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy"
target_style_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/target_style.npy"

log_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/logs/Perceptron/inference_Perceptron.log"

# choose slicing days here
slicing_days=(10 30 50 90 150)   # <-- edit as you like

# boundaries
boundaries=("0.33_0.66" "0.25_0.5_0.75" "0.2_0.4_0.6_0.8")

# target types
target_types=("center-out" "random")

for target_type in "${target_types[@]}"; do
  out_dir="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Results/Perceptron/${finger_ID}_${mode}/${target_type}"
  mkdir -p "${out_dir}"

  for slicing_day in "${slicing_days[@]}"; do
    for boundary in "${boundaries[@]}"; do

      label_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/${finger_ID}_${mode}_labels_${boundary}_shift0.npy"

      prefix="perceptron_${finger_ID}_${mode}_${target_type}_day${slicing_day}_b${boundary}"

      echo "=== RUN target_type=${target_type} slicing_day=${slicing_day} boundary=${boundary} ===" | tee -a "${log_path}"

      python src/baseline_models/perceptron_finger_decoder.py \
        --sbp_path "${sbp_path}" \
        --label_path "${label_path}" \
        --day_info_path "${day_info_path}" \
        --slicing_day "${slicing_day}" \
        --target_type "${target_type}" \
        --target_style_path "${target_style_path}" \
        --max_iter 2000 \
        --scale \
        --out_dir "${out_dir}" \
        --prefix "${prefix}" \
        2>&1 | tee -a "${log_path}"

    done
  done
done
