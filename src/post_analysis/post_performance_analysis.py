import numpy as np
import argparse
import os

from src.post_analysis.data_analysis import *
from src.post_analysis.figures_plot import *
from src.utils.npy_loader import npy_loader

# ------------------------- #
# Example usage: python -m src.post_analysis.post_performance_analysis --post_analysis --gamma 0.1 --transition_only
# ------------------------- #

parser = argparse.ArgumentParser(description="Post-analysis of model performance")
parser.add_argument("--path_to_prediction", type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/pred_results/run_eliminated1_with_hpo_results_Mar18th")
parser.add_argument("--path_to_day_num", type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy")
parser.add_argument("--path_to_trial_bin", type=str, default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy")
parser.add_argument("--post_analysis", action='store_true')
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--transition_only", action='store_true', help="Plot only trials containing all classes")

args = parser.parse_args()
class_task_table = {2: "0.5"}
models_to_analyze = ["DQN"]

if args.post_analysis:
    for shift in [0]:
        for num_class, task in class_task_table.items():
            print(f"\n{'#'*80}\nAnalyzing shift={shift}, task={task} ({num_class} classes)\n{'#'*80}")

            # 1. Load Raw Data
            raw_day_number = npy_loader(args.path_to_day_num)
            raw_time_within_trial = npy_loader(args.path_to_trial_bin)

            # 2. Extract `master_indices` to sync arrays
            master_indices = None
            for temp_model in models_to_analyze:
                pattern = f"{args.path_to_prediction}/{num_class}classes/results_idx_{temp_model}_shift{shift}_seed*_gamma*.npy"
                temp_paths = filter_result_paths_by_gamma(pattern, args.gamma)
                
                if len(temp_paths) > 0:
                    sample_result = np.load(temp_paths[0], allow_pickle=True).item()
                    print(f"key in sample_result: {list(sample_result.keys())}")
                    if "master_indices" in sample_result:
                        master_indices = sample_result["master_indices"]
                        print(f"[*] Found master_indices in {os.path.basename(temp_paths[0])}")
                        break
            
            if master_indices is None:
                print(f"[WARN] Could not find any result files containing 'master_indices' for shift {shift}.")
                continue

            # 3. Shrink arrays
            day_number = raw_day_number[master_indices]
            time_within_trial = raw_time_within_trial[master_indices]
            new_trial_starts = np.insert(np.diff(time_within_trial) < 0, 0, True)
            time_within_trial[new_trial_starts] = 0.0
            print(f"[*] Data perfectly synced! Final length: {len(day_number)} samples.")

            # Dictionary to store paths for the joint Event-Aligned Recovery Plot
            model_paths_dict = {}

            # 4. Run Analysis Per Model
            for model in models_to_analyze:
                pattern = f"{args.path_to_prediction}/{num_class}classes/results_idx_{model}_shift{shift}_seed*_gamma*.npy"
                res_paths = filter_result_paths_by_gamma(pattern, args.gamma)
                print(f"Currently loading: {model}")

                if len(res_paths) == 0:
                    print(f"[WARN] No result files found for {model}")
                    continue

                # Save path for joint plotting later
                model_paths_dict[model] = res_paths

                # A. Daywise Trial Blocks Plot
                collected_seeds = plot_daywise_trial_blocks( 
                    res_paths, day_number, time_within_trial, block=30, do_plot=False, 
                    plot_transition_trials_only=args.transition_only 
                )

                # B. Metrics
                res = spearman_top_quantile(collected_seeds, model=model, delta_thr=0.01)
                out = stability_metrics_from_collected(collected_seeds, model=model, include_boundary=False)
                
                mean_perf = np.mean([d["acc_block_mean"] for c in collected_seeds for d in c.get(model, [])])
                
                perf_trend = daywise_first_second_half_means(collected_seeds, model, ratio=0.5)
                first_half_means = [v["first_half_mean"] for v in perf_trend.values()]
                second_half_means = [v["second_half_mean"] for v in perf_trend.values()]

                # Print Stats
                print(f"\n{'-'*60}")
                print(f"Model: {model} | #files: {len(res_paths)}")
                print(f"  Correlation (ρ):                      {res['rho']:>8.4f}")
                print(f"  Degrade Probability:                  {out['p_drop']:>8.4f}")
                print(f"  Variance:                             {out['variance']:>8.4f}")
                print(f"  Mean Performance:                     {mean_perf:>8.4f}")
                print(f"  1st Half vs 2nd Half:                 {np.mean(first_half_means):.4f} : {np.mean(second_half_means):.4f}")
            
            # 5. Execute Event-Aligned Recovery Plot across ALL models
            print(f"\n[*] Generating Event-Aligned Recovery Curve for all models...")
            plot_event_aligned_recovery(model_paths_dict, time_within_trial, pre_window=3)

            # 6. Execute Continuous F1-Score Plot
            print(f"[*] Generating Continuous F1-Score Plot...")
            metric = "accuracy"  # Change to "f1" if you want F1-Score instead
            plot_continuous_performance(model_paths_dict, day_number, metric="accuracy", resolution=1000) # <-- ADD THIS LINE