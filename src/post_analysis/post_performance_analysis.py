import numpy as np
from src.post_analysis.data_analysis import *
from src.post_analysis.figures_plot import *
from src.utils.npy_loader import npy_loader
import argparse
# ------------------------- #
# Example usage
# ------------------------- #

parser = argparse.ArgumentParser(description="Post-analysis of model performance")
parser.add_argument(
    "--path_to_prediction",
    type=str,
    default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/pred_results/position",
    help="Path to prediction results directory"
)
parser.add_argument(
    "--path_to_day_num",
    type=str,
    default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/day_number.npy",
    help="Path to day number npy file"
)
parser.add_argument(
    "--path_to_trial_bin",
    type=str,
    default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy",
    help="Path to trial bin npy file"
)
parser.add_argument(
    "--path_to_mask",
    type=str,
    default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/classes_dir/masks",
    help="Path to masks directory"
)
parser.add_argument(
    "--post_analysis",
    type=bool,
    default=True,
    help="Whether to run post-analysis"
)
parser.add_argument(
    "--gamma_mode",
    type=bool,
    default=True,
    help="Whether to use gamma mode"
)

args = parser.parse_args()


path_to_prediction = args.path_to_prediction
path_to_day_num = args.path_to_day_num
path_to_trial_bin = args.path_to_trial_bin
path_to_mask = args.path_to_mask

class_task_table = {
    3: "0.33_0.66",
    # 4: "0.25_0.5_0.75",
    # 5: "0.2_0.4_0.6_0.8"
}

post_analysis = args.post_analysis
gamma_mode = args.gamma_mode

# results_npy_paths: list of saved result dicts (one per decoder), e.g.
if post_analysis:
    for shift in [0]:  # need to done for 0 shifting
        for num_class, task in class_task_table.items():

            # load once per (shift, task) because mask depends on these
            day_number = npy_loader(path_to_day_num)
            time_within_trial = npy_loader(path_to_trial_bin)

            # mask for shifting
            if shift != 0:
                shift_mask_path = f"{path_to_mask}/idx_position_mask_{task}_shift{shift}.npy"
                shift_mask = npy_loader(shift_mask_path).astype(bool)

                # (optional) keep if you use it elsewhere; not used below
                count_1 = time_within_trial[time_within_trial == 0.0]

                day_number = day_number[shift_mask]
                time_within_trial = time_within_trial[shift_mask]

            # mask for day_slicing
            day_mask = day_number>90
            time_within_trial = time_within_trial[day_mask]
            day_number = day_number[day_mask]

            # print("shapes checking: ", day_number.shape, time_within_trial.shape) #(7594816,) (7594816,)

            for model in ["banditron", "banditronRP", "HRL", "AGREL"]:

                # --- NEW: collect multiple seeds (all matching files) ---
                pattern = f"{path_to_prediction}/{num_class}classes/results_idx_{model}_shift{shift}_seed*_gamma*.npy"
                res_paths = filter_result_paths_by_gamma(pattern, gamma_mode)
                # print(f"Found {len(res_paths)} result files: {res_paths}")

                if len(res_paths) == 0:
                    print(f"[WARN] No result files found: {pattern}")
                    continue

                # ------- precise analysis ------- #
                #1. split all performance based on trial and given block
                collected_seeds = plot_daywise_trial_blocks( 
                    res_paths, day_number, time_within_trial, block=20, do_plot=False
                )

                #2. calculate spearman correlation between performance and position change
                res = spearman_top_quantile(collected_seeds, model=model, delta_thr=0.01)
                

                #3. calculate several stability matrics
                out = stability_metrics_from_collected(collected_seeds, model=model, include_boundary=False)

                #4. mean performance across all days
                mean_perf = np.mean([
                    d["acc_block_mean"]
                    for c in collected_seeds          # loop over seeds
                    for d in c.get(model, [])         # rows for this model in each seed
                ])

                #5. first half vs second half performance trend
                perf_trend = daywise_first_second_half_means(collected_seeds, model, ratio=0.5)
                first_half_means = [v["first_half_mean"] for v in perf_trend.values()]
                second_half_means = [v["second_half_mean"] for v in perf_trend.values()]

                # # --- print summary ---
                print(f"\n{'='*70}")
                print(f"Model: {model} | Temporal shift: {shift} | decoding task: {task} | #seeds(files): {len(res_paths)}")
                # print(f"{'='*70}")
                print(f"  Correlation (ρ):                                {res['rho']:>8.4f}")
                print(f"  Degrade Probability:                            {out['p_drop']:>8.4f}")
                print(f"  Variance:                                       {out['variance']:>8.4f}")
                print(f"  Mean Performance:                               {mean_perf:>8.4f}")
                print(f"  First Half perf. : Second Half perf.:           {np.mean(first_half_means):>8.4f} : {np.mean(second_half_means):>8.4f}")
            # print(f"{'='*70}\n")



# visualizations
results_dir = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/pred_results/position/3classes"

for shift in [0, 1, 3, 5, 7, 9, 11]:
    plot_4models_daily_mean_std(results_dir, finger="idx", shift=0, models=["banditron","banditronRP","HRL","AGREL"], gamma_mode=False)
    #when you turn off the exploration, banditron and banditronRP give the same results as the only thing they differ is exploration, so changing seeds is no longer matter!
    # task: fix colors for  models

# out_mp4_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/video/video_no_explore.mp4"
# save_shift_video_mp4(results_dir, out_mp4_path, finger="idx", shifts=(0,1,3,5,7,9,11),
#                          models=("banditron","banditronRP","HRL","AGREL"),
#                          ylim=(0,1), secs_per_frame=0.5, gamma_mode=gamma_mode)