import numpy as np
import argparse

def build_trial_ids(trial_bin_used):
    """Reconstructs trial boundaries based on the 0.0 markers."""
    trial_starts = (trial_bin_used == 0.0)
    if len(trial_bin_used) > 0 and (trial_bin_used[0] != 0.0):
        trial_starts[0] = True
    return np.cumsum(trial_starts).astype(int) - 1

def main():
    parser = argparse.ArgumentParser(description="Check how many unique labels exist per trial.")
    parser.add_argument("--label_path", type=str, required=True, 
                        help="Path to your labels.npy file")
    parser.add_argument("--trial_bin_path", type=str, 
                        default="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/all/trial_bin.npy",
                        help="Path to your trial_bin.npy file")
    parser.add_argument("--max_samples", type=int, default=500000, 
                        help="Maximum number of samples to check. Set to 0 to run all.")
    parser.add_argument("--label_mask", type=str, default=None, 
                        help="E.g., '0,2' to drop class 1")
    
    args = parser.parse_args()

    print(f"Loading labels from: {args.label_path}")
    labels = np.load(args.label_path)
    
    print(f"Loading trial bins from: {args.trial_bin_path}")
    trial_bin = np.load(args.trial_bin_path)

    # 1. Instantly shrink the data for a fast check
    if args.max_samples > 0:
        print(f"\nSlicing data to the first {args.max_samples:,} samples for a quick check...")
        labels = labels[:args.max_samples]
        trial_bin = trial_bin[:args.max_samples]

    if len(labels) != len(trial_bin):
        min_len = min(len(labels), len(trial_bin))
        labels = labels[:min_len]
        trial_bin = trial_bin[:min_len]

    # 2. Build trial boundaries on UNMASKED data to preserve the true physics
    print("Building trial boundaries...")
    trial_ids = build_trial_ids(trial_bin)
    
    total_original_trials = len(np.unique(trial_ids))

    # 3. Apply the label mask (Drop Class 1, etc.)
    if args.label_mask is not None:
        if isinstance(args.label_mask, str):
            allowed_labels = [int(x.strip()) for x in args.label_mask.split(",") if x.strip()]
        else:
            allowed_labels = [int(x) for x in "".join(args.label_mask).split(",") if x.strip()]
            
        print(f"Applying label mask. Keeping only classes: {allowed_labels}")
        mask = np.isin(labels.astype(int), allowed_labels)
        
        # Sync the mask to both the labels AND the trial IDs!
        labels = labels[mask]
        trial_ids = trial_ids[mask]

    print("Calculating unique valid labels per trial...")
    
    unique_trials = np.unique(trial_ids)
    label_counts = {} 
    
    for tid in unique_trials:
        trial_labels = labels[trial_ids == tid]
        num_unique_in_trial = len(np.unique(trial_labels))
        
        if num_unique_in_trial not in label_counts:
            label_counts[num_unique_in_trial] = 0
        label_counts[num_unique_in_trial] += 1

    total_valid_trials = len(unique_trials)
    empty_trials = total_original_trials - total_valid_trials
    
    # Print the final report
    print("\n" + "="*50)
    print("                TRIAL LABEL SUMMARY")
    print("="*50)
    print(f"Total original trials detected: {total_original_trials:,}")
    if empty_trials > 0:
        print(f"Trials completely erased by mask: {empty_trials:,} (contained ONLY filtered labels)")
    print(f"Trials analyzed (with valid labels): {total_valid_trials:,}")
    print("-" * 50)
    
    if total_valid_trials > 0:
        for num_unique, count in sorted(label_counts.items()):
            percentage = (count / total_valid_trials) * 100
            print(f"Trials with exactly {num_unique} unique valid label(s): {count:,}  ({percentage:.2f}%)")
    else:
        print("No valid data left after applying the mask!")
        
    print("="*50)

if __name__ == "__main__":
    main()