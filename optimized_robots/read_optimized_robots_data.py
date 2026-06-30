import pickle
import numpy as np

def main():
    result_lists = []
    for i in range(1, 6):
        try:
            with open(f"optimized_robots/results{i}.pkl", "rb") as f:
                result_list = pickle.load(f)
                result_lists.append(result_list)
        except FileNotFoundError:
            print(f"Warning: optimized_robots/results{i}.pkl not found.")

    changes_to_skip = [0]

    changes_to_body_darwinian = {
        0: [],
        1: [],
        2: [],
        3: [],
        5: [],
        10: []
    }

    changes_to_body_lamarckian = {
        0: [],
        1: [],
        2: [],
        3: [],
        5: [],
        10: []
    }

    for result_list in result_lists:
        for entry in result_list:
            ch = entry['changes']
            if ch in changes_to_skip:
                continue

            if entry['lamarckian']:
                changes_to_body_lamarckian[ch].append(entry['qualities'])
            else:
                changes_to_body_darwinian[ch].append(entry['qualities'])

    # Sort keys to ensure consistent output
    sorted_keys = sorted([k for k in changes_to_body_darwinian.keys() if k not in changes_to_skip])
    
    for key in sorted_keys:
        darwinian_values = changes_to_body_darwinian[key]
        lamarckian_values = changes_to_body_lamarckian[key]
        
        if not darwinian_values or not lamarckian_values:
            continue
        
        # Truncate to min generations and min repetitions to allow numpy operations
        min_gens = min(min(len(v) for v in darwinian_values), min(len(v) for v in lamarckian_values))
        min_reps = min(len(darwinian_values), len(lamarckian_values))
        
        d_arr = np.array([v[:min_gens] for v in darwinian_values[:min_reps]])
        l_arr = np.array([v[:min_gens] for v in lamarckian_values[:min_reps]])
        
        # Difference: Lamarckian - Darwinian
        diff_arr = l_arr - d_arr
        
        # Compute stats: y is median, ymin is 25th percentile, ymax is 75th percentile
        y_values = np.median(diff_arr, axis=0)
        ymin_values = np.percentile(diff_arr, 25, axis=0)
        ymax_values = np.percentile(diff_arr, 75, axis=0)
        x_values = np.arange(len(y_values))
        
        cat = str(key)
        output_filename = f"optimized_robots/{cat}.txt"
        with open(output_filename, "w") as f:
            # Header: cat, x, y, ymin, ymax (tab separated)
            f.write("cat\tx\ty\tymin\tymax\n")
            for x, y, ymin, ymax in zip(x_values, y_values, ymin_values, ymax_values):
                f.write(f"{cat}\t{x}\t{y}\t{ymin}\t{ymax}\n")
        print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    main()
