import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def get_final_values(filename):
    """
    Loads a pickle file and extracts the final fitness for each individual,
    split by inheritance type.
    """
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
        
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # data['values'] is a list of trajectories.
    # We take the max of each trajectory (excluding the last element if present, 
    # to match the [:-1] logic from the original script).
    vals = np.array([np.max(v[:-1]) for v in data['values']])
    inherit = np.array(data['inherit'])
    
    lamarckian = vals[inherit == 1]
    darwinian = vals[inherit == -1]
    
    return lamarckian, darwinian

# Define the files for each category
categories = {
    "simple": ["simple-delta-results-from-scratch.pkl"],
    "changing": ["changing-delta-results-from-scratch.pkl"],
    "bidirectional": [
        "bidirectional-delta-results-from-scratch-1.pkl",
        "bidirectional-delta-results-from-scratch-2.pkl"
    ],
    "bidirectional2": [
        "bidirectional2-delta-results-from-scratch-1.pkl",
        "bidirectional2-delta-results-from-scratch-2.pkl"
    ]
}

lamarckian_to_plot = []
darwinian_to_plot = []
labels = []

for label, files in categories.items():
    if len(files) == 1:
        l, d = get_final_values(files[0])
        if l is not None:
            lamarckian_to_plot.append(l)
            darwinian_to_plot.append(d)
            labels.append(label)
    else:
        # For bidirectional and bidirectional2, take the average of the two versions
        l1, d1 = get_final_values(files[0])
        l2, d2 = get_final_values(files[1])
        
        # Average Lamarckian
        if l1 is not None and l2 is not None:
            n = min(len(l1), len(l2))
            lam_avg = (l1[:n] + l2[:n]) / 2.0
            lamarckian_to_plot.append(lam_avg)
        elif l1 is not None:
            lamarckian_to_plot.append(l1)
        elif l2 is not None:
            lamarckian_to_plot.append(l2)
        else:
            lamarckian_to_plot.append(np.array([]))

        # Average Darwinian
        if d1 is not None and d2 is not None:
            n = min(len(d1), len(d2))
            dar_avg = (d1[:n] + d2[:n]) / 2.0
            darwinian_to_plot.append(dar_avg)
        elif d1 is not None:
            darwinian_to_plot.append(d1)
        elif d2 is not None:
            darwinian_to_plot.append(d2)
        else:
            darwinian_to_plot.append(np.array([]))
            
        labels.append(label)

    # Save to .txt file
    l_vals = lamarckian_to_plot[-1]
    d_vals = darwinian_to_plot[-1]
    with open(f"{label}.txt", "w") as f:
        f.write("Darwinian\tLamarckian\n")
        # Ensure they are the same length (already handled for bidirectional)
        n = min(len(l_vals), len(d_vals))
        for i in range(n):
            f.write(f"{d_vals[i]}\t{l_vals[i]}\n")
    print(f"Saved data for {label} to {label}.txt")

# Create the box plot
plt.figure(figsize=(12, 7))

num_cats = len(labels)
positions = np.arange(num_cats)
width = 0.35

all_data = []
all_positions = []
box_colors = []

for i in range(num_cats):
    # Lamarckian box
    all_data.append(lamarckian_to_plot[i])
    all_positions.append(positions[i] - width/2)
    box_colors.append('lightblue')
    
    # Darwinian box
    all_data.append(darwinian_to_plot[i])
    all_positions.append(positions[i] + width/2)
    box_colors.append('lightcoral')

bplot = plt.boxplot(all_data, positions=all_positions, widths=width, patch_artist=True)

# Color the boxes
for patch, color in zip(bplot['boxes'], box_colors):
    patch.set_facecolor(color)

plt.xticks(positions, labels)
plt.title("Final Fitness Comparison: Lamarckian vs Darwinian (From Scratch)")
plt.ylabel("Fitness")
plt.xlabel("Environment")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Statistical testing and annotation
for i in range(num_cats):
    l_vals = lamarckian_to_plot[i]
    d_vals = darwinian_to_plot[i]
    
    if len(l_vals) > 0 and len(d_vals) > 0:
        # Perform Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(l_vals, d_vals, alternative='two-sided')
        
        # Determine annotation position
        # Get the max value in the current category to place the label above it
        max_l = np.max(l_vals) if len(l_vals) > 0 else -np.inf
        max_d = np.max(d_vals) if len(d_vals) > 0 else -np.inf
        y_max = max(max_l, max_d)
        
        # Add some vertical offset
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.05
        # If ylim is not yet settled, we might need to be careful or use a fixed percentage of data range
        # Let's use a small fraction of the current y-axis range if it's already set, 
        # or just wait until after plotting all data.
        
        # To be safe, we can do this after the boxplot call and before plt.show()
        # Since we are already after the boxplot call here.
        
        p_text = f"p = {p_val:.4e}" if p_val < 0.001 else f"p = {p_val:.3f}"
        
        # Draw a small line between boxes and the p-value
        x1, x2 = positions[i] - width/2, positions[i] + width/2
        y, h = y_max + offset, offset * 0.2
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y+h, p_text, ha='center', va='bottom', color='k', fontsize=10, fontweight='bold')

# Ensure there's space for annotations
plt.ylim(top=plt.ylim()[1] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', label='Lamarckian'),
    Patch(facecolor='lightcoral', label='Darwinian')
]
plt.legend(handles=legend_elements)

# Adjust layout and show/save
plt.tight_layout()
# plt.savefig("boxplots_final_generations.png") # Uncomment to save
plt.show()
