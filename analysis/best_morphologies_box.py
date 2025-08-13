import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'best', 1): 'Social learning - Best - N=1',
    (8, 'best', 8): 'Social learning - Best - N=8',
    (8, 'parent', 1): 'Social learning - Parent',
    (8, 'random', 1): 'Social learning - Random - N=1',
    (8, 'random', 8): 'Social learning - Random - N=8',
    (8, 'similar', 1): 'Social learning - Similar - N=1',
    (8, 'similar', 8): 'Social learning - Similar - N=8',
}

COLORS = {
    (-1, 'none', 0): 'red',
    (8, 'best', 1): 'black',
    (8, 'best', 8): 'grey',
    (8, 'parent', 1): 'orange',
    (8, 'random', 1): 'blue',
    (8, 'random', 8): 'cyan',
    (8, 'similar', 1): 'purple',
    (8, 'similar', 8): 'pink',
}

# Load and preprocess
df = pd.read_csv("results/main/carry.csv")
df['experiment_repetition'] = (df['experiment_repetition'] - 1) % 20 + 1

env_order = ['simple', 'steps', 'carry', 'catch']
data = []
colors = []
positions = []

legend_entries = {}
xtick_labels = []

position_counter = 0
spacer = 1  # space between environments
group_width = 0.8

# Collect plot data
env_centers = []
for env in env_order:
    env_df = df[df['original_environment'] == env]
    strategies = env_df.groupby(['inherit', 'type', 'pool'])

    group_positions = []
    for strat_key, strat_df in strategies:
        label = LABELS[strat_key]
        color = COLORS[strat_key]

        strat_df['objective_value_max_so_far'] = strat_df.groupby('experiment_repetition')['objective_value'].cummax()
        final_vals = strat_df.groupby('experiment_repetition')['objective_value_max_so_far'].last()

        data.append(final_vals)
        colors.append(color)
        positions.append(position_counter)
        group_positions.append(position_counter)

        legend_entries[label] = color
        position_counter += group_width

    # Store center of group for labeling
    center = np.mean(group_positions)
    env_centers.append((center, env))

    position_counter += spacer  # space after each env group

# Plot
plt.figure(figsize=(max(12, 0.4 * len(data)), 6))
box = plt.boxplot(data, patch_artist=True, positions=positions)

# Color boxes
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Remove x-tick labels
plt.xticks(ticks=positions, labels=[""] * len(positions))
plt.ylabel('Final max objective-value')
plt.title('Final max objective-value per strategy, grouped by environment')
plt.grid(True, linestyle='--', alpha=0.4)

# Add environment group labels below x-axis
ymin, ymax = plt.ylim()
for xpos, env_label in env_centers:
    plt.text(xpos, ymin - 0.05 * (ymax - ymin), env_label,
             ha='center', va='top', fontsize=10, fontweight='bold')

# Optional: Add legend
handles = [Patch(facecolor=color, label=label, alpha=0.6) for label, color in legend_entries.items()]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Strategy")

# Adjust layout to make room for bottom labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()
