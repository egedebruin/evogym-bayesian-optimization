import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import plot  # Your custom module

# Set global styles for beauty
sns.set_theme(style="whitegrid")
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5
})

# Plot setup
fig, ax = plt.subplots(figsize=(11, 6.5))

pop_size = 100
to_label = {
    (-1, 1): 'Evolution only',
    (-1, 30): 'Learn - No inheritance',
    (0, 30): 'Learn - Inherit samples',
    (5, 30): 'Learn - Reevaluate'
}
to_color = {
    (-1, 1): '#e41a1c',   # red
    (-1, 30): '#377eb8',  # blue
    (0, 30): '#4daf4a',   # green
    (5, 30): '#ff7f00'    # orange
}

max_y = []

for inherit in [-1, 0, 5]:
    for generations, learn in [(1000, 1), (33, 30)]:
        if learn == 1 and inherit != -1:
            continue
        print(f"Plotting for inherit {inherit} and learn {learn}")

        to_plot = []
        for repetition in range(1, 21):
            data_array = plot.get_data(f'results/learn-{learn}_inherit-{inherit}_repetition-{repetition}')
            if data_array is None or len(data_array) < generations:
                print("Incomplete data for:", learn, inherit, repetition)
                continue

            max_values = np.max(data_array, axis=1)
            max_so_far = np.maximum.accumulate(max_values)
            to_plot.append(max_so_far)

        if len(to_plot) == 0:
            continue

        function_evals = np.arange(1, generations + 1) * learn * pop_size
        mean_vals = np.mean(to_plot, axis=0)
        q25 = np.percentile(to_plot, 25, axis=0)
        q75 = np.percentile(to_plot, 75, axis=0)

        label = to_label.get((inherit, learn), f"Inherit {inherit}, Learn {learn}")
        color = to_color.get((inherit, learn), 'gray')

        ax.plot(function_evals, mean_vals, label=label, color=color)
        ax.fill_between(function_evals, q25, q75, color=color, alpha=0.2)
        max_y.append(np.max(q75) * 1.1)

# Prettification
ax.set_ylim(0, max(max_y))
ax.set_xlabel("Function evaluations")
ax.set_ylabel("Objective value")
ax.set_title("Max performance averaged over repetitions", pad=20, weight='bold')

# Style tweaks
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.grid(True, which='major', linestyle='--', alpha=0.4)
ax.legend(frameon=False, title="Strategy", loc="upper left")

# Add a light background panel
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')
fig.tight_layout()
plt.show()
