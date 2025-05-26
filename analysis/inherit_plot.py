import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import plot

pop_size = 200
to_label = {
    (-1, 1, False): 'Evolution only',
    (-1, 1, True): 'Evolution only - Random',
    (-1, 30, False): 'Learn - No inheritance',
    (0, 30, False): 'Learn - Inherit samples',
    (5, 30, False): 'Learn - Reevaluate'
}
to_color = {
    (-1, 1, False): '#e41a1c',  # red
    (-1, 1, True): '#000000',  # black
    (-1, 30, False): '#377eb8',  # blue
    (0, 30, False): '#4daf4a',  # green
    (5, 30, False): '#ff7f00'  # orange
}

def make_the_plot(inherit, generations, environment, ax, max_x, max_y):
    print(f"Plotting for inherit {inherit}")

    to_plot = []
    for repetition in range(1, 21):
        data_array = plot.get_data(f'results/learn-30_inherit-{inherit}_environment-{environment}_repetition-{repetition}',
                                   generations)
        if data_array is None or len(data_array) < generations:
            print("Incomplete data for:", inherit, environment, repetition,
                  len(data_array) if data_array is not None else "None")
            continue

        max_values = np.max(data_array, axis=1)
        max_so_far = np.maximum.accumulate(max_values)
        to_plot.append(max_so_far)

    if len(to_plot) == 0:
        return

    function_evals = np.arange(1, generations + 1) * 30 * pop_size
    x = function_evals
    mean_vals = np.mean(to_plot, axis=0)
    q25 = np.percentile(to_plot, 25, axis=0)
    q75 = np.percentile(to_plot, 75, axis=0)

    label = to_label.get((inherit, 30, False), f"Inherit {inherit}")
    color = to_color.get((inherit, 30, False), 'gray')

    ax.plot(x, mean_vals, label=label, color=color)
    ax.fill_between(x, q25, q75, color=color, alpha=0.2)
    max_x.append(np.max(x))
    max_y.append(np.max(q75) * 1.1)

def main():
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
    fig, ax = plt.subplots(ncols=3, figsize=(11, 6.5))

    max_x = []
    max_y = []

    generations = 80
    for i, environment in enumerate(['simple', 'carry', 'catch']):
        for inherit in [-1, 0, 5]:
            make_the_plot(inherit, generations, environment, ax[i], max_x, max_y)

        # Prettification
        ax[i].set_xlim(0, min(max_x))
        ax[i].set_ylim(0, max(max_y))
        ax[i].set_xlabel("Function evaluations")
        ax[i].set_ylabel("Objective value")
        ax[i].set_title("Max performance averaged over repetitions", pad=20, weight='bold')

        # Style tweaks
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_linewidth(1.2)
        ax[i].spines["bottom"].set_linewidth(1.2)

        ax[i].grid(True, which='major', linestyle='--', alpha=0.4)
        ax[i].legend(frameon=False, title="Strategy", loc="lower right")

        # Add a light background panel
        ax[i].set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    plt.savefig('plot.pdf')

if __name__ == '__main__':
    main()