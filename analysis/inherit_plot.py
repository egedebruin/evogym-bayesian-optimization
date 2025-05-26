import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plot  # Assumes you have a module `plot` with `get_data`

# Constants
POP_SIZE = 200
EVALS_PER_GEN = 30
REPETITIONS = 20
GENERATIONS = 80

LABELS = {
    (-1, 1, False): 'Evolution only',
    (-1, 1, True): 'Evolution only - Random',
    (-1, 30, False): 'Learn - No inheritance',
    (0, 30, False): 'Learn - Inherit samples',
    (5, 30, False): 'Learn - Reevaluate'
}

COLORS = {
    (-1, 1, False): '#e41a1c',  # red
    (-1, 1, True): '#000000',  # black
    (-1, 30, False): '#377eb8',  # blue
    (0, 30, False): '#4daf4a',  # green
    (5, 30, False): '#ff7f00'   # orange
}


def make_the_plot(inherit, environment, ax, max_x, max_y):
    print(f"Plotting: inherit={inherit}, environment={environment}")

    curves = []
    for repetition in range(1, REPETITIONS + 1):
        data_path = f'results/learn-30_inherit-{inherit}_environment-{environment}_repetition-{repetition}'
        data_array = plot.get_data(data_path, GENERATIONS)

        if data_array is None or len(data_array) < GENERATIONS:
            print(f"  Skipping: incomplete data (rep {repetition})")
            continue

        max_vals = np.max(data_array, axis=1)
        running_max = np.maximum.accumulate(max_vals)
        curves.append(running_max)

    if not curves:
        print(f"  No data to plot for inherit={inherit}, env={environment}")
        return

    x_vals = np.arange(1, GENERATIONS + 1) * EVALS_PER_GEN * POP_SIZE
    curves = np.array(curves)

    mean_vals = np.mean(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)

    label = LABELS.get((inherit, 30, False), f"Inherit {inherit}")
    color = COLORS.get((inherit, 30, False), 'gray')

    ax.plot(x_vals, mean_vals, label=label, color=color)
    ax.fill_between(x_vals, q25, q75, color=color, alpha=0.2)

    max_x.append(x_vals[-1])
    max_y.append(np.max(q75) * 1.1)


def main():
    # Style
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

    environments = ['carry', 'catch']
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6.5), sharey=True)
    fig.subplots_adjust(top=0.85)
    max_x, max_y = [], []

    for i, env in enumerate(environments):
        ax = axes[i]
        for inherit in [-1, 0, 5]:
            make_the_plot(inherit, env, ax, max_x, max_y)

        ax.set_xlim(0, max(max_x))
        ax.set_ylim(0, max(max_y))
        ax.set_xlabel("Function evaluations")
        if i == 0:
            ax.set_ylabel("Objective value")

        # Add nice title for each subplot
        ax.set_title(f"Environment: {env.capitalize()}", weight='bold', pad=15)

        # Plot appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.grid(True, which='major', linestyle='--', alpha=0.4)
        ax.set_facecolor('#f9f9f9')
        ax.legend(frameon=False, title="Strategy", loc="lower right")

    # Global title
    fig.suptitle("Max Performance over Generations", fontsize=18, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.patch.set_facecolor('white')
    plt.savefig("plot.pdf")
    plt.show()


if __name__ == '__main__':
    main()
