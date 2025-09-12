import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import plot  # Assumes you have a module `plot` with `get_data`

# Constants
POP_SIZE = 200
EVALS_PER_GEN = 50
REPETITIONS = 20
SUB_FOLDER = 'baseline'

LABELS = {
    (-1, 'none', 0): 'Individual',
    (8, 'best', 1): 'Best - N=1',
    (8, 'best', 8): 'Best - N=8',
    (8, 'parent', 1): 'Parent',
    (8, 'random', 1): 'Random - N=1',
    (8, 'random', 8): 'Random - N=8',
    (8, 'similar', 1): 'Similar - N=1',
    (8, 'similar', 8): 'Similar - N=8',
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

LINE_STYLES = {
    (-1, 'none', 0): '--',
    (8, 'best', 1): ':',
    (8, 'best', 8): ':',
    (8, 'parent', 1): '-',
    (8, 'random', 1): '-.',
    (8, 'random', 8): '-.',
    (8, 'similar', 1): (0, (3, 1, 1, 1)),
    (8, 'similar', 8): (0, (3, 1, 1, 1)),
}


def make_the_plot(inherit, inherit_type, inherit_pool, environment, ax):
    GENERATIONS = 50
    key = (inherit, inherit_type, inherit_pool)
    label = LABELS.get(key)
    color = COLORS.get(key)

    if label is None or color is None:
        return  # skip unknown combinations

    curves = []
    for repetition in range(1, REPETITIONS + 1):
        data_path = f'results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{inherit_type}_pool-{inherit_pool}_environment-{environment}_repetition-{repetition}'
        data_array = plot.get_data(data_path, GENERATIONS)

        if data_array is None or len(data_array) < GENERATIONS:
            print(f'No data found for {environment}, {inherit}, {inherit_type}, {inherit_pool}, {repetition}')
            print(len(data_array)) if data_array is not None else print("None")
            print()
            continue

        max_vals = np.max(data_array, axis=1)
        running_max = np.maximum.accumulate(max_vals)
        curves.append(max_vals)

    if not curves:
        return

    x_vals = np.arange(1, GENERATIONS + 1) * EVALS_PER_GEN * POP_SIZE
    curves = np.array(curves)

    mean_vals = np.mean(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)

    ax.plot(x_vals, mean_vals, label=label, color=color, linestyle=LINE_STYLES.get(key))
    ax.fill_between(x_vals, q25, q75, color=color, alpha=0.2)

    csv_data = {'cat': [LABELS[key] for _ in range(50)], 'x': range(50), 'y': np.mean(curves, axis=0),
                'ymin': np.percentile(curves, 25, axis=0),
                'ymax': np.percentile(curves, 25, axis=0)}
    pd.DataFrame(csv_data).to_csv(f'performance-line-{environment}-{key[1]}-{key[2]}.txt', index=False, sep='\t')


def main():
    # Style
    sns.set_theme(style="whitegrid")
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.2
    })

    environments = ['catch']
    strategy_keys = list(LABELS.keys())  # 6 total strategies

    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), sharey=False)
    fig.subplots_adjust(top=0.85, right=0.78)  # Make space for legend

    for i, env in enumerate(environments):
        ax = axes[i]
        for idx, key in enumerate(strategy_keys):
            make_the_plot(*key, env, ax)

        ax.set_title(f"Environment: {env.capitalize()}", weight='bold', pad=15)
        ax.set_xlabel("Evaluations")
        if i == 0:
            ax.set_ylabel("Fitness")

    # Global legend outside plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title="Strategy", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave room for legend
    plt.show()
    # plt.savefig(f'plot.pdf')


if __name__ == '__main__':
    main()
