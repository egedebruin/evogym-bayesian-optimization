import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plot  # Your custom module

# Constants
POP_SIZE = 200
EVALS_PER_GEN = 50
REPETITIONS = 10
SUB_FOLDER = 'baseline'

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


def collect_data(inherit, inherit_type, inherit_pool, environment):
    GENERATIONS = 50
    key = (inherit, inherit_type, inherit_pool)

    all_final_fitness = []

    for repetition in range(1, REPETITIONS + 1):
        data_path = f'results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{inherit_type}_pool-{inherit_pool}_environment-{environment}_repetition-{repetition}'
        data_array = plot.get_data(data_path, GENERATIONS)

        if data_array is None or len(data_array) < GENERATIONS:
            print(f'No data for {environment} | {key} | repetition {repetition}')
            continue

        max_vals = np.max(data_array, axis=1)
        running_max = np.maximum.accumulate(max_vals)
        final_fitness = running_max[-1]

        all_final_fitness.append(final_fitness)

    return all_final_fitness


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

    environments = ['simple', 'steps', 'carry', 'catch']
    strategy_keys = list(LABELS.keys())

    fig, axes = plt.subplots(nrows=len(environments), figsize=(12, 10))
    fig.subplots_adjust(top=0.9, right=0.75)  # Leave room on the right for the legend

    for i, env in enumerate(environments):
        ax = axes[i]

        data = []
        box_colors = []
        for key in strategy_keys:
            values = collect_data(*key, env)
            data.append(values)
            box_colors.append(COLORS[key])

        # Make the boxplot with numbers 1 to 8
        positions = np.arange(1, len(strategy_keys) + 1)
        bplot = ax.boxplot(
            data,
            patch_artist=True,
            positions=positions,
            vert=True
        )

        # Color boxes
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)

        ax.set_title(f"Environment: {env.capitalize()}", weight='bold', pad=10)
        ax.set_ylabel("Final Fitness")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(i) for i in positions])

    # Create legend with labels and colors
    legend_handles = []
    for idx, key in enumerate(strategy_keys):
        handle = plt.Line2D(
            [], [], marker='s', color='w', markerfacecolor=COLORS[key], markersize=10,
            label=f"{idx + 1}: {LABELS[key]}"
        )
        legend_handles.append(handle)

    fig.legend(handles=legend_handles, loc='center right', title="Strategies", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Space for legend
    plt.savefig('boxplots_numbered.pdf')


if __name__ == '__main__':
    main()
