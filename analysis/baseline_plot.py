import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import plot

# Constants
POP_SIZE = 100
EVALS_PER_GEN = 50
REPETITIONS = 20
SUB_FOLDER = 'new_paper'

LABELS = {
    (-1, 'none', 0): 'Darwinian',
    (8, 'best', 1): 'Best - N=1',
    (8, 'best', 8): 'Best - N=8',
    (1, 'parent', 1): 'Lamarckian',
    (8, 'parent', 1): 'Lamarckian',
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
    (-1, 'none', 0): '-',
    (8, 'best', 1): ':',
    (8, 'best', 8): ':',
    (1, 'parent', 1): '--',
    (8, 'parent', 1): '--',
    (8, 'random', 1): '-.',
    (8, 'random', 8): '-.',
    (8, 'similar', 1): (0, (3, 1, 1, 1)),
    (8, 'similar', 8): (0, (3, 1, 1, 1)),
}

LEARN_METHOD_TO_LABEL = {
    'bo': 'BO',
    'ppo': 'RL-PPO',
    'ddpg': 'RL'
}

LEARN_METHOD_TO_COLOR = {
    'bo': 'green',
    'ppo': 'blue',
    'ddpg': 'red'
}


def make_the_plot(inherit, inherit_type, inherit_pool, environment, ax, learn_method):
    def smooth(x, w):
        out = np.zeros_like(x, dtype=float)
        half = w // 2
        for i in range(len(x)):
            start = max(0, i - half)
            end = min(len(x), i + half + 1)
            out[i] = np.mean(x[start:end])
        return out

    extra = f"_changing-{environment}"
    # if environment[0] != 0:
    #     extra += f"_minmutation-{environment[0]}"
    # extra += f"_maxmutation-{environment[1]}"
    # extra = f"_changedegree-{environment}"

    GENERATIONS = 100
    key = (inherit, inherit_type, inherit_pool)
    # label = LABELS.get(key)
    # color = COLORS.get(key)
    label = f'{LEARN_METHOD_TO_LABEL.get(learn_method)}, {LABELS.get(key)}'
    color = LEARN_METHOD_TO_COLOR.get(learn_method)

    if label is None or color is None:
        return  # skip unknown combinations

    curves = []
    for repetition in range(1, REPETITIONS + 1):
        data_path = f'results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{inherit_type}_pool-{inherit_pool}_environment-changing_method-{learn_method}{extra}_repetition-{repetition}'
        data_array = plot.get_data(data_path, GENERATIONS)

        if data_array is None:
            continue
        if len(data_array) < GENERATIONS:
            print(f'No data found for {environment}, {inherit}, {inherit_type}, {inherit_pool}, {repetition}')
            print(len(data_array)) if data_array is not None else print("None")
            print()
            continue

        mean_vals = np.nanmean(data_array, axis=1)
        running_max = np.maximum.accumulate(mean_vals)
        curves.append(mean_vals)

    if not curves:
        return

    curves = np.array(curves)

    mean_vals = np.median(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)
    x_vals = np.arange(1, GENERATIONS + 1) * EVALS_PER_GEN * POP_SIZE

    window = 4

    mean_vals = smooth(mean_vals, window)
    q25 = smooth(q25, window)
    q75 = smooth(q75, window)
    x_vals = smooth(x_vals, window)

    label = f'{LEARN_METHOD_TO_LABEL.get(learn_method)}{LABELS.get(key)}'
    result_dict = {
        'cat': [label for _ in range(50)],
        'x': [x*2 for x in range(50)],
        'y': mean_vals,
        'ymin': q25,
        'ymax': q75
    }
    pd.DataFrame(result_dict).to_csv(f'{environment}-{label}.txt', index=False, sep='\t')

    ax.plot(x_vals, mean_vals, label=label, color=color, linestyle=LINE_STYLES.get(key))
    ax.fill_between(x_vals, q25, q75, color=color, alpha=0.2)


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

    environments = ['1e-06', '0.1', '0.2', '0.4', '1']
    strategy_keys = list(LABELS.keys())  # 6 total strategies

    fig, axes = plt.subplots(nrows=max(2, len(environments)), figsize=(10, 10), sharey=False)
    fig.subplots_adjust(top=0.85, right=0.78)  # Make space for legend

    for i, env in enumerate(environments):
        ax = axes[i]
        for idx, key in enumerate(strategy_keys):
            for learn_method in ['ddpg', 'bo']:
                make_the_plot(*key, env, ax, learn_method)
        name = 'Changing-0' if env == 'changing-1e-07' else (
            'Bidirectional with direction sensor' if env == 'bidirectional2' else (
            'Bidirectional without direction sensor' if env == 'bidirectional' else env.capitalize()
            )
        )
        ax.set_title(f"Environment: {name}", weight='bold', pad=15)
        ax.set_xlabel("Evaluations")
        if i == 0:
            ax.set_ylabel("Fitness")

    # Global legend outside plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title="Strategy", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave room for legend
    # plt.show()
    # plt.savefig(f'plot.pdf')


if __name__ == '__main__':
    main()
