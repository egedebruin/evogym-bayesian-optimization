import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import ListedColormap

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from analysis import run_best


def plot_matrices_grid(matrices_dict, save_path=None):
    """
    Plots a grid of matrices where rows are strategies and columns are repetitions.

    Parameters:
    - matrices_dict: Dictionary where keys are strategy labels, values are lists of 5x5 matrices.
    - save_path: Optional path to save the image.
    """
    colors = ['white', 'black', 'grey', 'orange', 'blue']
    cmap = ListedColormap(colors)

    strategy_labels = list(matrices_dict.keys())
    num_strategies = len(strategy_labels)
    num_repetitions = max(len(matrices_dict[label]) for label in strategy_labels)

    fig, axes = plt.subplots(num_strategies, num_repetitions, figsize=(num_repetitions * 2.5, num_strategies * 2.5))

    if num_strategies == 1:
        axes = np.array([axes])
    if num_repetitions == 1:
        axes = axes[:, np.newaxis]

    for row_idx, label in enumerate(strategy_labels):
        repetitions = matrices_dict[label]
        for col_idx in range(num_repetitions):
            ax = axes[row_idx, col_idx]
            if col_idx < len(repetitions):
                ax.imshow(repetitions[col_idx], cmap=cmap, vmin=0, vmax=4)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add strategy label on the left of each row
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=80, va='center')

            # Add repetition title at the top of each column
            if row_idx == 0:
                ax.set_title(f'Repetition {col_idx + 1}', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(f'morphologies.pdf')

# =============================
# Load the grids for all strategies and repetitions
# =============================

LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'parent', 1): 'Social learning - Parent',
    (8, 'best', 1): 'Social learning - Best - N=1',
    (8, 'best', 8): 'Social learning - Best - N=8',
    (8, 'random', 1): 'Social learning - Random - N=1',
    (8, 'random', 8): 'Social learning - Random - N=8',
    (8, 'similar', 1): 'Social learning - Similar - N=1',
    (8, 'similar', 8): 'Social learning - Similar - N=8',
}

SUB_FOLDER = 'baseline'
EVALS_PER_GEN = 50
ENVIRONMENT = 'simple'

strategy_keys = list(LABELS.keys())
matrices_dict = {}

for key in strategy_keys:
    label = LABELS[key]
    matrices_dict[label] = []

    for repetition in range(1, 21):
        folder = f'results/{SUB_FOLDER}/learn-{EVALS_PER_GEN}_inherit-{key[0]}_type-{key[1]}_pool-{key[2]}_environment-{ENVIRONMENT}_repetition-{repetition}/'
        print(f'Loading: {folder}')

        best_individual = run_best.get_best_individual(folder)
        grid = ast.literal_eval(best_individual[1])
        matrices_dict[label].append(grid)

# Plot all strategies in a grid
plot_matrices_grid(matrices_dict)
