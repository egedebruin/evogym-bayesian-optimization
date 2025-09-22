import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import ListedColormap

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from analysis import body_descriptors


LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'parent', 1): 'Parent',
    (8, 'best', 1): 'Best N=1',
    (8, 'best', 8): 'Best N=8',
    (8, 'similar', 1): 'Similar - N=1',
    (8, 'similar', 8): 'Similar - N=8',
    (8, 'random', 1): 'Random - N=1',
    (8, 'random', 8): 'Random - N=8',
}

EVALS_PER_GEN = 50
ENVIRONMENTS = ['simple', 'steps', 'carry', 'catch']
strategy_keys = list(LABELS.keys())
matrices_dict = {}

for key in strategy_keys:
    label = LABELS[key]
    matrices_dict[label] = []
    for environment in ENVIRONMENTS:

        best_grid = None
        best_value = 0
        for repetition in range(1, 21):
            folder = f'results/learn-{EVALS_PER_GEN}_inherit-{key[0]}_type-{key[1]}_pool-{key[2]}_environment-{environment}_repetition-{repetition}/'

            best_individual = body_descriptors.get_best_individual(folder)
            if not best_individual:
                continue
            if float(best_individual[5]) < best_value:
                continue
            best_grid = ast.literal_eval(best_individual[1])
            best_value = float(best_individual[5])
        matrices_dict[label].append(best_grid)

# Plot all strategies in a grid
for matrices in matrices_dict.values():
    result = ""
    for matrix in matrices:
        if matrix is None:
            continue
        result += '\\vsrevogym{5}{5}{'
        for row in matrix:
            for col in row:
                result += str(int(col))
            result += '-'
        result = result[:-1] + '} &'
    print(result[:-1] + "\\\\")
