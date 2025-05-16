import pandas as pd
from matplotlib import pyplot as plt

results = pd.read_csv('results/nn/environment_tester_results.csv')
groups = results.groupby(['environment', 'robot_id', 'repetition'])
to_plot = {
    'steps': [],
    'simple': [],
    'carry': [],
    'jump': [],
    'climb': []
}
for name, group in groups:
    to_plot[name[0]].append((max(group.loc[group['learn_iteration'] == 0]['objective_value']),
                             max(group['objective_value'])))


fig, ax = plt.subplots(nrows=5)
i = 0
for key, value in to_plot.items():
    min_val = 100
    max_val = -100
    for begin, end in value:
        ax[i].scatter(begin, end, color='black', alpha=0.1)
        min_val = min(min_val, begin, end)
        max_val = max(max_val, begin, end)
    ax[i].set_xlim(min_val - 1, max_val + 1)
    ax[i].set_ylim(min_val - 1, max_val + 1)
    ax[i].set_aspect('equal', adjustable='box')
    i += 1
plt.show()
