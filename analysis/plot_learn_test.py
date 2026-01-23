import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

name_to_label = {
    'random': 'Random',
    'individual': 'IL',
    'parent-same': 'SL: Same',
    'parent-change': 'Parent-Mutated',
}

name_to_color = {
    'random': 'red',
    'individual': 'blue',
    'parent-same': 'green',
    'parent-change': 'black',
}

def merge_dicts(dicts):
    merged = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            merged[key].extend(value)
    return dict(merged)


def plot_grouped_data(file_data):
    """
    key_filter: function(name) -> bool
        Returns True for keys that should be plotted.
    """
    for name, arrays in file_data.items():
        if name not in name_to_label.keys():
            continue
        data = np.stack(arrays)

        mean = np.mean(data, axis=0)
        q25 = np.quantile(data, 0.25, axis=0)
        q75 = np.quantile(data, 0.75, axis=0)

        # result_file_data = {
        #     'cat': [name_to_label[name] for _ in range(len(mean))],
        #     'x': [i for i in range(1, 1 + len(mean))],
        #     'y': list(mean),
        #     'ymin': list(q25),
        #     'ymax': list(q75),
        # }
        # df = pd.DataFrame(result_file_data)
        # df.to_csv(f"{name}.txt", '\t', index=False)

        plt.plot(mean, label=name_to_label[name], color=name_to_color[name])
        plt.fill_between(range(len(mean)), q25, q75, alpha=0.2, color=name_to_color[name])

    plt.xlabel("Learning iteration")
    plt.ylabel("Quality")
    plt.title("Title")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()


# ---- Load data --------------------------------------------------------------

all_data = []
for i in range(1, 6):
    with open(f"results/gecco2026/learn_results_change{i}.pkl", "rb") as f:
        all_data.append(pickle.load(f))
for i in range(1, 6):
    with open(f"results/gecco2026/learn_results{i}.pkl", "rb") as f:
        all_data.append(pickle.load(f))

file_data = merge_dicts(all_data)

# ---- Plot without "change" --------------------------------------------------

plot_grouped_data(
    file_data
)