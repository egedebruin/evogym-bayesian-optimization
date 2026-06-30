import pickle

import numpy as np
import matplotlib.pyplot as plt

with open("results/journal/simple-delta-results-from-scratch.pkl", "rb") as f:
    results_1 = pickle.load(f)

results_2 = None
#
# with open("simple-delta-results-with-inheritance-4.pkl", "rb") as f:
#     results_2 = pickle.load(f)

lamarckian = []
darwinian = []
for i, inherit in enumerate(results_1['inherit']):
    if inherit == 1:
        lamarckian.append(i)
    else:
        darwinian.append(i)

lamarckian_data = np.concatenate(
    [np.array([
        results_1['values'][i][:-1]
        for i in lamarckian
    ])] +
    ([] if results_2 is None else [np.array([
        results_2['values'][i][:-1]
        for i in lamarckian
    ])]),
    axis=0
)

darwinian_data = np.concatenate(
    [np.array([
        results_1['values'][i][:-1]
        for i in darwinian
    ])] +
    ([] if results_2 is None else [np.array([
        results_2['values'][i][:-1]
        for i in darwinian
    ])]),
    axis=0
)

lamarckian_data = np.maximum.accumulate(lamarckian_data, axis=1)

lamarckian_mean_values = np.mean(lamarckian_data, axis=0)
lamarckian_p25 = np.percentile(lamarckian_data, 25, axis=0)
lamarckian_p75 = np.percentile(lamarckian_data, 75, axis=0)
x = np.arange(len(lamarckian_mean_values))

plt.plot(x, lamarckian_mean_values, label="Lamarckian")
plt.fill_between(x, lamarckian_p25, lamarckian_p75, alpha=0.3)

darwinian_data = np.maximum.accumulate(darwinian_data, axis=1)

darwinian_mean_values = np.mean(darwinian_data, axis=0)
darwinian_p25 = np.percentile(darwinian_data, 25, axis=0)
darwinian_p75 = np.percentile(darwinian_data, 75, axis=0)
x = np.arange(len(darwinian_mean_values))

plt.plot(x, darwinian_mean_values, label="Darwinian")
plt.fill_between(x, darwinian_p25, darwinian_p75, alpha=0.3)

plt.title("Lifetime learning after 50 generations on Bidirectional without sensor")
plt.xlabel("Learn iteration")
plt.ylabel("Fitness")
plt.legend()
plt.show()