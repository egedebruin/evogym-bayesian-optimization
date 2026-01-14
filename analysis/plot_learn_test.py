import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("learn_results.pkl", "rb") as f:
    file_data = pickle.load(f)

# Stack arrays into 2D (rows = arrays, columns = elements)
for name, arrays in file_data.items():
    if len(arrays) == 0:
        continue
    data = np.stack(arrays)

    # Compute mean
    mean = np.mean(data, axis=0)

    # Compute 0.25 and 0.75 quantiles
    q25 = np.quantile(data, 0.25, axis=0)
    q75 = np.quantile(data, 0.75, axis=0)

    # Plot mean line
    plt.plot(mean, label=name, marker='o')

    # Shaded area for 0.25–0.75 quantile
    plt.fill_between(range(len(mean)), q25, q75, alpha=0.2)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Average with 25–75% Shaded Area')
plt.grid(True)
plt.legend()
plt.show()