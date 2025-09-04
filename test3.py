import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_str = open('test.txt').read()

# Convert to dictionary
data = []
for line in data_str.strip().splitlines():
    key, value = line.split(":")
    data.append(float(value.strip()))

values = np.array(data)

# Basic statistics
stats = {
    "count": len(values),
    "mean": np.mean(values),
    "median": np.median(values),
    "std_dev": np.std(values, ddof=1),
    "min": np.min(values),
    "max": np.max(values),
    "25th_percentile": np.percentile(values, 25),
    "75th_percentile": np.percentile(values, 75)
}

print(stats)

# Histogram
plt.figure(figsize=(10, 5))
plt.hist(values, bins=20, edgecolor='black')
plt.title("Histogram of Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

# Rolling mean and std (window=10)
rolling_mean = pd.Series(values).rolling(window=10).mean()
rolling_std = pd.Series(values).rolling(window=10).std()

plt.figure(figsize=(12, 5))
plt.plot(values, label="Values", alpha=0.6)
plt.plot(rolling_mean, label="Rolling Mean (window=10)", color='red')
plt.plot(rolling_std, label="Rolling Std (window=10)", color='green')
plt.title("Rolling Mean and Standard Deviation (window=10)")
plt.xlabel("Index")
plt.ylabel("Value / Rolling Stats")
plt.legend()
plt.grid(alpha=0.5)
plt.show()
