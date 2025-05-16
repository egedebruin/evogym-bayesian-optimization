import json
import os

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv(os.path.join('results', 'nn', 'parameters.csv'))
df['values'] = df['values'].apply(json.loads)
X = np.vstack(df['values'].values)

pca = PCA(n_components=301)  # or however many original features you have
pca.fit(X)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot
import matplotlib.pyplot as plt
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()
exit()

for learn, inherit in [(1, -1), (30, -1), (30, 0), (30, 5)]:
    current_df = df.loc[(df['learn'] == learn) & (df['inherit'] == inherit)]
    print(current_df)
