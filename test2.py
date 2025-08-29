import pandas as pd
from matplotlib import pyplot as plt

TO_COLOR = {
    True: 'red',
    False: 'blue',
}

df = pd.read_csv('test.csv')

groups = df.groupby('update_policy')

for update_policy, group in groups:
    group['fitness_so_far'] = group.groupby('repetition')['fitness'].cummax()

    summary = group.groupby('learn_iteration')['fitness_so_far'].agg(
        mean='mean',
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75)
    ).reset_index()

    plt.plot(summary['learn_iteration'], summary['mean'], color=TO_COLOR[update_policy])
    plt.fill_between(summary['learn_iteration'], summary['p25'], summary['p75'], alpha=0.2, color=TO_COLOR[update_policy])

plt.show()