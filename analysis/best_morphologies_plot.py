import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'best', 1): 'Social learning - Best - N=1',
    (8, 'best', 8): 'Social learning - Best - N=8',
    (8, 'parent', 1): 'Social learning - Parent',
    (8, 'random', 1): 'Social learning - Random - N=1',
    (8, 'random', 8): 'Social learning - Random - N=8',
    (8, 'similar', 1): 'Social learning - Similar - N=1',
    (8, 'similar', 8): 'Social learning - Similar - N=8',
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

df = pd.read_csv("best_morphologies_results.csv")
df = df.loc[df['repetition'] == 1]

groups = df.groupby(['inherit', 'type', 'pool'])
plt.figure(figsize=(8, 5))

for name, group in groups:
    print(name)
    filtered_group = group.drop(['inherit', 'type', 'pool'], axis=1)
    filtered_group['objective_value_max_so_far'] = filtered_group.groupby('experiment_repetition')['objective_value'].cummax()

    # Group by B and aggregate mean, 25th and 75th percentiles
    summary = filtered_group.groupby('learn_iteration')['objective_value_max_so_far'].agg(
        mean='mean',
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75)
    ).reset_index()

    # Plot: mean line with shaded IQR
    plt.plot(summary['learn_iteration'], summary['mean'], label=LABELS[name], color=COLORS[name])
    # plt.fill_between(summary['learn_iteration'], summary['p25'], summary['p75'], alpha=0.3, color=COLORS[name])

plt.xlabel('Learn iteration')
plt.ylabel('Max so far objective-value')
plt.title('Mean aMax so far objective-value by learn iteration averaged over best morphologies per run')
plt.legend()
plt.show()

