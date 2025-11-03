import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

LINE_STYLES = {
    'simple': '--',
    'steps': ':',
    'carry': '-',
    'catch': '-.',
}

df = pd.read_csv("simple.csv")

df['experiment_repetition'] = (df['experiment_repetition'] - 1) % 20 + 1

groups = df.groupby(['inherit', 'type', 'pool'])
plt.figure(figsize=(8, 5))

for name, group in groups:
    print(name)
    filtered_group = group.drop(['inherit', 'type', 'pool'], axis=1)

    nested_groups = filtered_group.groupby(['original_environment'])

    for nested_name, nested_group in nested_groups:
        nested_group['objective_value_max_so_far'] = nested_group.groupby('experiment_repetition')[
            'objective_value'].cummax()

        # Group by B and aggregate mean, 25th and 75th percentiles
        summary = nested_group.groupby('learn_iteration')['objective_value_max_so_far'].agg(
            mean='mean',
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75)
        ).reset_index()

        # Plot: mean line with shaded IQR
        plt.plot(summary['learn_iteration'], summary['mean'], label=LABELS[name], color=COLORS[name], linestyle=LINE_STYLES[nested_name[0]])
        # plt.fill_between(summary['learn_iteration'], summary['p25'], summary['p75'], alpha=0.3, color=COLORS[name])

plt.xlabel('Learn iteration')
plt.ylabel('Max so far objective-value')
plt.title('Mean max so far objective-value by learn iteration averaged over best morphologies')
# Create unique legend handles for labels (only once per name)
label_handles = []
seen_labels = set()
for name in LABELS:
    if LABELS[name] not in seen_labels:
        handle = Line2D([0], [0], color=COLORS[name], linestyle='-', label=LABELS[name])
        label_handles.append(handle)
        seen_labels.add(LABELS[name])

# Create unique legend handles for line styles (e.g., to show what each linestyle means)
# linestyle_handles = []
# seen_styles = set()
# for style_key, linestyle in LINE_STYLES.items():
#     if linestyle not in seen_styles:
#         handle = Line2D([0], [0], color='black', linestyle=linestyle, label=style_key)
#         linestyle_handles.append(handle)
#         seen_styles.add(linestyle)

# Combine both sets of handles
plt.legend(handles=label_handles, loc='best')
plt.show()

