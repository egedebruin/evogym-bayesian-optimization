import pandas as pd
from matplotlib import pyplot as plt

TO_COLOR = {
    ('bo', 'DDPG'): 'red',
    ('rl', 'DDPG'): 'blue',
    ('rl', 'PPO'): 'black',
    ('borl', 'DDPG'): 'green',
}

df = pd.read_csv('results.csv')

groups = df.groupby(['strategy', 'rl_type'])

for experiment, group in groups:

    group['fitness_so_far'] = group.groupby('repetition')['fitness'].cummax()

    # Plot each repetition as an individual line
    for rep, rep_data in group.groupby('repetition'):
        plt.plot(
            rep_data['learn_iteration'],
            rep_data['fitness'],
            color=TO_COLOR[experiment],
        )

plt.xlabel('Learning Iteration')
plt.ylabel('Fitness so far')
plt.title('All Repetition Runs')
plt.show()
