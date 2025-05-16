import ast
import json
import os
from collections import defaultdict
import pandas as pd


def get_data(folder, total_generations):
    if not os.path.isdir(folder):
        return None
    if not os.path.isfile(os.path.join(folder, "individuals.txt")):
        return None
    all_values = defaultdict(list)
    previous_generation = -1
    with open(folder + "/individuals.txt", "r") as file:
        for line in file:
            robot_id, robot_experience = (line.split(";")[0], line.split(";")[3])
            generation = int(robot_id.split("-")[0])
            if generation % int(total_generations/66) != 0:
                continue
            if generation != previous_generation:
                previous_generation = generation
                print("Generation {}".format(generation))
            robot_experience = ast.literal_eval(robot_experience)
            best_brain = sorted(robot_experience, key=lambda evaluation: float(evaluation[1]), reverse=True)[0][0]
            all_values[generation].append(list(best_brain.values()))

    return all_values

result = {
    'learn': [],
    'inherit': [],
    'repetition': [],
    'generation': [],
    'robot_id': [],
    'values': []
}

for learn, inherit, generations in [(1, -1, 2000), (30, -1, 66), (30, 0, 66), (30, 5, 66), ]:
    for repetition in range(1, 21):
        print("Repetition:", repetition)
        data = get_data(f'results/nn/learn-{learn}_inherit-{inherit}_repetition-{repetition}', generations)
        for key, robots in data.items():
            for i, values in enumerate(robots):
                result['learn'].append(learn)
                result['inherit'].append(inherit)
                result['repetition'].append(repetition)
                result['generation'].append(key)
                result['robot_id'].append(i)
                result['values'].append(values)

df = pd.DataFrame(result)
df['values'] = df['values'].apply(json.dumps)
df.to_csv('test.csv', index=False)