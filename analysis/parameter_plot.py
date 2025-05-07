import ast
import os


def get_data(folder):
    if not os.path.isdir(folder):
        return None
    if not os.path.isfile(os.path.join(folder, "individuals.txt")):
        return None
    individuals_file = open(folder + "/individuals.txt", "r")
    all_individuals = [(individual.split(";")[3], float(individual.split(";")[5])) for individual in individuals_file.read().splitlines()]
    all_individuals = [x[0] for x in sorted(all_individuals, key=lambda x: x[1], reverse=True)][:200]
    all_values = []
    i = 0
    for robot_experience in all_individuals:
        i += 1
        robot_experience = ast.literal_eval(robot_experience)
        best_brain = sorted(robot_experience, key=lambda evaluation: float(evaluation[1]), reverse=True)[0][0]
        all_values = all_values + list(best_brain.values())
    return all_values

for learn, inherit in [(30, -1), (30, 0), (30, 5), (1, -1)]:
    values = []
    for repetition in range(1, 21):
        values = values + get_data(f'results/new/learn-{learn}_inherit-{inherit}_repetition-{repetition}')
    print(learn, inherit)
    print(values.count(0.0) / len(values))
    print(values.count(1.0) / len(values))
