import numpy as np
import matplotlib.pyplot as plt
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from configs import config

def get_data(folder, max_generations = -1):
    if not os.path.isdir(folder):
        return None
    if not os.path.isfile(os.path.join(folder, "individuals.txt")):
        return None
    individuals_file = open(folder + "/individuals.txt", "r")
    all_individuals = {individual.split(";")[0]: float(individual.split(";")[5]) for individual in individuals_file.read().splitlines()}

    populations_file = open(folder + "/populations.txt", "r")
    generations = populations_file.read().splitlines()

    objective_values_per_generation = []
    i = 1
    for generation in generations:
        if max_generations != -1 and i > max_generations:
            break
        i += 1
        generation_objective_values = []
        for individual in generation.split(";")[:-1]:
            generation_objective_values.append(all_individuals[individual])
        objective_values_per_generation.append(generation_objective_values)

    # Convert to numpy array for easy computation
    return np.array(objective_values_per_generation)

def main():
    data_array = get_data(config.FOLDER)

    # Calculate mean and standard deviation along the rows
    max_values = np.max(data_array, axis=1)
    min_values = np.min(data_array, axis=1)
    mean_values = np.mean(data_array, axis=1)
    q25 = np.percentile(data_array, 25, axis=1)
    q75 = np.percentile(data_array, 75, axis=1)

    max_so_far = []
    previous = 0
    for value in max_values:
        best = max(value, previous)
        max_so_far.append(best)
        previous = best
    max_so_far = np.array(max_so_far)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(max_so_far, label="Max so far", marker='o', color='red')
    plt.plot(max_values, label="Max", marker='o', color='blue')
    plt.plot(mean_values, label="Mean", marker='o', color='green')
    plt.fill_between(range(len(mean_values)), q25, q75, alpha=0.2, color='green')
    plt.ylim(min(min(min_values) - 5, 0), max(max_values) + 5)
    plt.xlabel("Generation")
    plt.ylabel("Objective value")
    plt.legend()
    plt.title("Max, mean and standard deviation")
    plt.show()

if __name__ == '__main__':
    main()