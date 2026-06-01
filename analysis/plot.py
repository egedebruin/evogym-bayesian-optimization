import numpy as np
import matplotlib.pyplot as plt
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from configs import config

def get_data(folder, max_generations=-1):
    if not os.path.isdir(folder):
        return None
    individuals_path = os.path.join(folder, "individuals.txt")
    if not os.path.isfile(individuals_path):
        return None

    # Read individuals file line by line (memory-safe)
    all_individuals = {}
    with open(individuals_path, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) > 5:
                all_individuals[parts[0]] = float(parts[5])

    populations_path = os.path.join(folder, "populations.txt")
    if not os.path.isfile(populations_path):
        return None

    objective_values_per_generation = []

    # Also stream populations file instead of reading all at once
    with open(populations_path, "r") as f:
        for i, line in enumerate(f, start=1):
            if max_generations != -1 and i > max_generations:
                break

            generation_objective_values = []
            for individual in line.strip().split(";")[:-1]:
                if individual in all_individuals:
                    generation_objective_values.append(all_individuals[individual])

            objective_values_per_generation.append(generation_objective_values)

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