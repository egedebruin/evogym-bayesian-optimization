import numpy as np
import matplotlib.pyplot as plt
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config

individuals_file = open(config.FOLDER + "individuals.txt", "r")
all_individuals = {individual.split(";")[0]: float(individual.split(";")[5]) for individual in individuals_file.read().splitlines()}

populations_file = open(config.FOLDER + "populations.txt", "r")
generations = populations_file.read().splitlines()

objective_values_per_generation = []
for generation in generations:
    generation_objective_values = []
    for individual in generation.split(";")[:-1]:
        generation_objective_values.append(all_individuals[individual])
    objective_values_per_generation.append(generation_objective_values)

# Convert to numpy array for easy computation
data_array = np.array(objective_values_per_generation)

# Calculate mean and standard deviation along the rows
max_values = np.max(data_array, axis=1)
mean_values = np.mean(data_array, axis=1)
std_values = np.std(data_array, axis=1)

max_so_far = []
previous = 0
for value in max_values:
    best = max(value, previous)
    max_so_far.append(best)
    previous = best
max_so_far = np.array(max_so_far)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(max_so_far, label="Max accumulated", marker='o', color='red')
plt.plot(max_values, label="Max", marker='o', color='blue')
plt.plot(mean_values, label="Mean", marker='o', color='green')
plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2, color='green')
plt.ylim(0, max(max_values) + 5)
plt.xlabel("Generation")
plt.ylabel("Objective value")
plt.legend()
plt.title("Max, mean and Standard Deviation")
plt.show()