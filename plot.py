import numpy as np
import matplotlib.pyplot as plt

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

# Plot
plt.figure(figsize=(8, 5))
plt.plot(mean_values, label="Mean", marker='o')
plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2)
plt.plot(max_values, label="Max", marker='o')
plt.ylim(0, max(max_values) + 5)
plt.xlabel("Generation")
plt.ylabel("Objective value")
plt.legend()
plt.title("Max, mean and Standard Deviation")
plt.show()