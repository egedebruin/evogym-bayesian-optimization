import ast
import pickle

import numpy as np

from configs import config
from robot.body import Body
from robot.active import Brain
from individual import Individual


def get_population():
    # Step 1: Load generation info first
    with open(config.FOLDER + "populations.txt", "r") as f:
        generations = f.read().splitlines()
    final_generation_ids = set(generations[-1].split(";")[:-1])

    # Step 2: Only store individuals needed for final generation
    all_individuals = {}
    with open(config.FOLDER + "individuals.txt", "r") as f:
        for line in f:
            if (parts := line.strip().split(";")) and len(parts) >= 7:
                ind_id = parts[0]
                if ind_id in final_generation_ids:
                    all_individuals[ind_id] = tuple(parts[1:7])  # ✅ Only store relevant individuals

    # Step 3: Load all experience (still full file — can be optimized similarly if needed)
    with open(config.FOLDER + "experience.pkl", "rb") as file:
        ids, experiences = pickle.load(file)

    all_experience = dict(zip(ids, experiences))

    population = []
    for individual_id in final_generation_ids:
        body_grid = np.array(ast.literal_eval(all_individuals[individual_id][0]))
        brain_string = all_individuals[individual_id][1]
        best_brain = ast.literal_eval(all_individuals[individual_id][2])
        parent_id = str(all_individuals[individual_id][3])
        objective_value = float(all_individuals[individual_id][4])
        original_generation = int(all_individuals[individual_id][5])

        experience = all_experience[individual_id]

        body = Body()
        body.replace_grid(body_grid)
        brain = Brain()
        brain.replace_parameters(brain_string)

        individual = Individual(individual_id, body, brain, original_generation)
        individual.add_restart_values(objective_value, best_brain, experience, parent_id)
        population.append(individual)
    return population, len(generations)

def get_rng():
    loaded_state = np.load(config.FOLDER + "rng_state.npy", allow_pickle=True).item()

    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = loaded_state
    return rng

def get_heights():
    with open(config.FOLDER + "environments.txt", "r") as f:
        generations = f.read().splitlines()
    final_heights = [int(height) for height in generations[-1].split(";")]
    return final_heights