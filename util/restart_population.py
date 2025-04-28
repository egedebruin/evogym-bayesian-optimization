import ast

import numpy as np

from configs import config
from robot.body import Body
from robot.active import Brain
from individual import Individual


def get_population():
    individuals_file = open(config.FOLDER + "individuals.txt", "r")
    all_individuals = {
        parts[0]: tuple(parts[1:7])
        for line in individuals_file.read().splitlines()
        if (parts := line.split(";")) and len(parts) >= 7
    }

    populations_file = open(config.FOLDER + "populations.txt", "r")
    generations = populations_file.read().splitlines()

    population = []
    for individual_id in generations[-1].split(";")[:-1]:
        body_grid = np.array(ast.literal_eval(all_individuals[individual_id][0]))
        brain_string = all_individuals[individual_id][1]
        experience = ast.literal_eval(all_individuals[individual_id][2])
        parent_id = str(all_individuals[individual_id][3])
        objective_value = float(all_individuals[individual_id][4])
        original_generation = int(all_individuals[individual_id][5])

        body = Body()
        body.replace_grid(body_grid)
        brain = Brain()
        brain.replace_parameters(brain_string)

        individual = Individual(individual_id, body, brain, original_generation)
        individual.add_restart_values(objective_value, experience, parent_id)
        population.append(individual)
    return population, len(generations) - 1

def get_rng():
    loaded_state = np.load(config.FOLDER + "rng_state.npy", allow_pickle=True).item()

    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = loaded_state
    return rng