import os

import numpy as np
import pandas as pd

import main
from configs import config
from individual import Individual
from robot.body import Body
from robot.active import Brain
from selection import Selection
from util import start
from util.logger_setup import logger


def get_offspring(population, generation_index, parent_selection, rng: np.random.Generator):
    selected_individuals = parent_selection.select(population, rng)

    offspring = []
    for i, individual in enumerate(selected_individuals):
        new_individual = individual.generate_new_individual(generation_index, i, rng)
        new_individual.inherit_experience(population, individual, rng)
        offspring.append(new_individual)
    return selected_individuals, offspring


STRATEGIES = [
    (-1, 'none', 0),
    (8, 'best', 1),
    (8, 'best', 8),
    (8, 'parent', 1),
    (8, 'random', 1),
    (8, 'random', 8),
    (8, 'similar', 1),
    (8, 'similar', 8)
]

def run(rep):
    config.INHERIT_TYPE = 'none'
    config.INHERIT_SAMPLES = -1
    config.SOCIAL_POOL = 0

    rng = start.make_rng_seed()

    individuals = []
    for i in range(config.POP_SIZE):
        body_size = rng.integers(config.MIN_INITIAL_SIZE, config.MAX_INITIAL_SIZE + 1)
        body = Body(config.GRID_LENGTH, body_size, rng)
        brain = Brain(config.GRID_LENGTH, rng)
        individual_id = f"0-{i}"
        individual = Individual(individual_id, body, brain, 0, [])
        individuals.append(individual)

    population = main.run_generation(individuals, rng)
    parent_selection = Selection(config.OFFSPRING_SIZE, config.PARENT_SELECTION, {'pool_size': config.PARENT_POOL})
    selected_individuals, offspring = get_offspring(population, 1, parent_selection, rng)

    result = {
        'strategy': [],
        'results': [],
    }
    for (samples, inherit_type, pool) in STRATEGIES:
        logger.info(f"Strategy: {samples}, {inherit_type}, {pool}")
        config.INHERIT_TYPE = inherit_type
        config.INHERIT_SAMPLES = samples
        config.SOCIAL_POOL = pool

        for i, individual in enumerate(selected_individuals):
            offspring[i].inherit_experience(population, individual, rng)
        new_population = main.run_generation(offspring, rng)

        objective_values = []
        for individual in new_population:
            objective_values.append((individual.id, individual.objective_value))

        result['strategy'].append(f"{samples},{inherit_type},{pool}")
        result['results'].append(objective_values)

    pd.DataFrame(result).to_csv(os.path.join(config.FOLDER, f'results{rep}.csv'), index=False)

if __name__ == '__main__':
    config.FOLDER = 'results/single_generation_test/'
    if not os.path.exists(config.FOLDER):
        os.makedirs(config.FOLDER)
    main.logger_setup()
    main.set_number_of_sensors()

    for repetition in range(200):
        run(repetition)