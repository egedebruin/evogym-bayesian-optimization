import concurrent.futures
import learn
from util import start
from configs import config
from robot.active import Brain
from robot.body import Body
from individual import Individual
from itertools import accumulate
import numpy as np
import pickle

LEARN_ITS = 200
INHERIT_LEARN_ITS = 50
INHERIT_SAMPLES = 8

grid1 = np.array([
        [0.0, 0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0],
        [2.0, 2.0, 1.0, 4.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
    ])
grid2 = np.array([
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 4.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [3.0, 3.0, 3.0, 3.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
grid3 = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 2.0, 0.0],
        [1.0, 0.0, 4.0, 2.0, 4.0],
        [3.0, 3.0, 3.0, 3.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
grid4 = np.array([
        [0.0, 0.0, 3.0, 3.0, 0.0],
        [4.0, 0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 4.0, 1.0, 1.0],
        [3.0, 3.0, 3.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
grid5 = np.array([
        [4.0, 2.0, 3.0, 0.0, 0.0],
        [2.0, 3.0, 4.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
grid6 = np.array([
        [2.0, 0.0, 0.0, 3.0, 4.0],
        [3.0, 0.0, 2.0, 1.0, 0.0],
        [3.0, 0.0, 4.0, 4.0, 4.0],
        [3.0, 3.0, 3.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 3.0],
    ])
grid7 = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 2.0, 4.0],
        [3.0, 0.0, 0.0, 4.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
grid8 = np.array([
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [4.0, 0.0, 1.0, 1.0, 0.0],
        [4.0, 0.0, 4.0, 1.0, 0.0],
        [3.0, 3.0, 3.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])

grids = [grid1, grid2, grid3, grid4, grid5, grid6, grid7, grid8]

def run(rng, grid, parent=None, run_learn=True):
    body = Body()
    body.replace_grid(grid)
    brain = Brain(config.GRID_LENGTH, rng)
    individual_id = "0"
    individual = Individual(individual_id, body, brain, 0, [])

    if not run_learn:
        return individual

    if parent is not None:
        individual.inherit_experience([], parent, rng, INHERIT_LEARN_ITS)

    objective_value, best_brain, experience, best_inherited_objective_value, _ = learn.learn(individual, rng)

    individual.add_evaluation(objective_value, best_brain, experience, best_inherited_objective_value)

    return individual

for repetition in range(1, 6):
    results = {}

    rng = start.make_rng_seed()

    # BASE
    config.LEARN_ITERATIONS = INHERIT_LEARN_ITS
    config.INHERIT_SAMPLES = -1
    config.SOCIAL_POOL = 0
    config.INHERIT_TYPE = 'none'

    individual_sets = []
    for grid in grids:
        mutated_individual = run(rng, grid, run_learn=False)
        mutated_individual.mutate(rng)
        individual_sets.append((grid, mutated_individual))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=8
    ) as executor:
        futures = []
        for individual_set in individual_sets:
            grid = individual_set[1].body.grid
            futures.append(executor.submit(run, rng, grid))
    evaluated_mutated_individuals = []
    for future in futures:
        evaluated_mutated_individuals.append(future.result())

    updated_individual_sets = []
    for individual_set in individual_sets:
        for evaluated_individual in evaluated_mutated_individuals:
            if np.array_equal(evaluated_individual.body.grid, individual_set[1].body.grid):
                updated_individual_sets.append((individual_set[0], evaluated_individual))
                continue

    # CHANGE FROM PARENT
    config.LEARN_ITERATIONS = LEARN_ITS
    config.INHERIT_SAMPLES = INHERIT_SAMPLES
    config.SOCIAL_POOL = 1
    config.INHERIT_TYPE = 'parent'

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=8
    ) as executor:
        futures = []
        for individual_set in updated_individual_sets:
            futures.append(executor.submit(run, rng, individual_set[0], individual_set[1]))
    evaluated_original_individuals = []
    for future in futures:
        evaluated_original_individuals.append(future.result())

    results['parent-change'] = [
        list(accumulate((exp[1] for exp in i.experience), max))
        for i in evaluated_original_individuals
    ]


    with open(f"learn_results{repetition}.pkl", "wb") as f:
        pickle.dump(results, f)