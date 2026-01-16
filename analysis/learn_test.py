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

def run(rng, grid, parent=None):
    body = Body()
    body.replace_grid(grid)
    brain = Brain(config.GRID_LENGTH, rng)
    individual_id = "0"
    individual = Individual(individual_id, body, brain, 0, [])
    if parent is not None:
        individual.inherit_experience([], parent, rng, INHERIT_LEARN_ITS)

    objective_value, best_brain, experience, best_inherited_objective_value, _ = learn.learn(individual, rng)

    individual.add_evaluation(objective_value, best_brain, experience, best_inherited_objective_value)

    return individual

for repetition in range(2, 6):
    results = {}

    rng = start.make_rng_seed()

    config.LEARN_ITERATIONS = LEARN_ITS
    for random_learning in [False, True]:
        # BASE
        config.RANDOM_LEARNING = random_learning
        config.INHERIT_SAMPLES = -1
        config.SOCIAL_POOL = 0
        config.INHERIT_TYPE = 'none'

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=8
        ) as executor:
            futures = []
            for grid in grids:
                futures.append(executor.submit(run, rng, grid))
        original_individuals = []
        for future in futures:
            original_individuals.append(future.result())

        if not random_learning:
            results['individual'] = [
                list(accumulate((exp[1] for exp in i.experience), max))
                for i in original_individuals
            ]
        if random_learning:
            results['random'] = [
                list(accumulate((exp[1] for exp in i.experience), max))
                for i in original_individuals
            ]
            continue

        # SAME AS PARENT
        for individual in original_individuals:
            individual.experience = individual.experience[:INHERIT_LEARN_ITS]

        config.INHERIT_SAMPLES = INHERIT_SAMPLES
        config.SOCIAL_POOL = 1
        config.INHERIT_TYPE = 'parent'

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=8
        ) as executor:
            futures = []
            for individual in original_individuals:
                grid = individual.body.grid
                futures.append(executor.submit(run, rng, grid, individual))
        same_individuals = []
        for future in futures:
            same_individuals.append(future.result())

        results['parent-same'] = [
            list(accumulate((exp[1] for exp in i.experience), max))
            for i in same_individuals
        ]

        # CHANGE FROM PARENT
        for individual in original_individuals:
            individual.body.mutate(rng)

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=8
        ) as executor:
            futures = []
            for individual in original_individuals:
                grid = individual.body.grid
                futures.append(executor.submit(run, rng, grid, individual))
        change_individuals = []
        for future in futures:
            change_individuals.append(future.result())

        results['parent-change-inherit'] = [
            list(accumulate((exp[1] for exp in i.experience), max))
            for i in change_individuals
        ]

        # NO INHERITANCE
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=8
        ) as executor:
            futures = []
            for individual in original_individuals:
                grid = individual.body.grid
                futures.append(executor.submit(run, rng, grid))
        change_individuals = []
        for future in futures:
            change_individuals.append(future.result())

        results['parent-change-individual'] = [
            list(accumulate((exp[1] for exp in i.experience), max))
            for i in change_individuals
        ]


    with open(f"learn_results{repetition}.pkl", "wb") as f:
        pickle.dump(results, f)