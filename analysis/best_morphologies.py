import ast
import concurrent.futures
import math
import os
import sys

import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from configs import config
from robot.active import Brain
from robot.active import Controller
from robot.sensors import Sensors
from robot.brain_nn import BrainNN
from util import start, world

LABELS = {
    (-1, 'none', 0): 'Individual learning',
    (8, 'parent', 1): 'Social learning - Parent',
    (8, 'best', 1): 'Social learning - Best - N=1',
    (8, 'best', 8): 'Social learning - Best - N=8',
    (8, 'random', 1): 'Social learning - Random - N=1',
    (8, 'random', 8): 'Social learning - Random - N=8',
    (8, 'similar', 1): 'Social learning - Similar - N=1',
    (8, 'similar', 8): 'Social learning - Similar - N=8',
}

SUB_FOLDER = 'baseline'
EVALS_PER_GEN = 50
ENVIRONMENT = 'carry'
REPS = 3

def main():
    config.ENVIRONMENT = ENVIRONMENT

    if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
        BrainNN.NUMBER_OF_INPUT_NEURONS = BrainNN.NUMBER_OF_INPUT_NEURONS + 2

    result_dict = {
        'original_environment': [],
        'inherit': [],
        'type': [],
        'pool': [],
        'repetition': [],
        'experiment_repetition': [],
        'learn_iteration': [],
        'objective_value': []
    }

    strategy_keys = list(LABELS.keys())

    for r in range(REPS):
        for key in strategy_keys:
            result = parallelize(key[0], key[1], key[2], 800)
            for repetition in range(len(result)):
                for learn_iteration in range(len(result[repetition][0])):
                    result_dict['original_environment'].append(result[repetition][1])
                    result_dict['inherit'].append(key[0])
                    result_dict['type'].append(key[1])
                    result_dict['pool'].append(key[2])
                    result_dict['repetition'].append(r + 1)
                    result_dict['experiment_repetition'].append(repetition + 1)
                    result_dict['learn_iteration'].append(learn_iteration + 1)
                    result_dict['objective_value'].append(result[repetition][0][learn_iteration])

    pd.DataFrame(result_dict).to_csv(f'results/{ENVIRONMENT}.csv', index=False)

def parallelize(inherit, i_type, pool, learn_iterations):
    grids = []
    for environment in ['simple', 'steps', 'carry', 'catch']:
        for repetition in range(1, 21):
            folder = f'results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{i_type}_pool-{pool}_environment-{environment}_repetition-{repetition}/'
            best_fitness = float('-inf')
            best_grid = None

            with open(folder + "/individuals.txt", "r") as file:
                for line in file:
                    parts = line.strip().split(";")
                    try:
                        fitness = float(parts[5])
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_grid = np.array(ast.literal_eval(parts[1]))
                    except (ValueError, IndexError, SyntaxError):
                        continue  # skip malformed lines

            if best_grid is not None:
                grids.append((best_grid, environment))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=80
    ) as executor:
        futures = []
        for grid, environment in grids:
            futures.append(executor.submit(learn, grid, learn_iterations, environment))

    result = []
    for future in futures:
        result.append(future.result())
    return result

def learn(grid, learn_iterations, original_environment):
    rng = start.make_rng_seed()

    brain = Brain(config.GRID_LENGTH, rng)

    sim, viewer = world.build_world(grid, rng)
    actuator_indices = sim.get_actuator_indices('robot')
    optimizer = BayesianOptimization(
        f=None,
        pbounds=brain.get_p_bounds(actuator_indices),
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2 ** 32)),
        acquisition_function=acquisition.UpperConfidenceBound(kappa=config.LEARN_KAPPA,
                                                              random_state=int(
                                                                  rng.integers(low=0, high=2 ** 32)))
    )
    optimizer.set_gp_params(
        kernel=Matern(nu=config.LEARN_NU, length_scale=config.LEARN_LENGTH_SCALE, length_scale_bounds="fixed"))
    optimizer.set_gp_params(alpha=config.LEARN_ALPHA)

    objective_value = -math.inf
    objective_values = []
    for bayesian_optimization_iteration in range(learn_iterations):
        print(f"Learn generation {bayesian_optimization_iteration + 1}")
        if bayesian_optimization_iteration == 0:
            next_point = brain.to_next_point(actuator_indices)
        else:
            next_point = optimizer.suggest()

        args = brain.next_point_to_controller_values(next_point, actuator_indices)
        controller = Controller(args)
        sensors = Sensors(grid)

        result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, True)
        objective_values.append(result)

        if result > objective_value:
            objective_value = result

        optimizer.register(params=next_point, target=result)
    sim.reset()
    viewer.close()
    return objective_values, original_environment

if __name__ == '__main__':
    main()