import ast
import concurrent.futures
import math

import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
import pandas as pd

from configs import config
from robot.active import Brain
from robot.active import Controller
from robot.sensors import Sensors
from util import start, world

def main():
    rng = start.make_rng_seed()
    result_dict = {
        'n_learn': [],
        'inherit': [],
        'repetition': [],
        'objective_value': []
    }

    for n_learn, inherit in [(1, -1), (30, -1), (30, 0), (30, 5)]:
        result = parallelize(n_learn, inherit, 500, rng)
        for repetition in range(len(result)):
            result_dict['n_learn'].append(n_learn)
            result_dict['inherit'].append(inherit)
            result_dict['repetition'].append(repetition + 1)
            result_dict['objective_value'].append(result[repetition])

    pd.DataFrame(result_dict).to_csv('../results/best_morphologies_results.csv', index=False)

def parallelize(n_learn, inherit, learn_iterations, rng):
    grids = []
    for repetition in range(1, 21):
        print(f"Repetition {repetition}")
        folder = f'results/new/learn-{n_learn}_inherit-{inherit}_repetition-{repetition}'
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
            grids.append(best_grid)

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=20
    ) as executor:
        futures = []
        for grid in grids:
            futures.append(executor.submit(learn, grid, learn_iterations, rng))

    result = []
    for future in futures:
        result.append(future.result())
    return result

def learn(grid, learn_iterations, rng):
    brain = Brain(config.GRID_LENGTH, rng)

    sim, viewer = world.build_world(grid)
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

        if result > objective_value:
            objective_value = result

        optimizer.register(params=next_point, target=result)
    sim.reset()
    viewer.close()
    return objective_value

if __name__ == '__main__':
    main()