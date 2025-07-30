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
ENVIRONMENT = 'simple'
REPS = 10
LENGTH_SCALES = [-1, 0.2, 0.5, 1.0, 10.0]

def main():
    config.ENVIRONMENT = ENVIRONMENT

    if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
        BrainNN.NUMBER_OF_INPUT_NEURONS = BrainNN.NUMBER_OF_INPUT_NEURONS + 2

    result_dict = {
        'inherit': [],
        'type': [],
        'pool': [],
        'repetition': [],
        'random': [],
        'length_scale': [],
        'iteration': [],
        'objective_value': []
    }

    strategy_keys = list(LABELS.keys())

    for key in strategy_keys:
        # returns a list of dicts, each one run (one rando=True OR one specific length_scale with rando=False)
        results = parallelize(key[0], key[1], key[2], 100)

        for res in results:
            rep = res['repetition']
            rando = res['rando']
            length_scale = res['length_scale']  # None for rando=True
            obj_vals = res['objective_values']

            for i, val in enumerate(obj_vals, start=1):
                result_dict['inherit'].append(key[0])
                result_dict['type'].append(key[1])
                result_dict['pool'].append(key[2])
                result_dict['repetition'].append(rep)
                result_dict['random'].append(rando)
                result_dict['length_scale'].append(length_scale)
                result_dict['iteration'].append(i)
                result_dict['objective_value'].append(val)

    pd.DataFrame(result_dict).to_csv(f'{ENVIRONMENT}.csv', index=False)


def parallelize(inherit, i_type, pool, learn_iterations):
    # Keep (repetition, best_grid) so we can track which run came from which rep
    grids = []  # list of tuples: (r, best_grid)
    for r in range(REPS):
        folder = (
            f'results/learn-{EVALS_PER_GEN}_inherit-{inherit}_type-{i_type}_pool-{pool}_environment-{ENVIRONMENT}_repetition-1'
        )
        best_fitness = float('-inf')
        best_grid = None

        with open(folder + "/individuals.txt", "r") as file:
            for line in file:
                parts = line.strip().split(";")
                fitness = float(parts[5])
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_grid = np.array(ast.literal_eval(parts[1]))

        if best_grid is not None:
            grids.append((r, best_grid))

    results = []
    if not grids:
        return results  # nothing to do

    with concurrent.futures.ProcessPoolExecutor(max_workers=51) as executor:
        futures = []

        # --- rando=True exactly once (pick the first available grid) ---
        r0, g0 = grids[0]
        futures.append(
            executor.submit(learn, g0, learn_iterations, True, None, r0)
        )

        # --- For every grid, run rando=False with the four length_scales ---
        for r, grid in grids:
            for ls in LENGTH_SCALES:
                futures.append(
                    executor.submit(learn, grid, learn_iterations, False, ls, r)
                )

        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    return results


def learn(grid, learn_iterations, rando, length_scale, repetition):
    """
    rando: bool -> if True, GP suggestions are not registered (pure random)
    length_scale: float or None -> GP Matern length_scale used when rando is False (or ignored if rando True)
    repetition: int -> which repetition this grid came from
    """
    rng = start.make_rng_seed()

    brain = Brain(config.GRID_LENGTH, rng)

    sim, viewer = world.build_world(grid, rng)
    actuator_indices = sim.get_actuator_indices('robot')

    optimizer = BayesianOptimization(
        f=None,
        pbounds=brain.get_p_bounds(actuator_indices),
        allow_duplicate_points=True,
        random_state=int(rng.integers(low=0, high=2 ** 32)),
        acquisition_function=acquisition.UpperConfidenceBound(
            kappa=config.LEARN_KAPPA,
            random_state=int(rng.integers(low=0, high=2 ** 32))
        )
    )

    # Choose kernel length_scale: when specified, override config
    ls = config.LEARN_LENGTH_SCALE if (length_scale is None or length_scale == -1) else float(length_scale)

    if length_scale == -1:
        optimizer.set_gp_params(
            kernel=Matern(nu=config.LEARN_NU, length_scale=ls)
        )
    else:
        optimizer.set_gp_params(
            kernel=Matern(nu=config.LEARN_NU, length_scale=ls, length_scale_bounds='fixed')
        )
    optimizer.set_gp_params(alpha=config.LEARN_ALPHA)

    objective_value = -math.inf
    objective_values = []

    for bayesian_optimization_iteration in range(learn_iterations):
        print(f"Learn generation {bayesian_optimization_iteration + 1}")
        next_point = optimizer.suggest()

        args = brain.next_point_to_controller_values(next_point, actuator_indices)
        controller = Controller(args)
        sensors = Sensors(grid)

        result = world.run_simulator(
            sim, controller, sensors, viewer, config.SIMULATION_LENGTH, True
        )
        objective_values.append(result)

        if result > objective_value:
            objective_value = result

        # When rando=False, we actually do BO; when rando=True, we skip registering.
        if not rando:
            optimizer.register(params=next_point, target=result)

    sim.reset()
    viewer.close()

    return {
        'objective_values': objective_values,
        'rando': rando,
        'length_scale': None if rando else ls,
        'repetition': repetition,
    }

if __name__ == '__main__':
    main()