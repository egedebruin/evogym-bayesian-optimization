import ast
import math

import numpy as np
import os
import sys

from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from configs import config

from util import world
from util import start
from robot.active import Brain
from robot.active import Controller
from robot.sensors import Sensors
from analysis import run_best

def learn_quickly(bo_brain, bo_grid, bo_rng, bo_sim, bo_viewer):
    actuator_indices = bo_sim.get_actuator_indices('robot')

    optimizer = BayesianOptimization(
        f=None,
        pbounds=bo_brain.get_p_bounds(actuator_indices),
        allow_duplicate_points=True,
        random_state=int(bo_rng.integers(low=0, high=2 ** 32)),
        acquisition_function=acquisition.UpperConfidenceBound(kappa=config.LEARN_KAPPA,
                                                              random_state=int(bo_rng.integers(low=0, high=2 ** 32)))
    )
    optimizer.set_gp_params(
        kernel=Matern(nu=config.LEARN_NU, length_scale=config.LEARN_LENGTH_SCALE, length_scale_bounds="fixed"),
        alpha=config.LEARN_ALPHA
    )

    objective_value = -math.inf
    best_brain = None
    for bayesian_optimization_iteration in range(config.LEARN_ITERATIONS):
        print(f"Bayesian optimization iteration: {bayesian_optimization_iteration + 1}")
        next_point = optimizer.suggest()

        bo_args = bo_brain.next_point_to_controller_values(next_point, actuator_indices)
        bo_controller = Controller(bo_args)
        bo_sensors = Sensors(bo_grid)

        bo_result = world.run_simulator(bo_sim, bo_controller, bo_sensors, bo_viewer, config.SIMULATION_LENGTH, True)
        if bo_result > objective_value:
            objective_value = bo_result
            best_brain = bo_args
        optimizer.register(params=next_point, target=bo_result)
    bo_sim.reset()
    bo_viewer.close()
    return best_brain

def main():
    best_individual = run_best.get_best_individual(30)
    rng = start.make_rng_seed()
    grid = np.array(ast.literal_eval(best_individual[1]))
    brain = Brain(config.GRID_LENGTH, rng)
    sim, viewer = world.build_world(grid, rng)

    args = learn_quickly(brain, grid, rng, sim, viewer)

    controller = Controller(args)
    sensors = Sensors(grid)
    result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, False)

    print("DB best value: ", best_individual[5])
    print("Rerun value: ", result)

if __name__ == "__main__":
    main()