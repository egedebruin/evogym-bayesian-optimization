import ast
from pathlib import Path
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from analysis.run_best import get_best_individual
from robot.active import Controller, Brain
from robot.sensors import Sensors
from util import world, start
from configs import config


FOLDER = "results/new_paper"
NUM_ROBOTS = 20
NUM_ITERATIONS = 100
NUM_PROCESSES = 10


def process_folder(args):
    robot_number, folder, environment, input_neurons = args

    # Each process gets its own RNG
    rng = start.make_rng_seed()

    config.ENVIRONMENT = environment
    Brain.NUMBER_OF_INPUT_NEURONS = input_neurons

    local_result = {
        'environment': [],
        'robot_number': [],
        'iteration': [],
        'value1': [],
        'value2': [],
    }

    individual = get_best_individual(str(folder) + "/")
    grid = np.array(ast.literal_eval(individual[1]))

    sim, viewer = world.build_world(grid)
    actuator_indices = sim.get_actuator_indices('robot')

    for iteration in range(NUM_ITERATIONS):
        brain = Brain(max_size=config.GRID_LENGTH, rng=rng)

        controller_values = Brain.next_point_to_controller_values(
            brain.to_next_point(actuator_indices),
            actuator_indices
        )

        args_dict = {
            k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in controller_values.items()
        }

        controller = Controller(args_dict)
        sensors = Sensors(grid)

        local_result['environment'].append(environment)
        local_result['robot_number'].append(robot_number)
        local_result['iteration'].append(iteration)

        for generation_index in [1, 2]:
            result = world.run_simulator(
                sim, controller, sensors, viewer,
                config.SIMULATION_LENGTH, True,
                generation_index=generation_index
            )

            local_result[f'value{generation_index}'].append(result)
            sim.reset()

    viewer.close()

    return local_result


if __name__ == "__main__":

    base_path = Path(FOLDER)
    prefix = "learn"

    matching_folders = [
        p for p in base_path.rglob("*")
        if p.is_dir() and p.name.startswith(prefix)
    ]

    selected_folders = random.sample(
        matching_folders,
        min(NUM_ROBOTS, len(matching_folders))
    )

    all_results = []

    for (environment, input_neurons) in [
        ('bidirectional', 30),
        ('bidirectional2', 31)
    ]:

        tasks = [
            (robot_number, folder, environment, input_neurons)
            for robot_number, folder in enumerate(selected_folders)
        ]

        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(process_folder, tasks)

        all_results.extend(results)

    # Merge dictionaries
    final_result = {
        'environment': [],
        'robot_number': [],
        'iteration': [],
        'value1': [],
        'value2': [],
    }

    for res in all_results:
        for key in final_result:
            final_result[key].extend(res[key])

    pd.DataFrame(final_result).to_csv("b-f-r-results.csv", index=False)