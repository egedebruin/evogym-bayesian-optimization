import ast
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from configs import config

from util import world
from util import start
from robot.active import Brain
from robot.active import Controller
from robot.sensors import Sensors
from main import set_number_of_sensors

def get_best_individual(min_generation=0):
    set_number_of_sensors()
    best_individual = None
    best_fitness = float("-inf")

    with open(config.FOLDER + "individuals.txt", "r") as file:
        for line in file:
            if not line.strip():
                continue  # Skip empty lines
            individual = line.strip().split(";")
            try:
                generation = int(individual[0].split("-")[0])
                fitness = float(individual[5])
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed line: {line.strip()} — {e}")
                continue
            if generation < min_generation:
                continue
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual

    return best_individual


def main():
    best_individual = get_best_individual()

    grid = np.array(ast.literal_eval(best_individual[1]))
    sim, viewer = world.build_world(grid, start.make_rng_seed())

    experience = ast.literal_eval(best_individual[3])
    # best_brain = sorted(experience, key=lambda evaluation: float(evaluation[1]), reverse=True)[0][0]
    args = Brain.next_point_to_controller_values(experience, sim.get_actuator_indices('robot'))

    controller = Controller(args)
    sensors = Sensors(grid)
    result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, False)

    print("DB best value: ", best_individual[5])
    print("Rerun value: ", result)

if __name__ == "__main__":
    main()