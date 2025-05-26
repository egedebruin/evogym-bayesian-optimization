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
    file = open(config.FOLDER + "individuals.txt", "r")
    all_individuals = file.read().splitlines()
    all_individuals = [individual.split(";") for individual in all_individuals]
    correct_individuals = []
    for individual in all_individuals:
        if int(individual[0].split("-")[0]) < min_generation:
            continue
        correct_individuals.append(individual)
    sorted_individuals = sorted(correct_individuals, key=lambda individual: float(individual[5]), reverse=True)
    result_individual = sorted_individuals[0]
    return result_individual

def main():
    best_individual = get_best_individual()

    grid = np.array(ast.literal_eval(best_individual[1]))
    sim, viewer = world.build_world(grid, start.make_rng_seed())

    experience = ast.literal_eval(best_individual[3])
    best_brain = sorted(experience, key=lambda evaluation: float(evaluation[1]), reverse=True)[0][0]
    args = Brain.next_point_to_controller_values(best_brain, sim.get_actuator_indices('robot'))

    controller = Controller(args)
    sensors = Sensors(grid)
    result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, False)

    print("DB best value: ", best_individual[5])
    print("Rerun value: ", result)

if __name__ == "__main__":
    main()