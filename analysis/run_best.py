import ast
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config

from util import world
from robot.active import Brain
from robot.active import Controller
from robot.sensors import Sensors

file = open(config.FOLDER + "individuals.txt", "r")
all_individuals = file.read().splitlines()
all_individuals = [individual.split(";") for individual in all_individuals]
sorted_individuals = sorted(all_individuals, key=lambda individual: float(individual[5]), reverse=True)
best_individual = sorted_individuals[0]

grid = np.array(ast.literal_eval(best_individual[1]))
sim, viewer = world.build_world(grid)

experience = ast.literal_eval(best_individual[3])
best_brain = sorted(experience, key=lambda evaluation: float(evaluation[1]), reverse=True)[0][0]
args = Brain.next_point_to_controller_values(best_brain, sim.get_actuator_indices('robot'))

controller = Controller(args)
sensors = Sensors(grid)
result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, False)

print("DB best value: ", best_individual[5])
print("Rerun value: ", result)