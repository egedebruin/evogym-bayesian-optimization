import ast
import numpy as np
import os
import sys
import json
import warnings

# Suppress UserWarnings from torch (CUDA initialization error)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA initialization: Unexpected error from cudaGetDeviceCount().*")

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from configs import config

from util import world
from util import start
from robot.active import Brain
from robot.brain_nn import BrainNN
from robot.active import Controller
from robot.sensors import Sensors
from main import set_number_of_sensors

def get_generation_zero_individuals(folder):
    set_number_of_sensors()
    individuals = []

    individuals_path = os.path.join(folder, "individuals.txt")
    if not os.path.exists(individuals_path):
        print(f"Error: {individuals_path} not found.")
        return []

    with open(individuals_path, "r") as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split(";")
            try:
                # Individual ID format is "gen-index"
                gen_part = parts[0].split("-")[0]
                generation = int(gen_part)
                if generation == 0:
                    individuals.append(parts)
            except (IndexError, ValueError):
                continue

    return individuals

def load_mapping(folder):
    mapping_path = os.path.join(folder, "robot_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            # Reverse mapping to get name from ID
            return {v: k for k, v in mapping.items()}
    return {}

def main():
    BrainNN.set_modular_vision()
    
    folder = config.FOLDER
    if not folder.endswith('/'):
        folder += '/'
        
    individuals = get_generation_zero_individuals(folder)
    id_to_name = load_mapping(folder)

    if not individuals:
        print("No individuals found for generation 0.")
        return

    print(f"Found {len(individuals)} individuals in generation 0.")

    for ind in individuals:
        ind_id = ind[0]
        robot_name = id_to_name.get(ind_id, "Unknown")
        fitness = ind[5]
        
        print(f"\nDisplaying Robot: {robot_name} (ID: {ind_id}, Fitness: {fitness})")
        
        grid = np.array(ast.literal_eval(ind[1]))
        sim, viewer = world.build_world(grid, rng=start.make_rng_seed())

        # The viewer title or an overlay would be nice, but evogym's viewer 
        # doesn't easily support dynamic text overlay from here.
        # We'll print to console as requested.

        experience = ast.literal_eval(ind[3])
        controller_values = Brain.next_point_to_controller_values(experience, sim.get_actuator_indices('robot'))
        args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
                for k, v in controller_values.items()}

        controller = Controller(args)
        sensors = Sensors(grid)
        
        # Run simulator with viewer
        result = world.run_simulator(
            sim, 
            controller, 
            sensors, 
            viewer, 
            config.SIMULATION_LENGTH, 
            False, 
            int(ind[6])
        )

        viewer.close()

        print(f"Finished simulating {robot_name}. Result: {result}")
        
        # Prompt user to continue to next robot if they want to see them one by one
        try:
            input("Press Enter to see the next robot (or Ctrl+C to stop)...")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

if __name__ == "__main__":
    main()
