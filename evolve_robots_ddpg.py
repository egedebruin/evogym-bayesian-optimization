import os
import warnings

# Suppress UserWarnings from torch (CUDA initialization error)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA initialization: Unexpected error from cudaGetDeviceCount().*")

# Force single-threaded execution before imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import argparse
import numpy as np

from robot.body import Body
from robot.active import Brain
from robot.brain_nn import BrainNN
from configs import config
from individual import Individual
from selection import Selection
from util import writer, start
from util.logger_setup import logger, logger_setup
import learn

def run_generation(individuals, heights, rng):
    new_population = []
    results, new_heights = learn.learn_individuals(individuals, heights, rng)
    
    if config.ENVIRONMENT in ['changing', 'random']:
        writer.write_to_environments_file(";".join(str(e) for e in new_heights))
    
    for (experience, inherited_objective_values, individual) in results:
        individual.add_evaluation(experience, inherited_objective_values)
        new_population.append(individual)
        writer.write_to_individuals_file(individual)
    return new_population, new_heights

def get_offspring(population, generation_index, parent_selection, rng:np.random.Generator):
    selected_individuals = parent_selection.select(population, rng)
    offspring = []
    for i, individual in enumerate(selected_individuals):
        new_individual = individual.generate_new_individual(generation_index, i, rng)
        new_individual.inherit_experience(population, individual, rng)
        offspring.append(new_individual)
    return offspring

def load_initial_population_from_json(json_path, rng):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    robots_data = data.get('robots', [])
    individuals = []
    robot_names = []
    for i, robot_entry in enumerate(robots_data):
        grid = np.array(robot_entry['robot'])
        body = Body()
        body.replace_grid(grid)
        brain = Brain(config.GRID_LENGTH, rng)
        individual_id = f"0-{i}"
        individual = Individual(individual_id, body, brain, 0, [])
        individuals.append(individual)
        robot_names.append(robot_entry.get('name', f"robot_{i}"))
    
    return individuals, robot_names

def main():
    parser = argparse.ArgumentParser(description='Evolve robots from robots.json using DDPG.')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations to evolve.')
    parser.add_argument('--env', type=str, default='simple', help='Environment to use.')
    parser.add_argument('--robots-json', type=str, default='../IN5490-evogym-builder/experiments/robots.json', help='Path to robots.json file.')
    parser.add_argument('--repetition', type=int, default=1, help='Repetition number for output folder.')
    
    args = parser.parse_args()
    
    # Set up config based on arguments
    config.ENVIRONMENT = args.env
    config.READ_ARGS = False # We handle args here
    
    if not os.path.exists(config.FOLDER):
        os.makedirs(config.FOLDER)
    
    logger_setup()
        
    rng = start.make_rng_seed()
    
    logger.info(f"Loading initial population from {args.robots_json}")
    individuals, robot_names = load_initial_population_from_json(args.robots_json, rng)
    config.POP_SIZE = len(individuals)
    config.OFFSPRING_SIZE = config.POP_SIZE
    config.PARALLEL_PROCESSES = config.OFFSPRING_SIZE
    
    # Save mapping between robot names and individual IDs
    mapping = {robot_names[i]: individuals[i].id for i in range(len(individuals))}
    mapping_path = os.path.join(config.FOLDER, "robot_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    logger.info(f"Robot mapping saved to {mapping_path}")
    
    logger.info("Generation 0")
    population, heights = run_generation(individuals, [], rng)
    writer.write_to_populations_file(population)
    writer.write_to_rng_file(rng)
    
    parent_selection = Selection(config.OFFSPRING_SIZE, config.PARENT_SELECTION, {'pool_size': config.PARENT_POOL})
    survivor_selection = Selection(config.POP_SIZE, config.SURVIVOR_SELECTION)
    
    for generation_index in range(1, args.generations + 1):
        logger.info(f"Generation {generation_index}/{args.generations}")
        
        offspring = get_offspring(population, generation_index, parent_selection, rng)
        evaluated_offspring, heights = run_generation(offspring, heights, rng)
        
        population += evaluated_offspring
        population = survivor_selection.select(population, rng)
        
        writer.write_to_populations_file(population)
        writer.write_to_rng_file(rng)

if __name__ == '__main__':
    main()
