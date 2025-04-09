import argparse

import numpy as np
import os
import time
from datetime import datetime

import selection
from robot.body import Body
from robot.brain import Brain
import config
from robot.individual import Individual
from util import restart_population, writer
import learn
from util.logger_setup import logger, logger_setup

def make_rng_seed():
	seed = int(datetime.now().timestamp() * 1e6) % 2**32
	logger.info(f"Random Seed: {seed}")
	return np.random.Generator(np.random.PCG64(seed))

def run_generation(individuals, rng):
	new_population = []
	results = learn.learn_individuals(individuals, rng)
	i = 0
	for (objective_value, experience, individual) in results:
		individual.add_evaluation(objective_value, experience)
		new_population.append(individual)
		writer.write_to_individuals_file(individual)
		i += 1
	return new_population

def get_offspring(population, offspring_size, generation_index, rng:np.random.Generator):
	selected_individuals = selection.select(population, offspring_size, config.PARENT_SELECTION, rng, config.PARENT_POOL)
	offspring = []
	for i, individual in enumerate(selected_individuals):
		new_individual = individual.generate_new_individual(generation_index, i, rng)
		offspring.append(new_individual)
	return offspring

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learn', help='Number learn generations.', required=True, type=int)
	parser.add_argument('--inherit-samples', help='Number of samples to inherit.', required=True, type=int)
	parser.add_argument('--repetition', help='Experiment number.', required=True, type=int)

	args = parser.parse_args()
	config.LEARN_ITERATIONS = args.learn
	config.INHERIT_SAMPLES = args.inherit_samples
	config.FOLDER = f"results/learn-{args.learn}_inherit-{args.inherit_samples}_repetition-{args.repetition}/"

def calculate_generations():
	number_of_generations = int(config.FUNCTION_EVALUATIONS / (config.LEARN_ITERATIONS * config.OFFSPRING_SIZE))
	number_of_generations_initial_population = int(config.POP_SIZE / config.OFFSPRING_SIZE)
	return number_of_generations - number_of_generations_initial_population

def main():
	if config.READ_ARGS:
		read_args()
	if not os.path.exists(config.FOLDER):
		os.makedirs(config.FOLDER)
	logger_setup()

	start_time = time.time()
	if os.path.exists(config.FOLDER + "populations.txt"):
		logger.info("Restarting populations...")
		population, num_generations = restart_population.get_population()
		rng = restart_population.get_rng()
		logger.info("Restart succeeded")
	else:
		logger.info(f"Generation 0")
		rng = make_rng_seed()
		num_generations = 0
		individuals = []
		for i in range(config.POP_SIZE):
			body = Body(config.GRID_LENGTH, config.INITIAL_SIZE, rng)
			brain = Brain(config.GRID_LENGTH, rng)
			individual_id = f"0-{i}"
			individual = Individual(individual_id, body, brain, 0, [])
			individuals.append(individual)

		population = run_generation(individuals, rng)
		writer.write_to_populations_file(population)
		writer.write_to_rng_file(rng)

	number_of_generations = calculate_generations()

	for i in range(number_of_generations):
		if i < num_generations:
			continue
		logger.info(f"Generation {i + 1}/{number_of_generations}")
		offspring = get_offspring(population, config.OFFSPRING_SIZE, i + 1, rng)
		population += run_generation(offspring, rng)
		population = selection.select(population, config.POP_SIZE, config.SURVIVOR_SELECTION)
		writer.write_to_populations_file(population)
		writer.write_to_rng_file(rng)

	logger.info(f"Finished in {time.time() - start_time} seconds.")

if __name__ == '__main__':
	main()