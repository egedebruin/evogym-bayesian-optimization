import numpy as np
import os

from robot.body import Body
from robot.active import Brain
from robot.brain_nn import BrainNN
from configs import config
from individual import Individual
from selection import Selection
from util import restart_population, writer, start
import learn
from util.logger_setup import logger, logger_setup

def run_generation(individuals, rng):
	new_population = []
	results = learn.learn_individuals(individuals, rng)
	i = 0
	for (objective_value, best_brain, experience, best_inherited_objective_value, individual) in results:
		individual.add_evaluation(objective_value, best_brain, experience, best_inherited_objective_value)
		new_population.append(individual)
		writer.write_to_individuals_file(individual)
		i += 1
	return new_population

def get_offspring(population, generation_index, parent_selection, rng:np.random.Generator):
	selected_individuals = parent_selection.select(population, rng)
	offspring = []
	for i, individual in enumerate(selected_individuals):
		new_individual = individual.generate_new_individual(generation_index, i, rng)
		new_individual.inherit_experience(population, individual, rng)
		offspring.append(new_individual)
	return offspring

def calculate_generations():
	number_of_generations = int(config.FUNCTION_EVALUATIONS / (config.LEARN_ITERATIONS * config.OFFSPRING_SIZE))
	number_of_generations_initial_population = int(config.POP_SIZE / config.OFFSPRING_SIZE)
	return number_of_generations - number_of_generations_initial_population

def set_number_of_sensors():
	if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
		BrainNN.NUMBER_OF_INPUT_NEURONS = BrainNN.NUMBER_OF_INPUT_NEURONS + 2

def main():
	if config.READ_ARGS:
		start.read_args()
	if not os.path.exists(config.FOLDER):
		os.makedirs(config.FOLDER)
	logger_setup()
	set_number_of_sensors()

	if os.path.exists(config.FOLDER + "populations.txt"):
		logger.info("Restarting populations...")
		population, num_generations = restart_population.get_population()
		rng = restart_population.get_rng()
		logger.info("Restart succeeded")
	else:
		logger.info(f"Generation 0")
		rng = start.make_rng_seed()
		num_generations = 0
		individuals = []
		for i in range(config.POP_SIZE):
			body_size = rng.integers(config.MIN_INITIAL_SIZE, config.MAX_INITIAL_SIZE + 1)
			body = Body(config.GRID_LENGTH, body_size, rng)
			brain = Brain(config.GRID_LENGTH, rng)
			individual_id = f"0-{i}"
			individual = Individual(individual_id, body, brain, 0, [])
			individuals.append(individual)

		population = run_generation(individuals, rng)
		writer.write_to_populations_file(population)
		writer.write_to_rng_file(rng)

	number_of_generations = calculate_generations()

	parent_selection = Selection(config.OFFSPRING_SIZE, config.PARENT_SELECTION, {'pool_size': config.PARENT_POOL})
	survivor_selection = Selection(config.POP_SIZE, config.SURVIVOR_SELECTION)
	for generation_index in range(1, number_of_generations + 1):
		if generation_index < num_generations:
			continue
		logger.info(f"Generation {generation_index}/{number_of_generations}")
		offspring = get_offspring(population, generation_index, parent_selection, rng)
		population += run_generation(offspring, rng)
		population = survivor_selection.select(population, rng)
		writer.write_to_populations_file(population)
		writer.write_to_rng_file(rng)

if __name__ == '__main__':
	main()