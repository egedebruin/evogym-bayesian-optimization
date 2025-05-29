import concurrent.futures
import math

import numpy as np
from bayes_opt import BayesianOptimization, acquisition

from sklearn.gaussian_process.kernels import Matern

from robot.active import Controller
from configs import config
from robot.sensors import Sensors
from util import world
from util.logger_setup import logger


def learn_individuals(individuals, rng):
	with concurrent.futures.ProcessPoolExecutor(
			max_workers=config.PARALLEL_PROCESSES
	) as executor:
		futures = []
		for individual in individuals:
			futures.append(executor.submit(learn, individual, rng))

	result = []
	for future in futures:
		result.append(future.result())
	return result

def learn(individual, rng):
	robot_body = individual.body
	brain = individual.brain
	sim, viewer = world.build_world(robot_body.grid, rng)
	try:
		actuator_indices = sim.get_actuator_indices('robot')
	except ValueError:
		logger.error('Failed to get actuator indices')
		logger.error(robot_body.grid)
		return -math.inf, [], individual

	optimizer = BayesianOptimization(
		f=None,
		pbounds=brain.get_p_bounds(actuator_indices),
		allow_duplicate_points=True,
		random_state=int(rng.integers(low=0, high=2 ** 32)),
		acquisition_function=acquisition.UpperConfidenceBound(kappa=config.LEARN_KAPPA,
															  random_state=int(rng.integers(low=0, high=2 ** 32)))
	)
	optimizer.set_gp_params(kernel=Matern(nu=config.LEARN_NU, length_scale=config.LEARN_LENGTH_SCALE, length_scale_bounds="fixed"))

	inherited_experience = brain.update_experience_with_actuator_indices(individual.inherited_experience, actuator_indices)

	alphas = np.array([])
	if config.INHERIT_SAMPLES == 0:
		for experience_sample, objective_value in inherited_experience:
			alphas = np.append(alphas, config.LEARN_INHERITED_ALPHA)
			optimizer.register(params=experience_sample, target=objective_value)
			optimizer.set_gp_params(alpha=alphas)

	objective_value = -math.inf
	best_brain = None
	experience = []
	for bayesian_optimization_iteration in range(config.LEARN_ITERATIONS):
		logger.info(f"Learn generation {bayesian_optimization_iteration + 1}")
		if bayesian_optimization_iteration == 0 and config.INHERIT_SAMPLES == -1:
			next_point = brain.to_next_point(actuator_indices)
		elif bayesian_optimization_iteration < config.INHERIT_SAMPLES and len(inherited_experience) > 0:
			next_point = inherited_experience[bayesian_optimization_iteration][0]
		else:
			next_point = optimizer.suggest()

		args = brain.next_point_to_controller_values(next_point, actuator_indices)
		controller = Controller(args)
		sensors = Sensors(robot_body.grid)

		result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, True)
		if result > objective_value:
			objective_value = result
			best_brain = next_point

		alphas = np.append(alphas, config.LEARN_ALPHA)
		optimizer.register(params=next_point, target=result)
		optimizer.set_gp_params(alpha=alphas)
		experience.append((next_point, objective_value))
	sim.reset()
	viewer.close()
	return objective_value, best_brain, experience, individual