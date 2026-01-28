import concurrent.futures
import math
from collections import deque

import numpy as np
import torch
from bayes_opt import acquisition
from sklearn.gaussian_process.kernels import Matern

from custom_bayesian_optimization import CustomBayesianOptimization
from reinforcement_learning.ddpg import DDPG
from reinforcement_learning.ppo import PPO

from robot.active import Controller
from configs import config
from robot.sensors import Sensors
from util import world
from util.logger_setup import logger

TYPE_DDPG = 'ddpg'
TYPE_PPO = 'ppo'
TYPE_BO = 'bo'
RL_TYPES = [TYPE_DDPG, TYPE_PPO]


def learn_individuals(individuals, heights, rng):
	current_world, new_heights = world.get_environment(rng, previous_heights=heights, generation_index=individuals[0].original_generation)
	current_world = world.add_extra_attributes(current_world, rng)

	with concurrent.futures.ProcessPoolExecutor(
			max_workers=config.PARALLEL_PROCESSES
	) as executor:
		futures = []
		for individual in individuals:
			futures.append(executor.submit(learn, individual, rng, current_world))

	result = []
	for future in futures:
		result.append(future.result())
	return result, new_heights

def learn(individual, rng, current_world):
	robot_body = individual.body
	brain = individual.brain
	sim, viewer = world.build_world(robot_body.grid, world=current_world)
	try:
		actuator_indices = sim.get_actuator_indices('robot')
	except ValueError:
		logger.error('Failed to get actuator indices')
		logger.error(robot_body.grid)
		return -math.inf, [], individual

	rl_agent = None
	if config.LEARN_METHOD == TYPE_DDPG:
		rl_agent = DDPG(len(actuator_indices))
	elif config.LEARN_METHOD == TYPE_PPO:
		rl_agent = PPO(len(actuator_indices))

	optimizer = CustomBayesianOptimization(
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
	previous_policy = None
	best_inherited_objective_value = -math.inf
	experience = []
	transition_buffer = deque(maxlen=2000)
	for iteration in range(config.LEARN_ITERATIONS):
		logger.info(f"Learn iteration {iteration + 1}")

		next_point = get_next_point_from_inheritance(iteration, optimizer, brain, actuator_indices, inherited_experience)

		if config.LEARN_METHOD == TYPE_BO:
			controller_values = brain.next_point_to_controller_values(next_point, actuator_indices)
			args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
				for k, v in controller_values.items()}
		elif config.LEARN_METHOD in RL_TYPES and iteration == 0:
			# First iteration
			rl_agent.set_update_networks(True, True)
			controller_values = brain.next_point_to_controller_values(next_point, actuator_indices)
			args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
					for k, v in controller_values.items()}
			rl_agent.set_policy_optimizer(args, rl_agent.policy_lr)
		elif config.LEARN_METHOD in RL_TYPES and iteration + 1 == config.LEARN_ITERATIONS:
			# Last iteration
			rl_agent.set_update_networks(False, False)
			controller_values = brain.next_point_to_controller_values(best_brain, actuator_indices)
			args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
					for k, v in controller_values.items()}
			next_point = best_brain
		elif config.LEARN_METHOD in RL_TYPES:
			# Intermediate iteration
			if config.LEARN_METHOD == TYPE_PPO and iteration % config.RL_EVALUATIONS_FREQUENCY == config.RL_EVALUATIONS_FREQUENCY - 1:
				rl_agent.set_update_networks(False, False)
			else:
				rl_agent.set_update_networks(True, True)
			args = previous_policy
			controller_values = {k: v.detach().numpy() for k, v in args.items()}
			next_point = brain.controller_values_to_next_point(controller_values)
		else:
			raise NotImplementedError

		controller = Controller(args)
		controller.set_rl_agent(rl_agent)

		sensors = Sensors(robot_body.grid)

		result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, True, individual.original_generation, transition_buffer)
		if result > objective_value:
			objective_value = result
			best_brain = next_point
			if iteration < config.INHERIT_SAMPLES and len(inherited_experience) > 0:
				best_inherited_objective_value = result
			if iteration == 0 and config.INHERIT_SAMPLES == -1:
				best_inherited_objective_value = result

		if config.LEARN_METHOD == TYPE_BO and not config.RANDOM_LEARNING:
			alphas = np.append(alphas, config.LEARN_ALPHA)
			optimizer.register(params=next_point, target=result)
			optimizer.set_gp_params(alpha=alphas)
		experience.append((next_point, result))
		previous_policy = controller.policy_weights
	sim.reset()
	viewer.close()
	return objective_value, best_brain, experience, best_inherited_objective_value, individual

def get_next_point_from_inheritance(iteration, optimizer, brain, actuator_indices, inherited_experience):
	if iteration == 0 and config.INHERIT_SAMPLES == -1 and config.DARWINIAN:
		return brain.to_next_point(actuator_indices)
	elif iteration < config.INHERIT_SAMPLES and len(inherited_experience) > 0:
		return inherited_experience[iteration][0]
	else:
		return optimizer.suggest()