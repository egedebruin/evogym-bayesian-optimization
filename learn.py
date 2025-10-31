import concurrent.futures
import math
from collections import deque

import numpy as np
import torch
from bayes_opt import BayesianOptimization, acquisition

from sklearn.gaussian_process.kernels import Matern

from reinforcement_learning.ddpg import DDPG
from robot.active import Controller
from configs import config
from robot.running_norm import RunningNorm
from robot.sensors import Sensors
from util import world
from util.logger_setup import logger

TYPE_DDPG = 'ddpg'
TYPE_BO = 'bo'


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

	rl_agent = None
	if config.LEARN_METHOD == TYPE_DDPG:
		args = DDPG.create_global_critic_params(len(actuator_indices))
		args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
				for k, v in args.items()}
		velocity_norm = RunningNorm()
		rl_agent = DDPG(args, velocity_norm)

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
	previous_policy = None
	experience = []
	transition_buffer = deque(maxlen=2000)
	for iteration in range(config.LEARN_ITERATIONS):
		logger.info(f"Learn generation {iteration + 1}")
		if iteration == 0 and config.INHERIT_SAMPLES == -1:
			next_point = brain.to_next_point(actuator_indices)
		elif iteration < config.INHERIT_SAMPLES and len(inherited_experience) > 0:
			next_point = inherited_experience[iteration][0]
		else:
			next_point = optimizer.suggest()

		if config.LEARN_METHOD == TYPE_BO:
			args = brain.next_point_to_controller_values(next_point, actuator_indices)
			args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
				for k, v in args.items()}
			args = Controller.get_parameters_from_args(args)
		elif config.LEARN_METHOD == TYPE_DDPG:
			if iteration == 0:
				print("Initial RL part")
				args = brain.next_point_to_controller_values(next_point, actuator_indices)
				args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
						for k, v in args.items()}
				args = Controller.get_parameters_from_args(args)
				rl_agent.set_policy_optimizer(args, rl_agent.policy_lr)
			else:
				print("Secondary RL part")
				args = previous_policy
		else:
			raise NotImplementedError

		controller = Controller(args)
		controller.set_rl_agent(rl_agent)

		sensors = Sensors(robot_body.grid)

		result = world.run_simulator(sim, controller, sensors, viewer, config.SIMULATION_LENGTH, True, transition_buffer)
		if result > objective_value:
			objective_value = result
			best_brain = next_point

		alphas = np.append(alphas, config.LEARN_ALPHA)
		optimizer.register(params=next_point, target=result)
		optimizer.set_gp_params(alpha=alphas)
		experience.append((next_point, result))
		previous_policy = controller.policy_weights
	sim.reset()
	viewer.close()
	return objective_value, best_brain, experience, individual