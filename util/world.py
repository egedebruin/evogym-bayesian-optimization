import numpy as np
import os

from evogym import EvoWorld, EvoSim, EvoViewer, utils

from configs import config


def build_world(robot_structure):
	if config.ENVIRONMENT == 'simple' or config.ENVIRONMENT == 'jump':
		env = 'simple_environment.json'
	elif config.ENVIRONMENT == 'climb':
		env = 'climb_environment.json'
	elif config.ENVIRONMENT == 'carry':
		env = 'carry_environment.json'
	elif config.ENVIRONMENT == 'rugged':
		env = 'rugged_environment.json'
	elif config.ENVIRONMENT == 'steps':
		env = 'steps_environment.json'
	else:
		raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")
	world = EvoWorld.from_json(os.path.join('worlds', env))
	world.add_from_array(
		name='robot',
		structure=robot_structure,
		x=1,
		y=1,
		connections=utils.get_full_connectivity(robot_structure)
	)
	EvoSim._has_displayed_version = True
	sim = EvoSim(world)
	viewer = EvoViewer(sim)
	viewer.track_objects('robot')

	return sim, viewer

def run_simulator(sim, controller, sensors, viewer, simulator_length, headless):
	sim.reset()
	extra = []
	start_position = sim.object_pos_at_time(sim.get_time(), 'robot')
	if config.ENVIRONMENT == 'carry':
		extra.append(sim.object_pos_at_time(sim.get_time(), 'package'))

	for simulation_step in range(simulator_length):
		if config.ENVIRONMENT == 'jump':
			if simulation_step > 50:
				extra.append(np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[1]))
		if simulation_step % 5 == 0:
			sensor_input = sensors.get_input_from_sensors(sim.object_pos_at_time(sim.get_time(), 'robot'),
										sim.object_vel_at_time(sim.get_time(), 'robot'),
										sim.get_actuator_indices('robot'),
										sim.get_time())
			action = controller.control(sensor_input)
			sim.set_action(
				'robot',
				action
			)
		sim.step()
		if not headless:
			viewer.render('screen')

	end_position = sim.object_pos_at_time(sim.get_time(), 'robot')
	if config.ENVIRONMENT == 'carry':
		extra.append(sim.object_pos_at_time(sim.get_time(), 'package'))
	return calculate_objective_value(start_position, end_position, extra)

def calculate_objective_value(start_position, end_position, extra):
	if config.ENVIRONMENT == 'simple' or config.ENVIRONMENT == 'rugged' or config.ENVIRONMENT == 'steps':
		return np.mean(end_position[0]) - np.mean(start_position[0])
	elif config.ENVIRONMENT == 'climb':
		return np.mean(end_position[1]) - np.mean(start_position[1])
	elif config.ENVIRONMENT == 'jump':
		return np.max(extra)
	elif config.ENVIRONMENT == 'carry':
		if np.min(extra[1][1]) < 1.5:
			return 0.0
		return np.mean(extra[1][0] - np.mean(extra[0][0]))
	else:
		raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")