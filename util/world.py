import numpy as np
import os

from evogym import EvoWorld, EvoSim, EvoViewer, utils, WorldObject

from configs import config

def build_world(robot_structure):
	env = get_environment()
	world = EvoWorld.from_json(os.path.join('worlds', env))
	world.add_from_array(
		name='robot',
		structure=robot_structure,
		x=12,
		y=1,
		connections=utils.get_full_connectivity(robot_structure)
	)

	world = add_extra_attributes(world)
	EvoSim._has_displayed_version = True
	sim = EvoSim(world)
	viewer = EvoViewer(sim)
	tracker(viewer)

	return sim, viewer

def run_simulator(sim, controller, sensors, viewer, simulator_length, headless):
	sim.reset()
	extra_metrics = []
	start_position = sim.object_pos_at_time(sim.get_time(), 'robot')
	extra_metrics = extra_metrics_for_objective_value('before', sim, extra_metrics)

	for simulation_step in range(simulator_length):
		if simulation_step > 50:
			extra_metrics = extra_metrics_for_objective_value('during', sim, extra_metrics)
		if simulation_step % 5 == 0:
			sensor_input = sensors.get_input_from_sensors(sim)
			action = controller.control(sensor_input)
			sim.set_action(
				'robot',
				action
			)
		sim.step()
		if not headless:
			viewer.render('screen')

	end_position = sim.object_pos_at_time(sim.get_time(), 'robot')
	extra_metrics = extra_metrics_for_objective_value('after', sim, extra_metrics)
	return calculate_objective_value(start_position, end_position, extra_metrics)

# Below are Environment specific things
def get_environment():
	if config.ENVIRONMENT == 'simple' or config.ENVIRONMENT == 'jump' or config.ENVIRONMENT == 'catch':
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
	return env

def tracker(viewer):
	if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
		viewer.track_objects('package', 'robot')
	else:
		viewer.track_objects('robot')

def add_extra_attributes(world):
	if config.ENVIRONMENT == 'catch':
		# self.offsetx = random.randint(-6, 4)
		# self.offsety = random.randint(0, 5)

		offsetx = 0
		offsety = 0

		package = WorldObject.from_json(os.path.join('worlds', 'package.json'))
		package.set_pos(6, 41)
		package.rename('package')
		world.add_object(package)

		peg1 = WorldObject.from_json(os.path.join('worlds', 'peg.json'))
		peg1.set_pos(2, 39)
		peg1.rename('peg1')
		world.add_object(peg1)

		peg2 = WorldObject.from_json(os.path.join('worlds', 'peg.json'))
		peg2.set_pos(4, 25)
		peg2.rename('peg2')
		world.add_object(peg2)

	return world

def calculate_objective_value(start_position, end_position, extra_metrics):
	if config.ENVIRONMENT == 'simple' or config.ENVIRONMENT == 'rugged' or config.ENVIRONMENT == 'steps':
		return np.mean(end_position[0]) - np.mean(start_position[0])
	elif config.ENVIRONMENT == 'climb':
		return np.mean(end_position[1]) - np.mean(start_position[1])
	elif config.ENVIRONMENT == 'jump':
		return np.max(extra_metrics)
	elif config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
		if np.min(extra_metrics[1][1]) < 1.5:
			return 0.0
		return np.mean(extra_metrics[1][0] - np.mean(extra_metrics[0][0]))
	else:
		raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")

def extra_metrics_for_objective_value(timing, sim, extra):
	if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
		if timing == 'before' or timing == 'after':
			extra.append(sim.object_pos_at_time(sim.get_time(), 'package'))

	if config.ENVIRONMENT == 'jump':
		if timing =='during':
			extra.append(np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[1]))

	return extra