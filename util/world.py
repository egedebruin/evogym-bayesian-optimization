import numpy as np
import os

from evogym import EvoWorld, EvoSim, EvoViewer, utils

def build_world(robot_structure):
	world = EvoWorld.from_json(os.path.join('worlds', 'simple_environment.json'))
	world.add_from_array(
		name='robot',
		structure=robot_structure,
		x=3,
		y=1,
		connections=utils.get_full_connectivity(robot_structure)
	)
	EvoSim._has_displayed_version = True
	sim = EvoSim(world)
	viewer = EvoViewer(sim)
	viewer.track_objects('ground')

	return sim, viewer

def run_simulator(sim, controller, viewer, simulator_length, headless):
	sim.reset()
	start_position = np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[0])
	for _ in range(simulator_length):
		action = controller.control()
		sim.set_action(
			'robot',
			action
		)
		sim.step()
		if not headless:
			viewer.render('screen')
	end_position = np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[0])
	return end_position - start_position