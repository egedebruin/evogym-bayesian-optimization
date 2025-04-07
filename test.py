from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
import time

from robot.body import Body
import config
from robot.controller import Controller
from util import world
import main

rng = main.make_rng_seed()
robot_body = Body(5, 10, rng)
print(robot_body.grid)
sim, viewer = world.build_world(robot_body.grid)
print(sim.get_actuator_indices('robot'))

optimizer = BayesianOptimization(
		f=None,
		pbounds=Controller.get_p_bounds(sim.get_actuator_indices('robot')),
		allow_duplicate_points=True,
		acquisition_function=acquisition.UpperConfidenceBound(kappa=config.LEARN_KAPPA)
	)
optimizer.set_gp_params(alpha=config.LEARN_ALPHA)
optimizer.set_gp_params(kernel=Matern(nu=config.LEARN_NU, length_scale=config.LEARN_LENGTH_SCALE, length_scale_bounds="fixed"))

next_point = optimizer.suggest()
amplitudes, phase_offsets = Controller.next_point_to_controller_values(next_point)
controller = Controller(amplitudes, phase_offsets)

start_time = time.time()
world.run_simulator(sim, controller, viewer, 1500, False)
print(time.time() - start_time)