import random
from collections import deque

import numpy as np
import pandas as pd

from configs import config
from individual import Individual
from robot.body import Body
from robot.active import Brain
from robot.controller_nn import ControllerNN, RunningNorm
from robot.sensors import Sensors
from util import world, start

from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
import torch

def create_global_critic_params(num_actuators):
    input_dim = 29
    hidden_dim1 = 128   # first hidden layer
    hidden_dim2 = 64    # second hidden layer
    output_dim = 1      # Q-value
    M = num_actuators

    def rand_list(shape, low, high):
        return [[random.uniform(low, high) for _ in range(shape[1])] for _ in range(shape[0])] \
            if len(shape) == 2 else [random.uniform(low, high) for _ in range(shape[0])]

    input_size = M * (input_dim + output_dim)

    params = {
        # first layer
        'critic_hidden_weights': rand_list((input_size, hidden_dim1), -0.1, 0.1),
        'critic_hidden_biases': rand_list((hidden_dim1,), -0.01, 0.01),

        # second layer
        'critic_hidden2_weights': rand_list((hidden_dim1, hidden_dim2), -0.1, 0.1),
        'critic_hidden2_biases': rand_list((hidden_dim2,), -0.01, 0.01),

        # output layer
        'critic_output_weights': rand_list((hidden_dim2, output_dim), -0.1, 0.1),
        'critic_output_biases': rand_list((output_dim,), -0.01, 0.01)
    }

    return params



def lists_to_torch_params(param_lists):
    return {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in param_lists.items()}


rng = start.make_rng_seed()
grid = np.array([
    [0.0, 1.0, 2.0, 2.0, 4.0],
    [1.0, 1.0, 4.0, 1.0, 4.0],
    [3.0, 2.0, 3.0, 4.0, 3.0],
    [3.0, 3.0, 4.0, 0.0, 0.0],
    [2.0, 1.0, 4.0, 4.0, 0.0],
])
results = {
    'update_policy': [],
    'repetition': [],
    'learn_iteration': [],
    'fitness': [],
}

file = open('test.txt', 'w')
file.write("")
file.close()

for repetition in range(100):
    body = Body()
    body.replace_grid(grid)
    brain = Brain(5, rng)
    individual_id = "x"
    individual = Individual(individual_id, body, brain, 0, [])

    sim, viewer = world.build_world(grid, rng)
    actuator_indices = sim.get_actuator_indices('robot')

    optimizer = BayesianOptimization(
            f=None,
            pbounds=brain.get_p_bounds(actuator_indices),
            allow_duplicate_points=True,
            random_state=int(rng.integers(low=0, high=2 ** 32)),
            acquisition_function=acquisition.UpperConfidenceBound(kappa=config.LEARN_KAPPA,
                                                                  random_state=int(rng.integers(low=0, high=2 ** 32)))
        )
    optimizer.set_gp_params(kernel=Matern(nu=config.LEARN_NU, length_scale=config.LEARN_LENGTH_SCALE, length_scale_bounds="fixed"))
    optimizer.set_gp_params(alpha=config.LEARN_ALPHA)

    next_point = optimizer.suggest()
    args = brain.next_point_to_controller_values(next_point, actuator_indices)
    critic_args = create_global_critic_params(num_actuators=12)

    transition_buffer = deque(maxlen=2000)
    velocity_norm = RunningNorm()
    for bayesian_optimization_iteration in range(1000):
        combined_args = {**args, **critic_args}
        torch_args = lists_to_torch_params(combined_args)
        controller = ControllerNN(torch_args, velocity_norm)
        sensors = Sensors(grid)

        without_update = world.run_simulator(sim, controller, sensors, viewer, 500, True, False, False, None)
        with_update = world.run_simulator(sim, controller, sensors, viewer, 500, True, bayesian_optimization_iteration > 3, bayesian_optimization_iteration < 10, transition_buffer)

        file = open('test.txt', 'a')
        file.write(str(bayesian_optimization_iteration) + ": " + str(with_update - without_update) + "\n")
        file.close()

        # optimizer.register(params=next_point, target=without_update)

        # print(objective_value)
        args_torch = {
            'hidden_weights': controller.hidden_weights.detach().clone(),
            'hidden_biases': controller.hidden_biases.detach().clone(),
            'output_weights': controller.output_weights.detach().clone(),
            'output_biases': controller.output_biases.detach().clone(),
        }
        args = {k: v.cpu().numpy().tolist() for k, v in args_torch.items()}

        critic_args_torch = {
            'critic_hidden_weights': controller.critic_hidden_weights.detach().clone(),
            'critic_hidden_biases': controller.critic_hidden_biases.detach().clone(),
            'critic_hidden2_weights': controller.critic_hidden2_weights.detach().clone(),
            'critic_hidden2_biases': controller.critic_hidden2_biases.detach().clone(),
            'critic_output_weights': controller.critic_output_weights.detach().clone(),
            'critic_output_biases': controller.critic_output_biases.detach().clone()
        }

        critic_args = {k: v.cpu().numpy().tolist() for k, v in critic_args_torch.items()}

        results['update_policy'].append(True)
        results['repetition'].append(repetition + 1)
        results['learn_iteration'].append(bayesian_optimization_iteration)
        results['fitness'].append(with_update)
    pd.DataFrame(results).to_csv('test.csv')