import random
from collections import deque

import numpy as np
import pandas as pd

from configs import config
from robot.body import Body
from robot.active import Brain
from robot.controller_nn import ControllerNN, RunningNorm
from robot.controller_nn_ppo import ControllerNNPPO
from robot.sensors import Sensors
from util import world, start

from bayes_opt import BayesianOptimization, acquisition
from sklearn.gaussian_process.kernels import Matern
import torch

ONLY_BO = 'bo'
ONLY_RL = 'rl'
BO_AND_RL = 'borl'
PPO = 'PPO'
DDPG = 'DDPG'

def create_global_critic_params(num_actuators, rl_type):
    input_dim = 29
    hidden_dim1 = 128   # first hidden layer
    hidden_dim2 = 64    # second hidden layer
    output_dim = 1      # Q-value
    M = num_actuators

    def rand_list(shape, low, high):
        return [[random.uniform(low, high) for _ in range(shape[1])] for _ in range(shape[0])] \
            if len(shape) == 2 else [random.uniform(low, high) for _ in range(shape[0])]

    input_size = M * (input_dim + output_dim)
    if rl_type == PPO:
        input_size = M * input_dim

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

def everybody_do_the_vroom(rng, grid, strategy, rl_type):
    body = Body()
    body.replace_grid(grid)
    brain = Brain(5, rng)

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

    critic_args = create_global_critic_params(num_actuators=22, rl_type=rl_type)

    transition_buffer = deque(maxlen=2000)
    velocity_norm = RunningNorm()

    best_value = 0
    best_args = None
    values = []
    for iteration in range(200):
        if iteration == 0 or strategy == ONLY_BO or (strategy == BO_AND_RL and iteration < 50):
            next_point = optimizer.suggest()
            args = brain.next_point_to_controller_values(next_point, actuator_indices)
        if strategy == BO_AND_RL and iteration == 50:
            args = best_args

        combined_args = {**args, **critic_args}
        torch_args = lists_to_torch_params(combined_args)
        if rl_type == PPO:
            controller = ControllerNNPPO(torch_args, velocity_norm)
        else:
            controller = ControllerNN(torch_args, velocity_norm)
        sensors = Sensors(grid)

        update_critic = False if strategy == ONLY_BO else iteration > 3
        update_policy = False if strategy == ONLY_BO else (iteration > 3 if strategy == ONLY_RL else iteration >= 50)
        update_norm = False if strategy == ONLY_BO else iteration < 5
        with_update = world.run_simulator(sim, controller, sensors, viewer, 500, False,
                                          update_critic, update_policy, update_norm, rl_type, transition_buffer)

        values.append(with_update)

        if strategy == ONLY_BO or (strategy == BO_AND_RL and iteration < 50):
            optimizer.register(params=next_point, target=with_update)
            if best_args is None or with_update > best_value:
                best_value = with_update
                best_args = args

        if strategy == ONLY_RL or (strategy == BO_AND_RL and iteration >= 50):
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
    return values

def main():
    rng = start.make_rng_seed()
    grid = np.array([
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 0.0, 3.0, 3.0],
        [3.0, 3.0, 0.0, 3.0, 3.0],
        [3.0, 3.0, 0.0, 3.0, 3.0],
    ])
    results = {
        'strategy': [],
        'repetition': [],
        'learn_iteration': [],
        'fitness': [],
        'rl_type': [],
    }

    for repetition in range(1, 2):
        print('REPETITION: ', repetition)
        for strategy in [ONLY_RL, ONLY_BO, BO_AND_RL]:
            if strategy == ONLY_RL:
                values = everybody_do_the_vroom(rng, grid, strategy, PPO)
                for learn_iteration, value in enumerate(values):
                    results['strategy'].append(strategy)
                    results['repetition'].append(repetition)
                    results['learn_iteration'].append(learn_iteration)
                    results['fitness'].append(value)
                    results['rl_type'].append(PPO)
            continue
            values = everybody_do_the_vroom(rng, grid, strategy, DDPG)
            for learn_iteration, value in enumerate(values):
                results['strategy'].append(strategy)
                results['repetition'].append(repetition)
                results['learn_iteration'].append(learn_iteration)
                results['fitness'].append(value)
                results['rl_type'].append(DDPG)

    pd.DataFrame(results).to_csv('results.csv')

if __name__ == "__main__":
    main()