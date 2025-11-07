import random
from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from robot.brain_nn import BrainNN


class RL(ABC, nn.Module):

    @staticmethod
    def from_args(args, name):
        if name not in args:
            raise KeyError(f"Missing parameter '{name}' in args")
        return nn.Parameter(args[name].detach().clone())

    def __init__(self, num_actuators):
        super().__init__()

        self.critic_hidden_weights = None
        self.critic_hidden_biases = None
        self.critic_hidden2_weights = None
        self.critic_hidden2_biases = None
        self.critic_output_weights = None
        self.critic_output_biases = None

        self.do_update_policy = False
        self.do_update_critic = False
        self.do_update_norm = True

        args = self.create_global_critic_params(num_actuators)
        args = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
                for k, v in args.items()}
        self.policy_optimizer = None
        self.critic_optimizer = None
        self.set_critic_parameters(args)
        self.set_critic_optimizer(self.critic_lr)


    def set_critic_optimizer(self, critic_lr):
        critic_params = [self.critic_hidden_weights, self.critic_hidden_biases,
                         self.critic_hidden2_weights, self.critic_hidden2_biases,
                         self.critic_output_weights, self.critic_output_biases]

        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

    def set_policy_optimizer(self, policy_weights, policy_lr):
        self.policy_optimizer = torch.optim.Adam(policy_weights, lr=policy_lr)

    def set_critic_parameters(self, args):
        # Critic weights
        self.critic_hidden_weights = RL.from_args(args, 'critic_hidden_weights')
        self.critic_hidden_biases = RL.from_args(args, 'critic_hidden_biases')
        self.critic_hidden2_weights = RL.from_args(args, 'critic_hidden2_weights')
        self.critic_hidden2_biases = RL.from_args(args, 'critic_hidden2_biases')

        self.critic_output_weights = RL.from_args(args, 'critic_output_weights')
        self.critic_output_biases = RL.from_args(args, 'critic_output_biases')

    def post_action(self, policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer):
        return

    @abstractmethod
    def control(self, sensor_input, policy_weights):
        pass

    @abstractmethod
    def get_input_size(self, num_actuators, policy_input, policy_output):
        pass

    def post_rollout(self, last_sensor_input, policy_weights):
        return

    def forward_critic(self, critic_input):
        h1 = F.relu(critic_input @ self.critic_hidden_weights + self.critic_hidden_biases)
        h2 = F.relu(h1 @ self.critic_hidden2_weights + self.critic_hidden2_biases)
        q_value = h2 @ self.critic_output_weights + self.critic_output_biases
        return q_value.squeeze(-1)

    def set_update_networks(self, update_critic, update_policy):
        self.do_update_critic = update_critic
        self.do_update_policy = update_policy

    def create_global_critic_params(self, num_actuators):
        input_dim = BrainNN.NUMBER_OF_INPUT_NEURONS
        hidden_dim1 = 128  # first hidden layer
        hidden_dim2 = 64  # second hidden layer
        output_dim = 1  # Q-value

        def rand_list(shape, low, high):
            return [[random.uniform(low, high) for _ in range(shape[1])] for _ in range(shape[0])] \
                if len(shape) == 2 else [random.uniform(low, high) for _ in range(shape[0])]

        params = {
            # first layer
            'critic_hidden_weights': rand_list((self.get_input_size(num_actuators, input_dim, output_dim), hidden_dim1), -0.1, 0.1),
            'critic_hidden_biases': rand_list((hidden_dim1,), -0.01, 0.01),

            # second layer
            'critic_hidden2_weights': rand_list((hidden_dim1, hidden_dim2), -0.1, 0.1),
            'critic_hidden2_biases': rand_list((hidden_dim2,), -0.01, 0.01),

            # output layer
            'critic_output_weights': rand_list((hidden_dim2, output_dim), -0.1, 0.1),
            'critic_output_biases': rand_list((output_dim,), -0.01, 0.01)
        }

        return params

    def clip_policy_params(self, policy_weights):
        # Hidden weights: [-1, 1]
        policy_weights['hidden_weights'].data.clamp_(-1.0, 1.0)

        # Hidden biases: [-0.1, 0.1]
        policy_weights['hidden_biases'].data.clamp_(-0.1, 0.1)

        # Output weights: [-2, 2]
        policy_weights['output_weights'].data.clamp_(-2.0, 2.0)

        # Output biases: [-1, 1]
        policy_weights['output_biases'].data.clamp_(-1.0, 1.0)
