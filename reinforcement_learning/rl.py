import random
from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RL(ABC, nn.Module):

    @staticmethod
    def from_args(args, name):
        if name not in args:
            raise KeyError(f"Missing parameter '{name}' in args")
        return nn.Parameter(args[name].detach().clone())

    def __init__(self, velocity_norm):
        super().__init__()
        self.critic_hidden_weights = None
        self.critic_hidden_biases = None
        self.critic_hidden2_weights = None
        self.critic_hidden2_biases = None
        self.critic_output_weights = None
        self.critic_output_biases = None

        self.velocity_indices = list(range(9, 27))
        self.velocity_norm = velocity_norm

        self.do_update_policy = False
        self.do_update_critic = False
        self.do_update_norm = True

    def set_critic_parameters(self, args):
        # Critic weights
        self.critic_hidden_weights = RL.from_args(args, 'critic_hidden_weights')
        self.critic_hidden_biases = RL.from_args(args, 'critic_hidden_biases')
        self.critic_hidden2_weights = RL.from_args(args, 'critic_hidden2_weights')
        self.critic_hidden2_biases = RL.from_args(args, 'critic_hidden2_biases')

        self.critic_output_weights = RL.from_args(args, 'critic_output_weights')
        self.critic_output_biases = RL.from_args(args, 'critic_output_biases')

    @abstractmethod
    def post_action(self, policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer):
        pass

    def forward_critic(self, critic_input):
        h1 = F.relu(critic_input @ self.critic_hidden_weights + self.critic_hidden_biases)
        h2 = F.relu(h1 @ self.critic_hidden2_weights + self.critic_hidden2_biases)
        q_value = h2 @ self.critic_output_weights + self.critic_output_biases
        return q_value.squeeze(-1)

    def update_norm(self, sensor_input, next_sensor_input):
        if not self.do_update_norm:
            return
        vel_indices = list(range(9, 27))

        # Convert list of arrays to single NumPy array first
        sensor_input = torch.as_tensor(np.array(sensor_input, dtype=np.float32))
        next_sensor_input = torch.as_tensor(np.array(next_sensor_input, dtype=np.float32))

        # sensor_input: (M, input_dim) for one transition
        vels = sensor_input[:, vel_indices].cpu().numpy()
        mask = (vels != 0)
        self.velocity_norm.update(vels, mask=mask)

        next_vels = next_sensor_input[:, vel_indices].cpu().numpy()
        next_mask = (next_vels != 0)
        self.velocity_norm.update(next_vels, mask=next_mask)

    def norm_sensor_input(self, sensor_input):
        processed_inputs = []
        vel_indices = list(range(9, 27))

        for sensors in sensor_input:
            sensors = np.array(sensors, dtype=np.float32)

            # Just normalize using *existing* stats, never update here
            vels = sensors[vel_indices]
            mask = (vels != 0)
            vels_normed = self.velocity_norm.normalize(vels, mask=mask)
            sensors[vel_indices] = vels_normed

            processed_inputs.append(sensors)
        return processed_inputs

    def set_update_boolean_values(self, iteration):
        if iteration > 3:
            self.do_update_critic = True
            self.do_update_policy = True

        if iteration > 5:
            self.do_update_norm = True

    @staticmethod
    def create_global_critic_params(num_actuators):
        input_dim = 29
        hidden_dim1 = 128  # first hidden layer
        hidden_dim2 = 64  # second hidden layer
        output_dim = 1  # Q-value

        def rand_list(shape, low, high):
            return [[random.uniform(low, high) for _ in range(shape[1])] for _ in range(shape[0])] \
                if len(shape) == 2 else [random.uniform(low, high) for _ in range(shape[0])]

        input_size = num_actuators * (input_dim + output_dim)

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
