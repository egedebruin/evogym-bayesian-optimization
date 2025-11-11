import numpy as np
import torch
import torch.nn.functional as F

from configs import config
from robot.controller import Controller
from robot.running_norm import RunningNorm


class ControllerNN(Controller):

    def __init__(self, args):
        self.hidden_weights = args['hidden_weights']
        self.hidden_biases = args['hidden_biases']
        self.output_weights = args['output_weights']
        self.output_biases = args['output_biases']

        self.policy_weights = {
            'hidden_weights': self.hidden_weights,
            'hidden_biases': self.hidden_biases,
            'output_weights': self.output_weights,
            'output_biases': self.output_biases
        }

        self.rl_agent = None
        self.velocity_indices = list(range(9, 27))
        self.velocity_norm = RunningNorm(0.0, 100.0, 'linear')
        self.package_indices = list(range(30, 32))
        self.package_norm = RunningNorm(0.0, 1.0, 'tanh', scale=10)

    def set_rl_agent(self, rl_agent):
        self.rl_agent = rl_agent

    def control(self, sensor_input):
        """
        sensor_inputs: list of length M, each element = list/array of input_dim features
        """
        if self.rl_agent is not None:
            return self.rl_agent.control(sensor_input, self.policy_weights)
        with torch.no_grad():
            sensor_tensor = torch.tensor(np.array(sensor_input), dtype=torch.float32)

            hidden = F.relu(sensor_tensor @ self.hidden_weights + self.hidden_biases)
            raw_output = hidden @ self.output_weights + self.output_biases
            raw_action = torch.sigmoid(raw_output).cpu().numpy()

        return raw_action

    def adjust_sensor_input(self, sensor_input):
        return self.norm_sensor_input(sensor_input)

    def norm_sensor_input(self, sensor_input):
        processed_inputs = []

        for sensors in sensor_input:
            new_sensors = np.array(sensors, dtype=np.float32)

            # Just normalize using *existing* stats, never update here
            vels = new_sensors[self.velocity_indices]
            mask = (vels != 0)
            vels_normed = self.velocity_norm.normalize(vels, mask=mask)
            new_sensors[self.velocity_indices] = vels_normed

            if config.ENVIRONMENT in ['carry', 'catch']:
                package_sensors = new_sensors[self.package_indices]
                package_sensors_normed = self.package_norm.normalize(package_sensors)
                new_sensors[self.package_indices] = package_sensors_normed

            processed_inputs.append(new_sensors)
        return processed_inputs

    def post_action(self, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer):
        if self.rl_agent is None:
            return
        self.rl_agent.post_action(self.policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer)

    def post_rollout(self, last_sensor_input):
        if self.rl_agent is None:
            return
        self.rl_agent.post_rollout(last_sensor_input, self.policy_weights)