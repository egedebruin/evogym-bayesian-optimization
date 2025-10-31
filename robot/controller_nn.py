import numpy as np
import torch
import torch.nn.functional as F

from reinforcement_learning.rl import RL
from robot.controller import Controller


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

    def set_rl_agent(self, rl_agent):
        self.rl_agent = rl_agent

    def control(self, sensor_input):
        """
        sensor_inputs: list of length M, each element = list/array of input_dim features
        """
        with torch.no_grad():
            sensor_tensor = torch.tensor(np.array(sensor_input), dtype=torch.float32)

            hidden = F.relu(sensor_tensor @ self.hidden_weights + self.hidden_biases)
            raw_output = hidden @ self.output_weights + self.output_biases
            raw_action = torch.sigmoid(raw_output).cpu().numpy()

        return raw_action, {}

    def adjust_sensor_input(self, sensor_input):
        if self.rl_agent is None:
            return sensor_input

        return self.rl_agent.norm_sensor_input(sensor_input)

    def post_action(self, sim, sensor_input, normalized_sensor_input, raw_action, previous_position, sensors, buffer):
        if self.rl_agent is None:
            return

        reward = (np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[0]) - np.mean(previous_position[0])) * 10

        next_sensor_input = sensors.get_input_from_sensors(sim)
        normalized_next_sensor_input = self.rl_agent.norm_sensor_input(next_sensor_input)
        self.rl_agent.post_action(self.policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer)

    @staticmethod
    def get_parameters_from_args(args):
        return {
            'hidden_weights': RL.from_args(args, 'hidden_weights'),
            'hidden_biases': RL.from_args(args, 'hidden_biases'),
            'output_weights': RL.from_args(args, 'output_weights'),
            'output_biases': RL.from_args(args, 'output_biases')
        }