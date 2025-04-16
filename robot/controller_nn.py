import numpy as np

from robot.controller import Controller

class ControllerNN(Controller):

    def __init__(self, args):
        self.hidden_weights = args['hidden_weights']
        self.hidden_biases = args['hidden_biases']
        self.output_weights = args['output_weights']
        self.output_biases = args['output_biases']

    def control(self, sensor_inputs):
        output = []
        for sensor_input in sensor_inputs:
            hidden_pre_activation = sensor_input @ self.hidden_weights + self.hidden_biases
            hidden_activation = ControllerNN.relu(hidden_pre_activation)
            output_pre_activation = hidden_activation @ self.output_weights + self.output_biases
            output.append(ControllerNN.sigmoid(output_pre_activation)[0] + 0.6)
        return np.array(output)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )