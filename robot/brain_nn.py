import ast

import numpy as np

from configs import config
from robot.brain import Brain


class BrainNN(Brain):
    NUMBER_OF_INPUT_NEURONS = 29
    NUMBER_OF_HIDDEN_NEURONS = 10
    NUMBER_OF_OUTPUT_NEURONS = 1
    
    weights: dict
    biases: dict

    def random_brain(self, max_size, rng):
        self.weights = {'hidden': [rng.random() for _ in range(BrainNN.NUMBER_OF_INPUT_NEURONS * BrainNN.NUMBER_OF_HIDDEN_NEURONS)],
                        'output': [rng.random() for _ in range(BrainNN.NUMBER_OF_HIDDEN_NEURONS * BrainNN.NUMBER_OF_OUTPUT_NEURONS)]}

        self.biases = {'hidden': [rng.random() for _ in range(BrainNN.NUMBER_OF_HIDDEN_NEURONS)], 'output': [rng.random() for _ in range(BrainNN.NUMBER_OF_OUTPUT_NEURONS)]}

    def replace_parameters(self, parameters_string):
        self.weights = ast.literal_eval(parameters_string.split('|')[0])
        self.biases = ast.literal_eval(parameters_string.split('|')[1])

    def mutate(self, rng):
        for key, key_weights in self.weights.items():
            for i in range(len(key_weights)):
                self.weights[key][i] = np.clip(key_weights[i] + rng.normal(loc=0, scale=config.MUTATION_STD), 0, 1)

        for key, key_biases in self.biases.items():
            for i in range(len(key_biases)):
                self.biases[key][i] = np.clip(key_biases[i] + rng.normal(loc=0, scale=config.MUTATION_STD), 0, 1)

    def to_next_point(self, actuator_indices):
        next_point = {}
        for i in range(BrainNN.NUMBER_OF_HIDDEN_NEURONS):
            next_point['hidden-bias_' + str(i)] = self.biases['hidden'][i]
            for j in range(BrainNN.NUMBER_OF_INPUT_NEURONS):
                next_point['hidden_' + str(j) + '_' + str(i)] = self.weights['hidden'][i * BrainNN.NUMBER_OF_INPUT_NEURONS + j]

        for i in range(BrainNN.NUMBER_OF_OUTPUT_NEURONS):
            next_point['output-bias_' + str(i)] = self.biases['output'][i]
            for j in range(BrainNN.NUMBER_OF_HIDDEN_NEURONS):
                next_point['output_' + str(j) + '_' + str(i)] = self.weights['output'][i * BrainNN.NUMBER_OF_HIDDEN_NEURONS + j]
        return next_point

    def to_string(self):
        return f"{str(self.weights)}|{str(self.biases)}"

    @staticmethod
    def get_p_bounds(actuator_indices):
        pbounds = {}
        for i in range(BrainNN.NUMBER_OF_HIDDEN_NEURONS):
            pbounds['hidden-bias_' + str(i)] = (0, 1)
            for j in range(BrainNN.NUMBER_OF_INPUT_NEURONS):
                pbounds['hidden_' + str(j) + '_' + str(i)] = (0, 1)

        for i in range(BrainNN.NUMBER_OF_OUTPUT_NEURONS):
            pbounds['output-bias_' + str(i)] = (0, 1)
            for j in range(BrainNN.NUMBER_OF_HIDDEN_NEURONS):
                pbounds['output_' + str(j) + '_' + str(i)] = (0, 1)
        return pbounds

    @staticmethod
    def next_point_to_controller_values(next_point, actuator_indices):
        args = {'hidden_weights': np.zeros(shape=(BrainNN.NUMBER_OF_INPUT_NEURONS, BrainNN.NUMBER_OF_HIDDEN_NEURONS)), 'hidden_biases': np.zeros(shape=(1, BrainNN.NUMBER_OF_HIDDEN_NEURONS)),
                'output_weights': np.zeros(shape=(BrainNN.NUMBER_OF_HIDDEN_NEURONS, BrainNN.NUMBER_OF_OUTPUT_NEURONS)), 'output_biases': np.zeros(shape=(1, BrainNN.NUMBER_OF_OUTPUT_NEURONS))}

        for key, value in next_point.items():
            if 'hidden-bias' in key:
                position = key.split('_')[1]
                args['hidden_biases'][0][int(position)] = value * 0.2 - 0.1
                continue
            if 'output-bias' in key:
                position = key.split('_')[1]
                args['output_biases'][0][int(position)] = value * 2 - 1
                continue
            if 'hidden' in key:
                position_0, position_1 = key.split('_')[1:]
                args['hidden_weights'][int(position_0)][int(position_1)] = value * 2 - 1
                continue
            if 'output' in key:
                position_0, position_1 = key.split('_')[1:]
                args['output_weights'][int(position_0)][int(position_1)] = value * 4 - 2
                continue

        return args

    @staticmethod
    def controller_values_to_next_point(controller_values):
        next_point = {}
        for position, value in enumerate(controller_values['hidden_biases'][0]):
            adjusted_value = (value + 0.1) / 0.2
            next_point[f'hidden-bias_{position}'] = adjusted_value
        for position, value in enumerate(controller_values['output_biases'][0]):
            adjusted_value = (value + 1) / 2
            next_point[f'output-bias_{position}'] = adjusted_value
        for position_0, nested_list in enumerate(controller_values['hidden_weights']):
            for position_1, value in enumerate(nested_list):
                adjusted_value = (value + 1) / 2
                next_point[f'hidden_{position_0}_{position_1}'] = adjusted_value
        for position_0, nested_list in enumerate(controller_values['output_weights']):
            for position_1, value in enumerate(nested_list):
                adjusted_value = (value + 2) / 4
                next_point[f'output_{position_0}_{position_1}'] = adjusted_value

        return next_point