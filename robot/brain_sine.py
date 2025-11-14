import ast

import numpy as np

from configs import config
from robot.brain import Brain


class BrainSine(Brain):
    NUMBER_OF_CONTROLLER_VALUES = 4

    grid: np.array

    def random_brain(self, max_size, rng):
        self.grid = np.full((max_size, max_size, BrainSine.NUMBER_OF_CONTROLLER_VALUES), 0.0)

        for i in range(max_size):
            for j in range(max_size):
                for k in range(BrainSine.NUMBER_OF_CONTROLLER_VALUES):
                    self.grid[i, j, k] = rng.random()

    def replace_parameters(self, parameters_string):
        self.grid = np.array(ast.literal_eval(parameters_string))

    def mutate(self, rng):
        max_size = self.grid.shape[0]
        for i in range(max_size):
            for j in range(max_size):
                for k in range(BrainSine.NUMBER_OF_CONTROLLER_VALUES):
                    self.grid[i, j, k] = np.clip(self.grid[i, j, k] + rng.normal(loc=0, scale=config.MUTATION_STD), 0, 1)

    def to_next_point(self, actuator_indices):
        max_size = self.grid.shape[0]
        next_point = {}
        for index in actuator_indices:
            next_point['amplitude_' + str(index)] = self.grid[index % max_size, int(index / max_size), 0]
            next_point['phase_offset_sin_' + str(index)] = self.grid[index % max_size, int(index / max_size), 1]
            next_point['phase_offset_cos_' + str(index)] = self.grid[index % max_size, int(index / max_size), 2]
            next_point['angular_offset_' + str(index)] = self.grid[index % max_size, int(index / max_size), 3]
        return next_point

    def update_experience_with_actuator_indices(self, experience, actuator_indices):
        max_size = self.grid.shape[0]

        updated_experience = []
        for brain_sample, objective_value in experience:
            new_brain_sample = {}

            for actuator_index in actuator_indices:
                amplitude_key = f'amplitude_{actuator_index}'
                phase_offset_sin_key = f'phase_offset_sin_{actuator_index}'
                phase_offset_cos_key = f'phase_offset_cos_{actuator_index}'
                angular_offset_key = f'angular_offset_{actuator_index}'

                for actuator_key, grid_position in \
                        [(amplitude_key, 0), (phase_offset_sin_key, 1), (phase_offset_cos_key, 2), (angular_offset_key, 3)]:
                    if actuator_key in brain_sample:
                        new_brain_sample[actuator_key] = brain_sample[actuator_key]
                    else:
                        new_brain_sample[actuator_key] = self.grid[actuator_index % max_size, actuator_index // max_size, grid_position]
            updated_experience.append((new_brain_sample, objective_value))

        return updated_experience

    def to_string(self):
        return self.grid.tolist()

    @staticmethod
    def get_p_bounds(actuator_indices):
        pbounds = {}
        for index in actuator_indices:
            pbounds['amplitude_' + str(index)] = (0, 1)
            pbounds['phase_offset_sin_' + str(index)] = (0, 1)
            pbounds['phase_offset_cos_' + str(index)] = (0, 1)
            pbounds['angular_offset_' + str(index)] = (0, 1)
        return pbounds

    @staticmethod
    def next_point_to_controller_values(next_point, actuator_indices):
        args = {'amplitudes': [], 'phase_offsets': [], 'angular_offsets': []}

        for index in actuator_indices:
            args['amplitudes'].append(next_point['amplitude_' + str(index)] * 0.5)
            args['phase_offsets'].append(BrainSine.sin_cos_to_angle(
                next_point['phase_offset_sin_' + str(index)] * 2 - 1,
                next_point['phase_offset_cos_' + str(index)] * 2 - 1
            ))
            args['angular_offsets'].append(next_point['angular_offset_' + str(index)] - 0.5)

        return args

    @staticmethod
    def controller_values_to_next_point(controller_values):
        raise NotImplementedError


    @staticmethod
    def sin_cos_to_angle(sin_val, cos_val):
        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi  # Adjust to the range [0, 2π]
        return angle
