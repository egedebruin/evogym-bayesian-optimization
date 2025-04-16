import math
from collections import defaultdict
from typing import Any

import numpy as np

import config

class Sensors:
    sensor_grid_to_sensor_index: dict
    voxel_index_to_sensor_index: defaultdict[Any, list]

    def __init__(self, robot_structure):
        self.robot_structure = robot_structure
        self.get_sensor_grid()
        self.get_voxel_to_sensor_index()

    def get_sensor_grid(self):
        sensor_index = 0
        self.sensor_grid_to_sensor_index = {}
        for x_sensor in range(config.GRID_LENGTH + 1):
            for y_sensor in range(config.GRID_LENGTH + 1):
                if x_sensor < config.GRID_LENGTH and y_sensor < config.GRID_LENGTH and self.robot_structure[x_sensor, y_sensor] > 0:
                    # Add sensors to sensor grid if it doesn't exist yet
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            if (x_sensor + dx, y_sensor + dy) in self.sensor_grid_to_sensor_index:
                                continue
                            self.sensor_grid_to_sensor_index[(x_sensor + dx, y_sensor + dy)] = sensor_index
                            sensor_index += 1

    def get_voxel_to_sensor_index(self):
        self.voxel_index_to_sensor_index = defaultdict(list)
        for x_voxel in range(config.GRID_LENGTH):
            for y_voxel in range(config.GRID_LENGTH):
                if self.robot_structure[x_voxel, y_voxel] == 0:
                    continue
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        self.voxel_index_to_sensor_index[x_voxel * config.GRID_LENGTH + y_voxel].append(
                            self.sensor_grid_to_sensor_index[(x_voxel + dx, y_voxel + dy)])

    def get_input_from_sensors(self, positions, velocities, actuator_indices, current_time):
        input_vectors = []
        for actuator_index in actuator_indices:
            voxel_sizes = []
            voxel_velocities = []
            actuator_x, actuator_y = (actuator_index // config.GRID_LENGTH, actuator_index % config.GRID_LENGTH)
            for x_neighbor in [-1, 0, 1]:
                for y_neighbor in [-1, 0, 1]:
                    neighbor_x, neighbor_y = (actuator_x + x_neighbor, actuator_y + y_neighbor)
                    neighbor_index = neighbor_x * config.GRID_LENGTH + neighbor_y
                    if neighbor_x < 0 or neighbor_x >= config.GRID_LENGTH or neighbor_y < 0 or neighbor_y >= config.GRID_LENGTH or neighbor_index not in self.voxel_index_to_sensor_index.keys():
                        voxel_sizes.append(0)
                        voxel_velocities.append(0)
                        voxel_velocities.append(0)
                        continue

                    sensor_indices = self.voxel_index_to_sensor_index[neighbor_index]
                    corners = []
                    velocities_x = []
                    velocities_y = []
                    for sensor_index in sensor_indices:
                        corners.append((positions[0][sensor_index], positions[1][sensor_index]))
                        velocities_x.append(velocities[0][sensor_index])
                        velocities_y.append(velocities[1][sensor_index])
                    voxel_sizes.append(Sensors.rectangle_size(corners))
                    voxel_velocities.append(sum(velocities_x) / len(velocities_x))
                    voxel_velocities.append(sum(velocities_y) / len(velocities_y))
            input_vectors.append(np.array(voxel_sizes + voxel_velocities + [current_time % 25]))
        return input_vectors

    @staticmethod
    def distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def rectangle_size(corners):
        a, b, c, d = corners
        width = Sensors.distance(a, b)
        height = Sensors.distance(a, c)
        return width * height