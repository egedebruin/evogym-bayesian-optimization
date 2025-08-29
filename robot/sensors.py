import math
from collections import defaultdict
from typing import Any

import numpy as np

from configs import config
from robot.touch_sensor_util import detect_ground_contact

class Sensors:
    sensor_grid_to_sensor_index: dict
    voxel_index_to_sensor_index: defaultdict[Any, list]

    def __init__(self, robot_structure):
        self.robot_structure = robot_structure
        self._get_sensor_grid()
        self._get_voxel_to_sensor_index()

    def _get_sensor_grid(self):
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

    def _get_voxel_to_sensor_index(self):
        self.voxel_index_to_sensor_index = defaultdict(list)
        for x_voxel in range(config.GRID_LENGTH):
            for y_voxel in range(config.GRID_LENGTH):
                if self.robot_structure[x_voxel, y_voxel] == 0:
                    continue
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        self.voxel_index_to_sensor_index[x_voxel * config.GRID_LENGTH + y_voxel].append(
                            self.sensor_grid_to_sensor_index[(x_voxel + dx, y_voxel + dy)])

    def get_input_from_sensors(self, sim):
        robot_positions = sim.object_pos_at_time(sim.get_time(), 'robot')
        robot_velocities = sim.object_vel_at_time(sim.get_time(), 'robot')
        robot_actuator_indices = sim.get_actuator_indices('robot')
        ground_positions = sim.object_pos_at_time(sim.get_time(), 'ground')
        current_time = sim.get_time()

        input_vectors = []
        for actuator_index in robot_actuator_indices:
            sensor_input = np.array([])

            # Size-and-speed sensors
            actuator_input = self._get_input_actuator(actuator_index, robot_positions, robot_velocities)
            sensor_input = np.concatenate((sensor_input, actuator_input))

            # Ground-touch sensor
            contact = detect_ground_contact(robot_positions, ground_positions, self.voxel_index_to_sensor_index, [actuator_index])
            # sensor_input = np.concatenate((sensor_input, [1.0 if contact else 0.0]))

            if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
                # Distance-to-package sensor
                package_positions = sim.object_pos_at_time(sim.get_time(), 'package')
                package_input = self._get_input_package(actuator_index, robot_positions, package_positions)
                sensor_input = np.concatenate((sensor_input, package_input))

            # Time sensor
            # Cyclic input: map 0..25 → 0..2π
            cyc = current_time % 25  # gives 0..25
            theta = 2 * np.pi * cyc / 25  # normalize to 0..2π

            cyc_sin = np.sin(theta)
            cyc_cos = np.cos(theta)

            # Append to your input vector
            input_vectors.append(np.concatenate((sensor_input, [cyc_sin, cyc_cos])))

        return input_vectors

    def _get_input_actuator(self, actuator_index, robot_positions, robot_velocities):
        voxel_sizes = []
        voxel_velocities = []
        actuator_x, actuator_y = (actuator_index // config.GRID_LENGTH, actuator_index % config.GRID_LENGTH)
        for x_neighbor in [-1, 0, 1]:
            for y_neighbor in [-1, 0, 1]:
                voxel_size, voxel_velocity_x, voxel_velocity_y = self._get_input_from_neighbor(
                    actuator_x + x_neighbor,
                    actuator_y + y_neighbor,
                    robot_positions,
                    robot_velocities)
                voxel_sizes.append(voxel_size)
                voxel_velocities.append(voxel_velocity_x)
                voxel_velocities.append(voxel_velocity_y)
        return np.array(voxel_sizes + voxel_velocities)

    def _get_input_package(self, actuator_index, robot_positions, package_positions):
        sensor_indices = self.voxel_index_to_sensor_index[actuator_index]
        minimum_x_distance = math.inf
        minimum_y_distance = math.inf
        for sensor_index in sensor_indices:
            robot_sensor_position = (robot_positions[0][sensor_index], robot_positions[1][sensor_index])
            for package_index in range(len(package_positions[0])):
                package_sensor_position = (package_positions[0][package_index], package_positions[1][package_index])

                x_distance = package_sensor_position[0] - robot_sensor_position[0]
                y_distance = package_sensor_position[1] - robot_sensor_position[1]

                if abs(x_distance) < abs(minimum_x_distance):
                    minimum_x_distance = x_distance
                if abs(y_distance) < abs(minimum_y_distance):
                    minimum_y_distance = y_distance
        return np.array([minimum_x_distance, minimum_y_distance])

    def _get_input_from_neighbor(self, neighbor_x, neighbor_y, positions, velocities):
        neighbor_index = neighbor_x * config.GRID_LENGTH + neighbor_y
        if (neighbor_x < 0 or neighbor_x >= config.GRID_LENGTH or
                neighbor_y < 0 or neighbor_y >= config.GRID_LENGTH or
                neighbor_index not in self.voxel_index_to_sensor_index.keys()):
            return 0, 0, 0

        sensor_indices = self.voxel_index_to_sensor_index[neighbor_index]
        corners = []
        velocities_x = []
        velocities_y = []
        for sensor_index in sensor_indices:
            corners.append((positions[0][sensor_index], positions[1][sensor_index]))
            velocities_x.append(velocities[0][sensor_index])
            velocities_y.append(velocities[1][sensor_index])

        return (Sensors.rectangle_size(corners),
                sum(velocities_x) / len(velocities_x),
                sum(velocities_y) / len(velocities_y))

    @staticmethod
    def distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def rectangle_size(corners):
        a, b, c, d = corners
        width = Sensors.distance(a, b)
        height = Sensors.distance(a, c)
        return width * height