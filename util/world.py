import json
import uuid

import numpy as np
import os

from evogym import EvoWorld, EvoSim, EvoViewer, utils, WorldObject

from configs import config
from util import writer
from worlds import random_environment_creator, random_steps_environment_creator


def build_world(robot_structure, rng=None, world=None, generation_index=None):
    if world is None:
        world, _ = get_environment(rng, generation_index=generation_index)
        world = add_extra_attributes(world, rng)
    x, y = start_position_robot()
    world.add_from_array(
        name='robot',
        structure=robot_structure,
        x=x,
        y=y,
        connections=utils.get_full_connectivity(robot_structure)
    )
    EvoSim._has_displayed_version = True
    sim = EvoSim(world)
    viewer = EvoViewer(sim)
    tracker(viewer)

    return sim, viewer


def run_simulator(sim, controller, sensors, viewer, simulator_length, headless, generation_index=0, transition_buffer=None):
    sim.reset()
    extra_metrics = []
    start_position = sim.object_pos_at_time(sim.get_time(), 'robot')
    extra_metrics = extra_metrics_for_objective_value('before', sim, extra_metrics)
    previous_position = None
    sensor_input = None
    normalized_sensor_input = None
    raw_action = None
    current_extra_metrics = None

    for simulation_step in range(simulator_length):

        # ----- During phase metrics -----
        if simulation_step > 50:
            extra_metrics = extra_metrics_for_objective_value('during', sim, extra_metrics)

        # ----- Action step -----
        if simulation_step % 5 == 0:
            sensor_input = sensors.get_input_from_sensors(sim, generation_index)
            normalized_sensor_input = controller.adjust_sensor_input(sensor_input)

            raw_action = controller.control(normalized_sensor_input)
            adjusted_action = raw_action + 0.6

            sim.set_action('robot', adjusted_action)

            previous_position = sim.object_pos_at_time(sim.get_time(), 'robot')
            current_extra_metrics = []
            current_extra_metrics = extra_metrics_for_objective_value('before', sim, current_extra_metrics)

        # ----- Learning step -----
        if simulation_step % 5 == 4:
            current_position = sim.object_pos_at_time(sim.get_time(), 'robot')
            current_extra_metrics = extra_metrics_for_objective_value('after', sim, current_extra_metrics)
            reward = calculate_reward(previous_position, current_position, current_extra_metrics, generation_index) * 10

            next_sensor_input = sensors.get_input_from_sensors(sim, generation_index)
            normalized_next_sensor_input = controller.adjust_sensor_input(next_sensor_input)
            controller.post_action(sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, transition_buffer)

        # Step simulation
        sim.step()

        # Render if not headless
        if not headless:
            viewer.render('screen')

    controller.post_rollout(sensor_input)

    end_position = sim.object_pos_at_time(sim.get_time(), 'robot')
    extra_metrics = extra_metrics_for_objective_value('after', sim, extra_metrics)
    return calculate_objective_value(start_position, end_position, extra_metrics, generation_index)


# Below are Environment specific things
def start_position_robot():
    if config.ENVIRONMENT == 'catch':
        return 16, 1
    if config.ENVIRONMENT in ['bidirectional', 'bidirectional2']:
        return 100, 1
    return 1, 1

def get_environment(rng, previous_heights=None, generation_index=None):
    heights = []
    if config.ENVIRONMENT in ['simple', 'jump', 'catch', 'bidirectional', 'bidirectional2']:
        world = EvoWorld.from_json(os.path.join('worlds', 'simple_environment.json'))
    elif config.ENVIRONMENT == 'climb':
        world = EvoWorld.from_json(os.path.join('worlds', 'climb_environment.json'))
    elif config.ENVIRONMENT == 'carry':
        world = EvoWorld.from_json(os.path.join('worlds', 'carry_environment.json'))
    elif config.ENVIRONMENT == 'rugged':
        world = EvoWorld.from_json(os.path.join('worlds', 'rugged_environment.json'))
    elif config.ENVIRONMENT == 'steps':
        world = EvoWorld.from_json(os.path.join('worlds', 'steps_environment.json'))
    elif config.ENVIRONMENT == 'randomsteps':
        platform_length = config.STEPS_CHANGE_DEGREE[generation_index % 2]
        contents, heights = random_steps_environment_creator.make(platform_length)
        world = create_random_file(contents)
    elif config.ENVIRONMENT in ['random', 'changing']:
        if config.ENVIRONMENT == 'random' or previous_heights is None or len(previous_heights) == 0:
            contents, heights = random_environment_creator.make(rng)
        else:
            contents, heights = random_environment_creator.change(rng, previous_heights)

        if config.WRITE_RANDOM_ENV:
            writer.write_to_environments_file(";".join(str(e) for e in heights))

        world = create_random_file(contents)
    else:
        raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")
    return world, heights

def create_random_file(contents):
    filename = f'{str(uuid.uuid4())}.json'
    directory = f'worlds/random/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + filename, 'w') as outfile:
        json.dump(contents, outfile, indent=4)
    world = EvoWorld.from_json(directory + filename)
    os.remove(directory + filename)
    return world


def tracker(viewer):
    if config.ENVIRONMENT in ['carry', 'catch']:
        viewer.track_objects('package', 'robot')
    else:
        viewer.track_objects('ground')

def add_extra_attributes(world, rng):
    if config.ENVIRONMENT == 'catch':
        offset_x = int(rng.integers(-6, 5)) # NOTE: This is always the same per generation
        offset_y = int(rng.integers(0, 6))

        if config.WRITE_RANDOM_ENV:
            writer.write_to_environments_file(f'{str(offset_x)};{str(offset_y)}')

        package = WorldObject.from_json(os.path.join('worlds', 'package.json'))
        package.set_pos(10 + offset_x, 41 + offset_y)
        package.rename('package')
        world.add_object(package)

        peg1 = WorldObject.from_json(os.path.join('worlds', 'peg.json'))
        peg1.set_pos(6 + offset_x, 39 + offset_y)
        peg1.rename('peg1')
        world.add_object(peg1)

        peg2 = WorldObject.from_json(os.path.join('worlds', 'peg.json'))
        peg2.set_pos(8 + offset_x, 25 + offset_y)
        peg2.rename('peg2')
        world.add_object(peg2)

    return world

def calculate_objective_value(start_position, end_position, extra_metrics, generation_index):
    if config.ENVIRONMENT in ['simple', 'rugged', 'steps', 'random', 'changing', 'randomsteps']:
        return np.mean(end_position[0]) - np.mean(start_position[0])
    elif config.ENVIRONMENT in ['bidirectional', 'bidirectional2']:
        if generation_index % 2 == 0:
            return np.mean(end_position[0]) - np.mean(start_position[0])
        return np.mean(start_position[0]) - np.mean(end_position[0])
    elif config.ENVIRONMENT == 'climb':
        return np.mean(end_position[1]) - np.mean(start_position[1])
    elif config.ENVIRONMENT == 'jump':
        return np.max(extra_metrics)
    elif config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
        if np.min(extra_metrics[1][1]) < 1.5:
            return -abs(np.mean(extra_metrics[1][0]) - np.mean(end_position[0]))
        return np.mean(extra_metrics[1][0]) - np.mean(extra_metrics[0][0])
    else:
        raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")

def calculate_reward(start_position, end_position, extra_metrics, generation_index):
    if config.ENVIRONMENT in ['simple', 'rugged', 'steps', 'random', 'bidirectional', 'bidirectional2', 'climb', 'changing', 'randomsteps']:
        return calculate_objective_value(start_position, end_position, extra_metrics, generation_index)
    elif config.ENVIRONMENT == 'jump':
        return np.mean(end_position[1]) - np.mean(start_position[1])
    elif config.ENVIRONMENT in ['carry', 'catch']:
        # positive reward for moving forward
        reward = 0
        if 1.5 < np.min(extra_metrics[1][1]) < 8.5:
            reward += (np.mean(extra_metrics[1][0]) - np.mean(extra_metrics[0][0]))

        # negative reward for robot/block separating
        reward += abs(np.mean(start_position[0]) - np.mean(extra_metrics[0][0])) - abs(
            np.mean(end_position[0]) - np.mean(extra_metrics[1][0]))
        return reward
    else:
        raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")

def extra_metrics_for_objective_value(timing, sim, extra):
    if config.ENVIRONMENT in ['carry', 'catch']:
        if timing == 'before' or timing == 'after':
            extra.append(sim.object_pos_at_time(sim.get_time(), 'package'))

    if config.ENVIRONMENT == 'jump':
        if timing =='during':
            extra.append(np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[1]))

    return extra