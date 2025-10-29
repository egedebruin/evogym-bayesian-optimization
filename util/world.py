import random

import numpy as np
import os

import torch
from evogym import EvoWorld, EvoSim, EvoViewer, utils, WorldObject

from configs import config

def build_world(robot_structure, rng):
    env = get_environment()
    world = EvoWorld.from_json(os.path.join('worlds', env))
    x, y = start_position_robot()
    world.add_from_array(
        name='robot',
        structure=robot_structure,
        x=x,
        y=y,
        connections=utils.get_full_connectivity(robot_structure)
    )

    world = add_extra_attributes(world, rng)
    EvoSim._has_displayed_version = True
    sim = EvoSim(world)
    viewer = EvoViewer(sim)
    tracker(viewer)

    return sim, viewer

def run_simulator(sim, controller, sensors, viewer, simulator_length, headless, update_critic, update_policy, update_norm, rl_type, transition_buffer):
    sim.reset()
    extra_metrics = []
    start_position = sim.object_pos_at_time(sim.get_time(), 'robot')
    extra_metrics = extra_metrics_for_objective_value('before', sim, extra_metrics)

    previous_position = start_position
    sensor_input = None
    raw_action = None

    if rl_type == 'PPO':
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []

    for simulation_step in range(simulator_length):

        # ----- During phase metrics -----
        if simulation_step > 50:
            extra_metrics = extra_metrics_for_objective_value('during', sim, extra_metrics)

        # ----- Action step -----
        if simulation_step % 5 == 0:
            sensor_input = sensors.get_input_from_sensors(sim)
            raw_action, extra_args = controller.control(sensor_input)
            adjusted_action = raw_action + 0.6

            sim.set_action('robot', adjusted_action)
            previous_position = sim.object_pos_at_time(sim.get_time(), 'robot')

            if rl_type == 'PPO':
                observations.append(extra_args['obs'])
                actions.append(raw_action)
                log_probs.append(extra_args['log_prob'])
                values.append(extra_args['value'])

        # ----- Learning step -----
        if simulation_step % 5 == 4:
            reward = (
                np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[0])
                - np.mean(previous_position[0])
            ) * 10

            # Store transition
            next_sensor_input = sensors.get_input_from_sensors(sim)
            if transition_buffer is not None:
                transition_buffer.append((sensor_input, raw_action, reward, next_sensor_input))

            if update_norm:
                controller.update_norm(sensor_input, next_sensor_input)

            if rl_type == 'DDPG':
                if update_critic:
                    for i in range(4):
                        # Make sure buffer has enough samples
                        batch_size = min(64, len(transition_buffer))

                        # Sample without replacement
                        batch = random.sample(transition_buffer, batch_size)

                        # Unpack batch into arrays
                        states, actions, rewards, next_states = zip(*batch)
                        states = np.stack(states)
                        actions = np.stack(actions)
                        rewards = np.array(rewards).reshape(-1, 1)
                        next_states = np.stack(next_states)

                        controller.update(states, actions, rewards, next_states, False)
                if update_policy:
                    # Make sure buffer has enough samples
                    batch_size = min(64, len(transition_buffer))

                    # Sample without replacement
                    batch = random.sample(transition_buffer, batch_size)

                    # Unpack batch into arrays
                    states, actions, rewards, next_states = zip(*batch)
                    states = np.stack(states)
                    actions = np.stack(actions)
                    rewards = np.array(rewards).reshape(-1, 1)
                    next_states = np.stack(next_states)
                    controller.update(states, actions, rewards, next_states, True)
            elif rl_type == 'PPO':
                rewards.append(reward)

        # Step simulation
        sim.step()

        # Render if not headless
        if not headless:
            viewer.render('screen')

    if rl_type == 'PPO' and update_policy:
        # 1) Stack into numpy first (preserve per-timestep structure)
        obs_np = np.stack(observations)  # [T, M, input_dim]
        acts_np = np.stack(actions)  # [T, M, action_dim] (if scalar actions -> [T, M, 1])
        logp_np = np.stack(log_probs)  # [T, M]
        vals_np = np.stack(values)  # [T] or [T, 1] depending on what you stored

        # 2) Convert to torch tensors on the correct device
        device = controller.hidden_weights.device
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)  # [T, M, input_dim]
        act_tensor = torch.tensor(acts_np, dtype=torch.float32, device=device)  # [T, M, action_dim]
        logp_tensor = torch.tensor(logp_np, dtype=torch.float32, device=device)  # [T, M]
        rew_tensor = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float32, device=device)  # [T]
        val_tensor = torch.tensor(vals_np, dtype=torch.float32, device=device).squeeze(-1)  # [T]
        logp_per_timestep = logp_tensor.sum(dim=1)  # [T]

        for i in range(5):
            # print(f"EPOCH {i+1}/5")
            controller.ppo_update(
                obs=obs_tensor,
                actions=act_tensor,
                log_probs_old=logp_per_timestep,
                rewards=rew_tensor,
                values=val_tensor,
                last_sensor_input=sensor_input
            )
            # print()

    end_position = sim.object_pos_at_time(sim.get_time(), 'robot')
    extra_metrics = extra_metrics_for_objective_value('after', sim, extra_metrics)
    return calculate_objective_value(start_position, end_position, extra_metrics)


# Below are Environment specific things
def start_position_robot():
    if config.ENVIRONMENT == 'catch':
        return 16, 1
    return 1, 1

def get_environment():
    if config.ENVIRONMENT == 'simple' or config.ENVIRONMENT == 'jump' or config.ENVIRONMENT == 'catch':
        env = 'simple_environment.json'
    elif config.ENVIRONMENT == 'climb':
        env = 'climb_environment.json'
    elif config.ENVIRONMENT == 'carry':
        env = 'carry_environment.json'
    elif config.ENVIRONMENT == 'rugged':
        env = 'rugged_environment.json'
    elif config.ENVIRONMENT == 'steps':
        env = 'steps_environment.json'
    else:
        raise ValueError(f"Environment {config.ENVIRONMENT} does not exist.")
    return env

def tracker(viewer):
    if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
        viewer.track_objects('package', 'robot')
    else:
        viewer.track_objects('robot')

def add_extra_attributes(world, rng):
    if config.ENVIRONMENT == 'catch':
        offset_x = int(rng.integers(-6, 5)) # NOTE: This is always the same per generation
        offset_y = int(rng.integers(0, 6))

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

def calculate_objective_value(start_position, end_position, extra_metrics):
    if config.ENVIRONMENT == 'simple' or config.ENVIRONMENT == 'rugged' or config.ENVIRONMENT == 'steps':
        return np.mean(end_position[0]) - np.mean(start_position[0])
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

def extra_metrics_for_objective_value(timing, sim, extra):
    if config.ENVIRONMENT == 'carry' or config.ENVIRONMENT == 'catch':
        if timing == 'before' or timing == 'after':
            extra.append(sim.object_pos_at_time(sim.get_time(), 'package'))

    if config.ENVIRONMENT == 'jump':
        if timing =='during':
            extra.append(np.mean(sim.object_pos_at_time(sim.get_time(), 'robot')[1]))

    return extra