import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforcement_learning.rl import RL


class DDPG(RL):

    def __init__(self, num_actuators):
        self.gamma = 0.95
        self.policy_lr = 1e-4
        self.critic_lr = 1e-4
        self.tau = 0.01

        super().__init__(num_actuators)

        self.target_critic_hidden_weights = None
        self.target_critic_hidden_biases = None
        self.target_critic_hidden2_weights = None
        self.target_critic_hidden2_biases = None
        self.target_critic_output_weights = None
        self.target_critic_output_biases = None

        self.set_target_critic_parameters()


    def set_target_critic_parameters(self):
        self.target_critic_hidden_weights = nn.Parameter(self.critic_hidden_weights.clone().detach(),
                                                         requires_grad=False)
        self.target_critic_hidden_biases = nn.Parameter(self.critic_hidden_biases.clone().detach(),
                                                        requires_grad=False)
        self.target_critic_hidden2_weights = nn.Parameter(self.critic_hidden2_weights.clone().detach(),
                                                          requires_grad=False)
        self.target_critic_hidden2_biases = nn.Parameter(self.critic_hidden2_biases.clone().detach(),
                                                         requires_grad=False)
        self.target_critic_output_weights = nn.Parameter(self.critic_output_weights.clone().detach(),
                                                         requires_grad=False)
        self.target_critic_output_biases = nn.Parameter(self.critic_output_biases.clone().detach(),
                                                        requires_grad=False)

    def forward_target_critic(self, critic_input):
        h1 = F.relu(critic_input @ self.target_critic_hidden_weights + self.target_critic_hidden_biases)
        h2 = F.relu(h1 @ self.target_critic_hidden2_weights + self.target_critic_hidden2_biases)
        q_value = h2 @ self.target_critic_output_weights + self.target_critic_output_biases
        return q_value.squeeze(-1)

    def set_policy_optimizer(self, policy_weights, policy_lr):
        super().set_policy_optimizer(list(policy_weights.values()), policy_lr)

    def post_action(self, policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer):
        buffer.append((normalized_sensor_input, raw_action, reward, normalized_next_sensor_input))
        self.update_norm(sensor_input, next_sensor_input)

        self.update(policy_weights, buffer)

    def control(self, sensor_input, policy_weights):
        """
        sensor_inputs: list of length M, each element = list/array of input_dim features
        """
        with torch.no_grad():
            sensor_tensor = torch.tensor(np.array(sensor_input), dtype=torch.float32)

            hidden = F.relu(sensor_tensor @ policy_weights['hidden_weights'] + policy_weights['hidden_biases'])
            raw_output = hidden @ policy_weights['output_weights'] + policy_weights['output_biases']
            raw_action = torch.sigmoid(raw_output).cpu().numpy()

        return raw_action

    def get_input_size(self, num_actuators, policy_input, policy_output):
        return num_actuators * (policy_input + policy_output)

    def update(self, policy_weights, buffer):
        # Make sure buffer has enough samples
        batch_size = min(64, len(buffer))

        # Sample without replacement
        batch = random.sample(buffer, batch_size)

        # Unpack batch into arrays
        states, actions, rewards, next_states = zip(*batch)
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.stack(next_states)

        sensor_inputs = torch.as_tensor(states, dtype=torch.float32)
        raw_actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_sensor_inputs = torch.as_tensor(next_states, dtype=torch.float32)

        if self.do_update_critic:
            for i in range(5):
                self.update_critic(policy_weights, sensor_inputs, raw_actions, rewards, next_sensor_inputs)
        if self.do_update_policy:
            self.update_policy(policy_weights, sensor_inputs)

    def update_critic(self, policy_weights, sensor_inputs, raw_actions, rewards, next_sensor_inputs):
        """
        sensor_inputs: (B, M, input_dim)
        raw_actions: (B, M, output_dim)
        rewards: (B, 1)
        next_sensor_inputs: (B, M, input_dim)
        """
        B, M, input_dim = sensor_inputs.shape

        # Concatenate all actuator states and actions per batch
        critic_inputs = torch.cat([sensor_inputs.reshape(B, -1), raw_actions.reshape(B, -1)], dim=-1)
        next_policy_actions = []

        with torch.no_grad():
            for m in range(M):
                next_hidden = F.relu(next_sensor_inputs[:, m, :] @ policy_weights['hidden_weights'] + policy_weights['hidden_biases'])
                next_a = torch.sigmoid(next_hidden @ policy_weights['output_weights'] + policy_weights['output_biases'])
                next_policy_actions.append(next_a)
            next_policy_actions = torch.cat(next_policy_actions, dim=-1)  # (B, M*output_dim)
            next_critic_inputs = torch.cat([next_sensor_inputs.reshape(B, -1), next_policy_actions], dim=-1)
            target_q = rewards.squeeze(-1) + self.gamma * self.forward_target_critic(next_critic_inputs)

        # Predicted Q
        predicted_q = self.forward_critic(critic_inputs)
        critic_loss = F.mse_loss(predicted_q, target_q)

        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_([
            self.critic_hidden_weights, self.critic_hidden_biases,
            self.critic_hidden2_weights, self.critic_hidden2_biases,
            self.critic_output_weights, self.critic_output_biases
        ], max_norm=1.0)

        self.critic_optimizer.step()
        self.soft_update_target()

    def soft_update_target(self):
        for target, source in [
            (self.target_critic_hidden_weights, self.critic_hidden_weights),
            (self.target_critic_hidden_biases, self.critic_hidden_biases),
            (self.target_critic_hidden2_weights, self.critic_hidden2_weights),
            (self.target_critic_hidden2_biases, self.critic_hidden2_biases),
            (self.target_critic_output_weights, self.critic_output_weights),
            (self.target_critic_output_biases, self.critic_output_biases),
        ]:
            target.data.copy_(self.tau * source.data + (1.0 - self.tau) * target.data)

    def update_policy(self, policy_weights, sensor_inputs):
        B, M, input_dim = sensor_inputs.shape

        all_actions = []

        for m in range(M):
            hidden = F.relu(sensor_inputs[:, m, :] @ policy_weights['hidden_weights'] + policy_weights['hidden_biases'])
            a = torch.sigmoid(hidden @ policy_weights['output_weights'] + policy_weights['output_biases'])
            all_actions.append(a)
        all_actions = torch.cat(all_actions, dim=-1)  # (B, M*output_dim)

        # Concatenate states and actions for global critic
        critic_inputs = torch.cat([sensor_inputs.reshape(B, -1), all_actions], dim=-1)
        policy_loss = -self.forward_critic(critic_inputs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_([
            policy_weights['hidden_weights'], policy_weights['hidden_biases'],
            policy_weights['output_weights'], policy_weights['output_biases']
        ], max_norm=1.0)
        self.policy_optimizer.step()
        self.clip_policy_params(policy_weights)