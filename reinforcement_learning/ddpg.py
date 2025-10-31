import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforcement_learning.rl import RL


class DDPG(RL):

    def __init__(self, args, velocity_norm):
        super().__init__(velocity_norm)
        self.policy_optimizer = None
        self.critic_optimizer = None

        self.target_critic_hidden_weights = None
        self.target_critic_hidden_biases = None
        self.target_critic_hidden2_weights = None
        self.target_critic_hidden2_biases = None
        self.target_critic_output_weights = None
        self.target_critic_output_biases = None

        self.set_critic_parameters(args)
        self.gamma = 0.95
        self.policy_lr = 1e-4
        self.critic_lr = 1e-4
        self.tau = 0.01
        self.set_critic_optimizer(self.critic_lr)

    def set_critic_parameters(self, args):
        super().set_critic_parameters(args)

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

    def set_critic_optimizer(self, critic_lr):
        critic_params = [self.critic_hidden_weights, self.critic_hidden_biases,
                         self.critic_hidden2_weights, self.critic_hidden2_biases,
                         self.critic_output_weights, self.critic_output_biases]

        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

    def set_policy_optimizer(self, policy_weights, policy_lr):
        self.policy_optimizer = torch.optim.Adam(list(policy_weights.values()), lr=policy_lr)

    def forward_target_critic(self, critic_input):
        h1 = F.relu(critic_input @ self.target_critic_hidden_weights + self.target_critic_hidden_biases)
        h2 = F.relu(h1 @ self.target_critic_hidden2_weights + self.target_critic_hidden2_biases)
        q_value = h2 @ self.target_critic_output_weights + self.target_critic_output_biases
        return q_value.squeeze(-1)

    def post_action(self, policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer):
        buffer.append((normalized_sensor_input, raw_action, reward, normalized_next_sensor_input))
        self.update_norm(sensor_input, next_sensor_input)

        self.update(policy_weights, buffer)

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

        for i in range(5):
            self.update_critic(policy_weights, sensor_inputs, raw_actions, rewards, next_sensor_inputs)
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

    def clip_policy_params(self, policy_weights):
        # Hidden weights: [-1, 1]
        policy_weights['hidden_weights'].data.clamp_(-1.0, 1.0)

        # Hidden biases: [-0.1, 0.1]
        policy_weights['hidden_biases'].data.clamp_(-0.1, 0.1)

        # Output weights: [-2, 2]
        policy_weights['output_weights'].data.clamp_(-2.0, 2.0)

        # Output biases: [-1, 1]
        policy_weights['output_biases'].data.clamp_(-1.0, 1.0)