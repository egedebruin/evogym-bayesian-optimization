import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot.controller import Controller

class ControllerNN(Controller, nn.Module):

    def __init__(self, args):
        super().__init__()

        self.policy_optimizer = None
        self.critic_optimizer = None
        self.gamma = 0.99

        # Policy network weights/biases (must exist)
        self.hidden_weights = nn.Parameter(torch.tensor(args['hidden_weights'], dtype=torch.float32))
        self.hidden_biases = nn.Parameter(torch.tensor(args['hidden_biases'], dtype=torch.float32))
        self.output_weights = nn.Parameter(torch.tensor(args['output_weights'], dtype=torch.float32))
        self.output_biases = nn.Parameter(torch.tensor(args['output_biases'], dtype=torch.float32))

        # Dimensions for critic layers based on policy shapes
        input_dim = args['hidden_weights'].shape[0]  # sensor input dim
        hidden_dim = args['hidden_weights'].shape[1]  # policy hidden layer size
        output_dim = args['output_weights'].shape[1]  # policy output dim (action dim)
        critic_input_dim = input_dim + output_dim  # critic input = state + action

        # Helper function to convert to nn.Parameter if exists else None
        def param_or_none(key):
            if key in args:
                return nn.Parameter(torch.tensor(args[key], dtype=torch.float32))
            else:
                return None

        # Critic weights or None if not in args
        self.critic_hidden_weights = param_or_none('critic_hidden_weights')
        self.critic_hidden_biases = param_or_none('critic_hidden_biases')
        self.critic_output_weights = param_or_none('critic_output_weights')
        self.critic_output_biases = param_or_none('critic_output_biases')

        # Initialize randomly if not provided
        if self.critic_hidden_weights is None:
            self.critic_hidden_weights = nn.Parameter(torch.randn(critic_input_dim, hidden_dim) * 0.1)
        if self.critic_hidden_biases is None:
            self.critic_hidden_biases = nn.Parameter(torch.zeros(hidden_dim))
        if self.critic_output_weights is None:
            self.critic_output_weights = nn.Parameter(torch.randn(hidden_dim, 1) * 0.1)
        if self.critic_output_biases is None:
            self.critic_output_biases = nn.Parameter(torch.zeros(1))

        self.set_optimizers()

    def set_optimizers(self, policy_lr=1e-3, critic_lr=1e-3):
        # Separate params for policy and critic
        policy_params = [self.hidden_weights, self.hidden_biases,
                         self.output_weights, self.output_biases]
        critic_params = [self.critic_hidden_weights, self.critic_hidden_biases,
                         self.critic_output_weights, self.critic_output_biases]

        self.policy_optimizer = torch.optim.Adam(policy_params, lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

    def forward_critic(self, sensor_inputs, actions):
        critic_input = torch.cat([sensor_inputs, actions], dim=-1)
        hidden = F.relu(critic_input @ self.critic_hidden_weights + self.critic_hidden_biases)
        q_value = hidden @ self.critic_output_weights + self.critic_output_biases
        return q_value.squeeze(-1)

    def control(self, sensor_inputs):
        with torch.no_grad():
            sensor_inputs = torch.FloatTensor(np.array(sensor_inputs))
            hidden = F.relu(sensor_inputs @ self.hidden_weights + self.hidden_biases)
            raw_action = torch.sigmoid(hidden @ self.output_weights + self.output_biases).numpy()
            adjusted_action = raw_action + 0.6
        return adjusted_action, raw_action

    def update(self, sensor_inputs, raw_actions, rewards, next_sensor_inputs):
        # Convert inputs to tensors
        sensor_inputs = torch.FloatTensor(sensor_inputs)
        raw_actions = torch.FloatTensor(raw_actions)
        rewards = torch.FloatTensor(rewards)
        next_sensor_inputs = torch.FloatTensor(next_sensor_inputs)
        # --- Critic update ---
        with torch.no_grad():
            # Compute next actions from current policy
            next_actions = torch.sigmoid(
                F.relu(next_sensor_inputs @ self.hidden_weights + self.hidden_biases) @ self.output_weights + self.output_biases
            )
            target_q = rewards + self.gamma * self.forward_critic(next_sensor_inputs, next_actions)

        predicted_q = self.forward_critic(sensor_inputs, raw_actions)
        critic_loss = F.mse_loss(predicted_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy update ---
        predicted_actions = torch.sigmoid(
            F.relu(sensor_inputs @ self.hidden_weights + self.hidden_biases) @ self.output_weights + self.output_biases
        )
        policy_loss = -self.forward_critic(sensor_inputs, predicted_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return critic_loss.item(), policy_loss.item()