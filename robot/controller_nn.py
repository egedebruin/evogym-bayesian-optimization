import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot.controller import Controller

class RunningNorm:
    def __init__(self, epsilon=1e-8, min_var=1e-6):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        self.min_var = min_var

    def update(self, x, mask=None):
        x = np.asarray(x)
        if mask is not None:
            x = x[mask]
        else:
            x = x.reshape(-1)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.shape[0]
        if self.count == 0:
            self.mean = batch_mean
            self.var = max(batch_var, self.min_var)
            self.count = batch_count
            return
        # Welford merge
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta*delta * self.count * batch_count / tot_count
        new_var = max(M2 / tot_count, self.min_var)
        self.mean, self.var, self.count = new_mean, new_var, tot_count


    def normalize(self, x, mask=None):
        """
        Normalize with running stats.
        If mask is given, skip normalization where mask=False (e.g. padded zeros).
        """
        if isinstance(x, torch.Tensor):
            mean, std = torch.tensor(self.mean, device=x.device), torch.tensor(self.var**0.5 + self.epsilon, device=x.device)
        else:
            mean, std = self.mean, (self.var**0.5 + self.epsilon)

        if mask is not None:
            x_norm = x.copy() if isinstance(x, np.ndarray) else x.clone()
            x_norm[mask] = (x[mask] - mean) / std
            return x_norm
        else:
            return (x - mean) / std


class ControllerNN(Controller, nn.Module):

    def __init__(self, args, velocity_norm):
        super().__init__()

        self.policy_optimizer = None
        self.critic_optimizer = None
        self.gamma = 0.95

        def from_args(name):
            if name not in args:
                raise KeyError(f"Missing parameter '{name}' in args")
            return nn.Parameter(args[name].detach().clone())

        # Policy network weights/biases
        self.hidden_weights = from_args('hidden_weights')
        self.hidden_biases = from_args('hidden_biases')
        self.output_weights = from_args('output_weights')
        self.output_biases = from_args('output_biases')

        # Critic weights
        self.critic_hidden_weights = from_args('critic_hidden_weights')
        self.critic_hidden_biases = from_args('critic_hidden_biases')
        self.critic_hidden2_weights = from_args('critic_hidden2_weights')
        self.critic_hidden2_biases = from_args('critic_hidden2_biases')

        self.critic_output_weights = from_args('critic_output_weights')
        self.critic_output_biases = from_args('critic_output_biases')

        # Target critic weights (extra layer too)
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

        self.set_optimizers()
        self.velocity_indices = list(range(9, 27))
        self.training_mode = True
        self.velocity_norm = velocity_norm
        self.freeze_norm = False
        self.update_weights = False


    def set_optimizers(self, policy_lr=1e-4, critic_lr=1e-4):
        # Separate params for policy and critic
        policy_params = [self.hidden_weights, self.hidden_biases,
                         self.output_weights, self.output_biases]
        critic_params = [self.critic_hidden_weights, self.critic_hidden_biases,
                         self.critic_hidden2_weights, self.critic_hidden2_biases,
                         self.critic_output_weights, self.critic_output_biases]

        self.policy_optimizer = torch.optim.Adam(policy_params, lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

    def forward_critic(self, critic_input):
        h1 = F.relu(critic_input @ self.critic_hidden_weights + self.critic_hidden_biases)
        h2 = F.relu(h1 @ self.critic_hidden2_weights + self.critic_hidden2_biases)
        q_value = h2 @ self.critic_output_weights + self.critic_output_biases
        return q_value.squeeze(-1)

    def forward_target_critic(self, critic_input):
        h1 = F.relu(critic_input @ self.target_critic_hidden_weights + self.target_critic_hidden_biases)
        h2 = F.relu(h1 @ self.target_critic_hidden2_weights + self.target_critic_hidden2_biases)
        q_value = h2 @ self.target_critic_output_weights + self.target_critic_output_biases
        return q_value.squeeze(-1)

    def soft_update_target(self, tau=0.01):
        for target, source in [
            (self.target_critic_hidden_weights, self.critic_hidden_weights),
            (self.target_critic_hidden_biases, self.critic_hidden_biases),
            (self.target_critic_hidden2_weights, self.critic_hidden2_weights),
            (self.target_critic_hidden2_biases, self.critic_hidden2_biases),
            (self.target_critic_output_weights, self.critic_output_weights),
            (self.target_critic_output_biases, self.critic_output_biases),
        ]:
            target.data.copy_(tau * source.data + (1.0 - tau) * target.data)

    def control(self, sensor_inputs):
        """
        sensor_inputs: list of length M, each element = list/array of input_dim features
        """
        with torch.no_grad():
            processed_inputs = []
            vel_indices = list(range(9, 27))

            for sensors in sensor_inputs:
                sensors = np.array(sensors, dtype=np.float32)

                # Just normalize using *existing* stats, never update here
                vels = sensors[vel_indices]
                mask = (vels != 0)
                vels_normed = self.velocity_norm.normalize(vels, mask=mask)
                sensors[vel_indices] = vels_normed

                processed_inputs.append(sensors)

            sensor_tensor = torch.tensor(np.array(processed_inputs), dtype=torch.float32,
                                         device=self.hidden_weights.device)

            hidden = F.relu(sensor_tensor @ self.hidden_weights + self.hidden_biases)
            raw_output = hidden @ self.output_weights + self.output_biases
            raw_actions = torch.sigmoid(raw_output).cpu().numpy()

        return raw_actions

    def update_norm(self, sensor_input, next_sensor_input):
        device = self.hidden_weights.device
        vel_indices = list(range(9, 27))

        # Convert list of arrays to single NumPy array first
        sensor_input = torch.as_tensor(np.array(sensor_input, dtype=np.float32), device=device)
        next_sensor_input = torch.as_tensor(np.array(next_sensor_input, dtype=np.float32), device=device)

        # sensor_input: (M, input_dim) for one transition
        vels = sensor_input[:, vel_indices].cpu().numpy()
        mask = (vels != 0)
        self.velocity_norm.update(vels, mask=mask)

        next_vels = next_sensor_input[:, vel_indices].cpu().numpy()
        next_mask = (next_vels != 0)
        self.velocity_norm.update(next_vels, mask=next_mask)

    def update(self, sensor_inputs, raw_actions, rewards, next_sensor_inputs, do_policy_update):
        device = self.hidden_weights.device

        sensor_inputs = torch.as_tensor(sensor_inputs, dtype=torch.float32, device=device)
        raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_sensor_inputs = torch.as_tensor(next_sensor_inputs, dtype=torch.float32, device=device)

        # --- Normalize using current stats (no updates) ---
        vel_indices = list(range(9, 27))
        vels = sensor_inputs[:, :, vel_indices].cpu().numpy()
        mask = (vels != 0)
        vels_normed = self.velocity_norm.normalize(vels, mask=mask)
        sensor_inputs[:, :, vel_indices] = torch.tensor(vels_normed, dtype=torch.float32, device=device)

        next_vels = next_sensor_inputs[:, :, vel_indices].cpu().numpy()
        next_mask = (next_vels != 0)
        next_vels_normed = self.velocity_norm.normalize(next_vels, mask=next_mask)
        next_sensor_inputs[:, :, vel_indices] = torch.tensor(next_vels_normed, dtype=torch.float32, device=device)

        self.update_critic(sensor_inputs, raw_actions, rewards, next_sensor_inputs)
        if do_policy_update:
            self.update_policy(sensor_inputs)

    def update_critic(self, sensor_inputs, raw_actions, rewards, next_sensor_inputs):
        """
        sensor_inputs: (B, M, input_dim)
        raw_actions: (B, M, output_dim)
        rewards: (B, 1)
        next_sensor_inputs: (B, M, input_dim)
        """
        device = self.hidden_weights.device
        B, M, input_dim = sensor_inputs.shape

        # Convert to tensors
        sensor_inputs = torch.as_tensor(sensor_inputs, dtype=torch.float32, device=device)
        raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_sensor_inputs = torch.as_tensor(next_sensor_inputs, dtype=torch.float32, device=device)

        # Concatenate all actuator states and actions per batch
        critic_inputs = torch.cat([sensor_inputs.reshape(B, -1), raw_actions.reshape(B, -1)], dim=-1)
        next_policy_actions = []

        with torch.no_grad():
            for m in range(M):
                next_hidden = F.relu(next_sensor_inputs[:, m, :] @ self.hidden_weights + self.hidden_biases)
                next_a = torch.sigmoid(next_hidden @ self.output_weights + self.output_biases)
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

    def clip_policy_params(self):
        # Hidden weights: [-1, 1]
        self.hidden_weights.data.clamp_(-1.0, 1.0)

        # Hidden biases: [-0.1, 0.1]
        self.hidden_biases.data.clamp_(-0.1, 0.1)

        # Output weights: [-2, 2]
        self.output_weights.data.clamp_(-2.0, 2.0)

        # Output biases: [-1, 1]
        self.output_biases.data.clamp_(-1.0, 1.0)

    def update_policy(self, sensor_inputs):
        device = self.hidden_weights.device
        B, M, input_dim = sensor_inputs.shape

        sensor_inputs = torch.as_tensor(sensor_inputs, dtype=torch.float32, device=device)
        all_actions = []

        for m in range(M):
            hidden = F.relu(sensor_inputs[:, m, :] @ self.hidden_weights + self.hidden_biases)
            a = torch.sigmoid(hidden @ self.output_weights + self.output_biases)
            all_actions.append(a)
        all_actions = torch.cat(all_actions, dim=-1)  # (B, M*output_dim)

        # Concatenate states and actions for global critic
        critic_inputs = torch.cat([sensor_inputs.reshape(B, -1), all_actions], dim=-1)
        policy_loss = -self.forward_critic(critic_inputs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_([
            self.hidden_weights, self.hidden_biases,
            self.output_weights, self.output_biases
        ], max_norm=1.0)
        self.policy_optimizer.step()
        self.clip_policy_params()