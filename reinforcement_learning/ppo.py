import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reinforcement_learning.rl import RL


class PPO(RL):
    def __init__(self, num_actuators):
        self.gamma = 0.95  # 0.95 -> 0.99, short-term -> long-term
        self.lam = 0.9  # 0.9 -> 0.99, stable gradients -> noisy gradients
        self.clip_eps = 0.5  # 0.1 -> 0.3, low changes -> high changes
        self.entropy_coef = 0.01  # 0.01 -> 0.1, fast convergence -> exploration
        self.value_coef = 0.2  # 0.1 -> 1.0, focus on policy -> focus on value
        self.max_grad_norm = 1.0  # 0.3 -> 1.0, lower gradients -> higher gradients
        self.critic_lr = 1e-2  # 1e-5 -> 3e-3, slow updates -> fast updates
        self.policy_lr = 1e-2  # 1e-5 -> 3e-3, slow updates -> fast updates
        self.init_log_std = -2 # 0 -> -5, high initial noise -> low initial noise

        super().__init__(num_actuators)

        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

        self.optimizer = None
        self.log_std = None

    def forward_policy(self, x, policy_weights):
        h = F.relu(x @ policy_weights['hidden_weights'] + policy_weights['hidden_biases'])
        mu = torch.sigmoid(h @ policy_weights['output_weights'] + policy_weights['output_biases'])
        std = torch.exp(self.log_std)
        return mu, std

    def get_action_and_value(self, obs, policy_weights):
        mu, std = self.forward_policy(obs, policy_weights)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        flat_obs = obs.reshape(obs.shape[0], -1)
        value = self.forward_critic(flat_obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions, policy_weights):
        mu, std = self.forward_policy(obs, policy_weights)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        flat_obs = obs.reshape(obs.shape[0], -1)
        values = self.forward_critic(flat_obs)
        return log_probs, entropy, values

    def compute_advantages(self, rewards, values, next_values):
        advantages = torch.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t].detach() - values[t].detach()
            advantages[t] = last_adv = delta + self.gamma * self.lam * last_adv
        returns = advantages + values
        return advantages, returns

    def set_policy_optimizer(self, policy_weights, policy_lr):
        self.log_std = nn.Parameter(torch.ones_like(policy_weights['output_biases']) * self.init_log_std)
        super().set_policy_optimizer(list(policy_weights.values()) + [self.log_std], policy_lr)

    def post_action(self, policy_weights, sensor_input, normalized_sensor_input, next_sensor_input,
                    normalized_next_sensor_input, reward, raw_action, buffer):
        super().post_action(policy_weights, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer)
        self.rewards.append(reward)

    def control(self, sensor_input, policy_weights):
        obs_tensor = torch.tensor(
            np.array(sensor_input, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob, value = self.get_action_and_value(obs_tensor, policy_weights)

        action_np = action_tensor.cpu().numpy().squeeze(0)  # [M, A]
        logp_np = log_prob.cpu().numpy().squeeze(0)  # [M]
        value_np = value.cpu().numpy().squeeze(0)  # scalar as [ ] -> becomes scalar if squeezed

        self.observations.append(obs_tensor.cpu().numpy().squeeze(0))
        self.actions.append(action_np)
        self.log_probs.append(logp_np)
        self.values.append(value_np)

        return action_np

    def get_input_size(self, num_actuators, policy_input, policy_output):
        return num_actuators * policy_input

    def post_rollout(self, last_sensor_input, policy_weights):
        # 1) Stack into numpy first (preserve per-timestep structure)
        obs_np = np.stack(self.observations)  # [T, M, input_dim]
        acts_np = np.stack(self.actions)  # [T, M, action_dim] (if scalar actions -> [T, M, 1])
        logp_np = np.stack(self.log_probs)  # [T, M]
        vals_np = np.stack(self.values)  # [T] or [T, 1] depending on what you stored

        # 2) Convert to torch tensors on the correct device
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32)  # [T, M, input_dim]
        act_tensor = torch.tensor(acts_np, dtype=torch.float32)  # [T, M, action_dim]
        logp_tensor = torch.tensor(logp_np, dtype=torch.float32)  # [T, M]
        rew_tensor = torch.tensor(np.array(self.rewards, dtype=np.float32), dtype=torch.float32)  # [T]
        val_tensor = torch.tensor(vals_np, dtype=torch.float32).squeeze(-1)  # [T]
        logp_per_timestep = logp_tensor.sum(dim=1)  # [T]

        for i in range(5):
            self.update(
                obs=obs_tensor,
                actions=act_tensor,
                log_probs_old=logp_per_timestep,
                rewards=rew_tensor,
                values=val_tensor,
                last_sensor_input=last_sensor_input,
                policy_weights=policy_weights
            )

        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

        return

    def update(self, obs, actions, log_probs_old, rewards, values, last_sensor_input, policy_weights, clip_eps=None):
        if clip_eps is None:
            clip_eps = self.clip_eps

        # --- Compute next value for advantage / return ---
        with torch.no_grad():
            next_obs = torch.tensor(np.array(last_sensor_input, dtype=np.float32)).unsqueeze(0)
            next_value = self.forward_critic(next_obs.reshape(1, -1))  # [1]
        if next_value.ndim == 0:
            next_value = next_value.unsqueeze(0)

        # --- Compute advantages and returns ---
        advantages, returns = self.compute_advantages(
            rewards=rewards,
            values=values,
            next_values=torch.cat([values[1:], next_value])  # [T]
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Evaluate current policy ---
        log_probs, entropy, values_pred = self.evaluate_actions(obs, actions, policy_weights)
        log_probs = log_probs.sum(dim=1)  # Sum over actuators per timestep
        entropy = entropy.mean(dim=1)  # Mean over actuators per timestep

        # --- Compute ratio ---
        diff = (log_probs - log_probs_old)
        ratio = diff.exp()

        # --- PPO surrogate loss ---
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # --- Value loss ---
        value_loss = F.mse_loss(values_pred.squeeze(-1), returns.detach())

        # --- Total loss ---
        loss = 0
        if self.do_update_critic:
            loss = loss + self.value_coef * value_loss
        if self.do_update_policy:
            loss = loss + policy_loss - self.entropy_coef * entropy.mean()

        # --- Total loss already computed as `loss` ---
        if self.do_update_critic:
            self.critic_optimizer.zero_grad()
        if self.do_update_policy:
            self.policy_optimizer.zero_grad()

        # Backprop
        if self.do_update_policy or self.do_update_critic:
            loss.backward()

        # Apply only the requested updates
        torch.nn.utils.clip_grad_norm_([
            policy_weights['hidden_weights'], policy_weights['hidden_biases'],
            policy_weights['output_weights'], policy_weights['output_biases']
        ], self.max_grad_norm)
        if self.do_update_critic:
            self.critic_optimizer.step()
        if self.do_update_policy:
            self.policy_optimizer.step()
            self.clip_policy_params(policy_weights)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "adv_mean": advantages.mean().item(),
            "adv_std": advantages.std().item(),
            "logp_mean": log_probs.mean().item(),
            "logp_std": log_probs.std().item()
        }