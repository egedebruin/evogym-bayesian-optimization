import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from robot.controller import Controller


class ControllerNNPPO(Controller, nn.Module):
    def __init__(self, args, velocity_norm):
        super().__init__()

        self.gamma = 0.99  # Shorter-term focus
        self.lam = 0.97  # Less smoothing, faster reaction
        self.clip_eps = 0.3  # Allow bigger updates
        self.entropy_coef = 0.05  # Focus on exploiting
        self.value_coef = 0.2  # Prioritize policy learning
        self.max_grad_norm = 0.5  # Allow stronger updates

        def from_args(name):
            if name not in args:
                raise KeyError(f"Missing parameter '{name}' in args")
            return nn.Parameter(args[name].detach().clone())

        # Shared structure from args
        self.hidden_weights = from_args("hidden_weights")
        self.hidden_biases = from_args("hidden_biases")
        self.output_weights = from_args("output_weights")
        self.output_biases = from_args("output_biases")

        # Critic weights (for V(s))
        self.critic_hidden_weights = from_args("critic_hidden_weights")
        self.critic_hidden_biases = from_args("critic_hidden_biases")
        self.critic_hidden2_weights = from_args("critic_hidden2_weights")
        self.critic_hidden2_biases = from_args("critic_hidden2_biases")
        self.critic_output_weights = from_args("critic_output_weights")
        self.critic_output_biases = from_args("critic_output_biases")

        self.velocity_norm = velocity_norm
        self.velocity_indices = list(range(9, 27))

        self.log_std = nn.Parameter(torch.ones_like(self.output_biases) * -3)

        self.set_optimizers()

    def set_optimizers(self, lr=1e-2, log_std_lr=1e-2):
        # collect all params except log_std
        params_except_log_std = [p for n, p in self.named_parameters() if n != 'log_std']
        self.optimizer = torch.optim.Adam([
            {'params': params_except_log_std, 'lr': lr},
            {'params': [self.log_std], 'lr': log_std_lr}
        ])

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

    def forward_policy(self, x):
        h = F.relu(x @ self.hidden_weights + self.hidden_biases)
        mu = torch.sigmoid(h @ self.output_weights + self.output_biases)
        std = torch.exp(self.log_std)
        return mu, std

    def forward_value(self, x):
        h1 = F.relu(x @ self.critic_hidden_weights + self.critic_hidden_biases)
        h2 = F.relu(h1 @ self.critic_hidden2_weights + self.critic_hidden2_biases)
        v = h2 @ self.critic_output_weights + self.critic_output_biases
        return v.squeeze(-1)

    def get_action_and_value(self, obs):
        mu, std = self.forward_policy(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = action.clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        flat_obs = obs.reshape(obs.shape[0], -1)
        value = self.forward_value(flat_obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mu, std = self.forward_policy(obs)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        flat_obs = obs.reshape(obs.shape[0], -1)
        values = self.forward_value(flat_obs)
        return log_probs, entropy, values

    def compute_advantages(self, rewards, values, next_values):
        advantages = torch.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.lam * last_adv
        returns = advantages + values
        return advantages, returns

    def ppo_update(self, obs, actions, log_probs_old, rewards, values, last_sensor_input, clip_eps=None):
        if clip_eps is None:
            clip_eps = self.clip_eps

        device = self.hidden_weights.device

        # --- Compute next value for advantage / return ---
        with torch.no_grad():
            next_obs = torch.tensor(np.array(last_sensor_input, dtype=np.float32), device=device).unsqueeze(0)
            next_value = self.forward_value(next_obs.reshape(1, -1)).to(values.device)  # [1]
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
        log_probs, entropy, values_pred = self.evaluate_actions(obs, actions)
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
        value_loss = F.mse_loss(values_pred.squeeze(-1), returns)

        # --- Total loss ---
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        # --- Total loss already computed as `loss` ---
        self.optimizer.zero_grad()
        loss.backward()

        # detect NaNs in grads
        nan_in_grad = any(torch.isnan(p.grad).any().item() for p in self.parameters() if p.grad is not None)
        if nan_in_grad:
            print("NaN in gradient, skipping optimizer.step()")
            self.optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {
            "loss": loss.item(),
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

    # --- Interface ---
    def control(self, sensor_input):
        obs_tensor = torch.tensor(
            np.array(sensor_input, dtype=np.float32), device=self.hidden_weights.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob, value = self.get_action_and_value(obs_tensor)

        action_np = action_tensor.cpu().numpy().squeeze(0)  # [M, A]
        logp_np = log_prob.cpu().numpy().squeeze(0)  # [M]
        value_np = value.cpu().numpy().squeeze(0)  # scalar as [ ] -> becomes scalar if squeezed

        # Return per-module actions (no batch) and extra info WITHOUT batch dimension
        return action_np, {
            'obs': obs_tensor.cpu().numpy().squeeze(0),  # [M, input_dim]
            'log_prob': logp_np,  # [M]
            'value': value_np  # scalar or array [ ] -> squeezed
        }
