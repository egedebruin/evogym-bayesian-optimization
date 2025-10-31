import numpy as np
import torch


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