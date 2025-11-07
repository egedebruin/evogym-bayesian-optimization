import numpy as np
import torch


class RunningNorm:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def normalize(self, x, mask=None):
        """
        Normalize with running stats.
        If mask is given, skip normalization where mask=False (e.g. padded zeros).
        """
        if isinstance(x, torch.Tensor):
            mean, std = torch.tensor(self.mean, device=x.device), torch.tensor(self.var**0.5, device=x.device)
        else:
            mean, std = self.mean, (self.var**0.5)

        if mask is not None:
            x_norm = x.copy() if isinstance(x, np.ndarray) else x.clone()
            x_norm[mask] = (x[mask] - mean) / std
            return x_norm
        else:
            return (x - mean) / std