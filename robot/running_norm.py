import numpy as np

class RunningNorm:
    def __init__(self, mean, var, mode, scale=1.0):
        self.mean = mean
        self.var = var
        self.mode = mode  # 'linear' or 'tanh'
        self.scale = scale

    def normalize(self, x, mask=None):
        """
        Normalize with running stats.
        If mask is given, skip normalization where mask=False (e.g. padded zeros).
        """
        mean, std = self.mean, np.sqrt(self.var)

        if mask is not None:
            x_norm = x.copy()
            x_norm[mask] = (x[mask] - mean) / std
        else:
            x_norm = (x - mean) / std

        if self.mode == 'linear':
            return x_norm
        elif self.mode == 'tanh':
            return np.tanh(x_norm / self.scale)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")