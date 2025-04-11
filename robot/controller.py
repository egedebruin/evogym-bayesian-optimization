import config
import numpy as np

class Controller:
    def __init__(self, amplitudes, phase_offsets, angular_offsets):
        self.amplitudes = amplitudes
        self.phase_offsets = phase_offsets
        self.angular_offsets = angular_offsets
        self.t = [0.0] * len(amplitudes)

    def control(self):
        result = []
        i = 0
        for amplitude, phase_offset, angular_offset in zip(self.amplitudes, self.phase_offsets, self.angular_offsets):
            target = np.clip(amplitude * np.sin(self.t[i] + phase_offset) + 1.1 + angular_offset, 0.6, 1.6)
            result.append(target)

            self.t[i] += config.CONTROLLER_DT
            i += 1
        return np.array(result)