import numpy as np

from robot.controller import Controller

class ControllerSine(Controller):

    DT = 0.5

    def __init__(self, args):
        self.amplitudes = args['amplitudes']
        self.phase_offsets = args['phase_offsets']
        self.angular_offsets = args['angular_offsets']
        self.t = [0.0] * len(self.amplitudes)

    def control(self, sensor_inputs):
        result = []
        i = 0
        for amplitude, phase_offset, angular_offset in zip(self.amplitudes, self.phase_offsets, self.angular_offsets):
            target = np.clip(amplitude * np.sin(self.t[i] + phase_offset) + 1.1 + angular_offset, 0.6, 1.6)
            result.append(target)

            self.t[i] += ControllerSine.DT
            i += 1
        return np.array(result)