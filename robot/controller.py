from abc import ABC, abstractmethod

class Controller(ABC):
    
    @abstractmethod
    def control(self, sensor_inputs):
        pass

    def adjust_sensor_input(self, sensor_input):
        return sensor_input

    def post_action(self, sim, sensor_input, normalized_sensor_input, raw_action, previous_position, sensors, buffer):
        return

    @staticmethod
    def get_parameters_from_args(args):
        return args