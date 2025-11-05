from abc import ABC, abstractmethod

class Controller(ABC):
    
    @abstractmethod
    def control(self, sensor_inputs):
        pass

    def adjust_sensor_input(self, sensor_input):
        return sensor_input

    def post_action(self, sensor_input, normalized_sensor_input, next_sensor_input, normalized_next_sensor_input, reward, raw_action, buffer):
        return

    def post_rollout(self, last_sensor_input):
        return

    @staticmethod
    def get_parameters_from_args(args):
        return args