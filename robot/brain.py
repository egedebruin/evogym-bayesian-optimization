from abc import ABC, abstractmethod

class Brain(ABC):

    def __init__(self, max_size=-1, rng=None):
        if max_size > 0:
            self.random_brain(max_size, rng)

    @abstractmethod
    def random_brain(self, max_size, rng):
        pass

    @abstractmethod
    def replace_parameters(self, parameters_string):
        pass

    @abstractmethod
    def mutate(self, rng):
        pass

    @abstractmethod
    def to_next_point(self, actuator_indices):
        pass

    @abstractmethod
    def to_string(self):
        pass

    def update_experience_with_actuator_indices(self, experience, actuator_indices):
        return experience

    @staticmethod
    @abstractmethod
    def get_p_bounds(actuator_indices):
        pass

    @staticmethod
    @abstractmethod
    def next_point_to_controller_values(next_point, actuator_indices):
        pass