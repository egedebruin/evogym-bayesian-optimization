from abc import ABC, abstractmethod

class Controller(ABC):

    @abstractmethod
    def __init__(self, args):
        pass
    
    @abstractmethod
    def control(self, sensor_inputs):
        pass