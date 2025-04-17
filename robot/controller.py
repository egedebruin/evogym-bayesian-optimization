from abc import ABC, abstractmethod

class Controller(ABC):
    
    @abstractmethod
    def control(self, sensor_inputs):
        pass