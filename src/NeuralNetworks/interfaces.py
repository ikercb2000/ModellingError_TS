# Other Modules

from abc import ABC, abstractmethod

# Interface Data Handler

class IDataHandler(ABC):

    @abstractmethod
    def __str__(self, **kwargs):
        return "Data Handler"

# Interface Model Generator

class IModelGenerator(ABC):

    @abstractmethod
    def __str__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, **kwargs):
        pass