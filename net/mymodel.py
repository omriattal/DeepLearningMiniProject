from abc import ABC, abstractmethod
from .extractor import Extractor


class MyModel(ABC):

    @abstractmethod
    def recurrence(self, xt, hiddens):
        pass
