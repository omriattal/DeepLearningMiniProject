from abc import ABC, abstractmethod
from .extractor import Extractor
from torch.nn import Module


class MyModel(ABC, Extractor, Module):

    @abstractmethod
    def recurrence(self, xt, hiddens):
        pass
