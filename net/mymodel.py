from abc import ABC, abstractmethod
from .gate_extractor import GateExtractor

"""
Used as an abstract class only
"""
class MyModel(ABC):

    @abstractmethod
    def my_forward_with_extraction(self, xt, hiddens):
        pass
