from abc import ABC, abstractmethod

"""
Used as an abstract class only
"""
class MyModel(ABC):

    @abstractmethod
    def my_forward_with_extraction(self, xt, hiddens):
        pass
