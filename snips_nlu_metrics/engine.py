from abc import ABCMeta, abstractmethod


class Engine(metaclass=ABCMeta):
    """Abstract class which represents an engine that can be used in the
    metrics API. All engine classes must inherit from `Engine`.
    """

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text):
        pass
