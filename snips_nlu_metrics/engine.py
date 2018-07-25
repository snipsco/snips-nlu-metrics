from __future__ import unicode_literals

from abc import ABCMeta, abstractmethod
from builtins import object

from future.utils import with_metaclass


class Engine(with_metaclass(ABCMeta, object)):
    """Abstract class which represents an engine that can be used in the
    metrics API. All engine classes must inherit from `Engine`.
    """

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def parse(self, text):
        pass
