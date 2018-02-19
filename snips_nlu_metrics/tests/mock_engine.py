from __future__ import unicode_literals

from builtins import object

from snips_nlu_metrics import Engine


def dummy_parsing_result(text):
    return {
        "input": text,
        "intent": None,
        "slots": []
    }


class MockTrainingEngine(object):
    def __init__(self, config=None):
        self.training_config = config
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True

    def to_dict(self):
        return dict()


class MockInferenceEngine(object):
    def __init__(self, data_zip):
        pass

    def parse(self, text):
        return dummy_parsing_result(text)


class MockEngine(Engine):
    def __init__(self):
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True

    def parse(self, text):
        return dummy_parsing_result(text)
