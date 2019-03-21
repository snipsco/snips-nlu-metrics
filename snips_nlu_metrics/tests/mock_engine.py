from __future__ import unicode_literals

from snips_nlu_metrics import Engine


def dummy_parsing_result(text):
    return {
        "input": text,
        "intent": {
            "intentName": None,
            "probability": 0.5
        },
        "slots": []
    }


class MockEngine(Engine):
    def __init__(self):
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True

    def parse(self, text):
        return dummy_parsing_result(text)


class MockEngineSegfault(Engine):
    def __init__(self):
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True

    def parse(self, text):
        # Simulate a segmentation fault
        exit(139)
