from __future__ import unicode_literals

from snips_nlu_metrics import Engine


def dummy_parsing_result(text, intent_name=None):
    return {
        "input": text,
        "intent": {
            "intentName": intent_name,
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


class KeyWordMatchingEngine(Engine):
    def __init__(self):
        self.fitted = False
        self.intents_list = []

    def fit(self, dataset):
        self.fitted = True
        self.intents_list = sorted(dataset["intents"])

    def parse(self, text, intents_filter=None):
        intent = None
        for intent_name in self.intents_list:
            if intent_name in text:
                intent = intent_name
                break
        if intents_filter is not None and intent not in intents_filter:
            intent = None
        return dummy_parsing_result(text, intent)


class MockEngineSegfault(Engine):
    def __init__(self):
        self.fitted = False

    def fit(self, dataset):
        self.fitted = True

    def parse(self, text):
        # Simulate a segmentation fault
        exit(139)
