from __future__ import unicode_literals

import unittest

from nlu_metrics.utils.nlu_engine_utils import get_inference_engine, \
    get_trained_engine


class TestNLUEngineUtils(unittest.TestCase):
    def test_get_trained_engine_should_use_provided_engine_class(self):
        # Given
        class TestTrainingEngine(object):
            def __init__(self, language):
                self.language = language
                self.fitted = False

            def fit(self, dataset, intents=None):
                self.fitted = True

        _dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "engineType": "regex",
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                        {"data": [{"text": "text3"}]}
                    ]
                },
                "intent2": {
                    "engineType": "regex",
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                    ]
                },
                "intent3": {
                    "engineType": "regex",
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                        {"data": [{"text": "text2"}]},
                        {"data": [{"text": "text3"}]},
                        {"data": [{"text": "text4"}]}
                    ]
                },
            },
            "entities": {},
            "snips_nlu_version": "0.1.0"
        }

        # When
        engine = get_trained_engine(_dataset, TestTrainingEngine)

        # Then
        self.assertTrue(engine.fitted, 1)
        self.assertEquals(engine.language, "en")

    def test_get_inference_engine_should_use_provided_engine_class(self):
        # Given
        class TestInferenceEngine(object):
            def __init__(self, language, data_zip):
                self.language = language

        # When
        inference_engine = get_inference_engine("en", dict(),
                                                TestInferenceEngine)

        # Then
        self.assertEquals(inference_engine.language, "en")
