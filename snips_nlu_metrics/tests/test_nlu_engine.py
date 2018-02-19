from __future__ import unicode_literals

import unittest

from snips_nlu_metrics.engine import (
    get_trained_nlu_engine, get_inference_nlu_engine)
from snips_nlu_metrics.tests.mock_engine import (
    MockTrainingEngine, MockInferenceEngine)


class TestNLUEngine(unittest.TestCase):
    def test_get_trained_engine_should_use_provided_engine_class(self):
        # Given
        _dataset = {
            "language": "en",
            "intents": {
                "intent1": {
                    "utterances": [
                        {"data": [{"text": "text1"}]},
                    ]
                },
                "intent2": {
                    "utterances": [
                        {"data": [{"text": "text2"}]},
                    ]
                },
            },
            "entities": {},
            "snips_nlu_version": "0.1.0"
        }

        # When
        engine = get_trained_nlu_engine(_dataset, MockTrainingEngine)

        # Then
        self.assertTrue(engine.fitted, 1)

    def test_get_inference_engine_should_use_provided_engine_class(self):
        # When
        inference_engine = get_inference_nlu_engine(dict(),
                                                    MockInferenceEngine)

        # Then
        self.assertIsInstance(inference_engine, MockInferenceEngine)
